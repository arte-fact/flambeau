//! `embed_gather_q4_k_f16` correctness sweep: every gathered F16 element must
//! carry the same bits as the host load path, `flambeau-quant`
//! `dequantize_to_vec(Q4K)` followed by `f16::from_f32` (what
//! `upload_dequant_to_f16` uploads today).
//!
//! Synthetic tables: random blocks with finite F16 `d` / `dmin` drawn from the
//! whole bit range (subnormals and 65504 included), every row of a 248 320-row
//! table, prefill / verify batch sizes at hidden 5120, the decode-graph
//! `ids = ctx + 1` layout, and extreme blocks (zero / subnormal / max scales,
//! saturated 6-bit scales and nibbles). [`check_gguf_token_embd`] runs the same
//! comparison over a real GGUF's `token_embd.weight`.
//!
//! Backend-generic; emits `certs/<backend>/<arch>/embed_gather_q4_k_f16.json`.

#![cfg(any(feature = "hip", feature = "cuda"))]

use std::path::Path;

use anyhow::{anyhow, bail, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, LaunchCfg, Stream};
use flambeau_quant::{GgmlDType, GgufFile, QK_K};
use half::f16;

use crate::backend::{BenchBackend, LaunchArgs};
use crate::cert::{now_utc_iso8601, Cert, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, rig_for};

const STEM: &str = "embed_gather_q4_k_f16";
const ENTRY: &str = "flambeau_embed_gather_q4_k_f16";
const BLOCK_BYTES: usize = 144;
const TOL_FORMULA: &str =
    "bit-identical to host dequantize_to_vec(Q4K) + f16::from_f32; max_rel_err = number of differing F16 elements";

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn finite_f16_bits(&mut self) -> u16 {
        let b = self.next() as u16;
        if (b >> 10) & 0x1F == 0x1F {
            b & !0x4000
        } else {
            b
        }
    }
}

fn write_block(out: &mut [u8], d: u16, dmin: u16, scales: &[u8; 12], qs: &[u8; 128]) {
    out[0..2].copy_from_slice(&d.to_le_bytes());
    out[2..4].copy_from_slice(&dmin.to_le_bytes());
    out[4..16].copy_from_slice(scales);
    out[16..144].copy_from_slice(qs);
}

fn random_table(seed: u64, vocab: usize, hidden: usize) -> Vec<u8> {
    let mut rng = SplitMix(seed);
    let n_blocks = vocab * hidden / QK_K;
    let mut raw = vec![0u8; n_blocks * BLOCK_BYTES];
    for blk in raw.chunks_exact_mut(BLOCK_BYTES) {
        let d = rng.finite_f16_bits();
        let dmin = rng.finite_f16_bits();
        let mut scales = [0u8; 12];
        let mut qs = [0u8; 128];
        for b in scales.iter_mut().chain(qs.iter_mut()) {
            *b = rng.next() as u8;
        }
        write_block(blk, d, dmin, &scales, &qs);
    }
    raw
}

/// One 256-wide row per (d, dmin, pattern) combination.
fn extreme_table(seed: u64) -> (Vec<u8>, usize) {
    const SCALE_BITS: [u16; 9] = [0x0000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0x8001, 0xBC00, 0xFBFF];
    let mut rng = SplitMix(seed);
    let mut raw = Vec::new();
    let mut rows = 0;
    for &d in &SCALE_BITS {
        for &dmin in &SCALE_BITS {
            for pattern in 0..4 {
                let mut scales = [0u8; 12];
                let mut qs = [0u8; 128];
                match pattern {
                    0 => {
                        scales = [0xFF; 12];
                        qs = [0xFF; 128];
                    }
                    1 => scales = [0xFF; 12],
                    2 => {
                        qs = [0x0F; 128];
                        for b in &mut scales {
                            *b = rng.next() as u8;
                        }
                    }
                    _ => {
                        for b in scales.iter_mut().chain(qs.iter_mut()) {
                            *b = rng.next() as u8;
                        }
                    }
                }
                let mut blk = [0u8; BLOCK_BYTES];
                write_block(&mut blk, d, dmin, &scales, &qs);
                raw.extend_from_slice(&blk);
                rows += 1;
            }
        }
    }
    (raw, rows)
}

fn host_rows(raw: &[u8], hidden: usize, ids: &[i32]) -> Result<Vec<u16>> {
    let row_bytes = hidden / QK_K * BLOCK_BYTES;
    let mut out = Vec::with_capacity(ids.len() * hidden);
    for &id in ids {
        let id = id as usize;
        let row = &raw[id * row_bytes..(id + 1) * row_bytes];
        let f = flambeau_quant::dequant::dequantize_to_vec(GgmlDType::Q4K, row, hidden)?;
        out.extend(f.iter().map(|&v| f16::from_f32(v).to_bits()));
    }
    Ok(out)
}

#[allow(clippy::undocumented_unsafe_blocks)]
fn device_rows<B: BenchBackend>(
    dev: &B::Device,
    kernel: &B::Kernel<'_>,
    d_table: DevicePtr,
    ids_buf: &[i32],
    ids_offset: usize,
    n_rows: usize,
    hidden: usize,
) -> Result<Vec<u16>> {
    let d_ids = alloc_and_upload(dev, ids_buf);
    let dst_bytes = n_rows * hidden * 2;
    let d_dst = dev.alloc(dst_bytes)?;
    let stream = dev.default_stream();
    let e = d_table.as_usize() as u64;
    let d = d_dst.as_usize() as u64;
    let i = d_ids.offset_bytes(ids_offset * 4).as_usize() as u64;
    let hidden_i = hidden as i32;
    let mut a = <B::Args<'_>>::default();
    a.push_arg(&e);
    a.push_arg(&d);
    a.push_arg(&i);
    a.push_arg(&hidden_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (hidden / QK_K) as u32, 1),
        block: (QK_K as u32, 1, 1),
        shared_bytes: 0,
    };
    B::launch(kernel, stream, cfg, a)?;
    stream.synchronize()?;
    let mut got = vec![0u16; n_rows * hidden];
    unsafe {
        dev.memcpy_async(stream, CopyDirection::DeviceToHost, DevicePtr(got.as_mut_ptr() as usize), d_dst, dst_bytes)?;
    }
    stream.synchronize()?;
    unsafe {
        dev.dealloc(d_ids, ids_buf.len() * 4)?;
        dev.dealloc(d_dst, dst_bytes)?;
    }
    Ok(got)
}

fn mismatches(got: &[u16], reference: &[u16]) -> usize {
    got.iter().zip(reference).filter(|(g, r)| g != r).count()
}

struct Case {
    vocab: usize,
    hidden: usize,
    n_rows: usize,
    decode_ctx: bool,
    all_rows: bool,
}

const CASES: &[Case] = &[
    Case { vocab: 248_320, hidden: 256, n_rows: 248_320, decode_ctx: false, all_rows: true },
    Case { vocab: 4096, hidden: 5120, n_rows: 1, decode_ctx: false, all_rows: false },
    Case { vocab: 4096, hidden: 5120, n_rows: 1, decode_ctx: true, all_rows: false },
    Case { vocab: 4096, hidden: 5120, n_rows: 8, decode_ctx: false, all_rows: false },
    Case { vocab: 4096, hidden: 5120, n_rows: 9, decode_ctx: false, all_rows: false },
    Case { vocab: 4096, hidden: 5120, n_rows: 513, decode_ctx: false, all_rows: false },
    Case { vocab: 1024, hidden: 2816, n_rows: 64, decode_ctx: false, all_rows: false },
];

/// Backend-generic entry point.
/// # Errors
/// Propagates device / launch / cert-write failures.
pub fn run_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image(STEM).ok_or_else(|| anyhow!("{STEM} not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, ENTRY)?;
    let pmc = B::attributes(&kernel)?;

    let mut results = Vec::new();
    for case in CASES {
        let seed = 0x0E4B_ED00 ^ ((case.vocab as u64) << 20) ^ ((case.hidden as u64) << 8) ^ case.n_rows as u64;
        let raw = random_table(seed, case.vocab, case.hidden);
        let mut rng = SplitMix(seed ^ 0x1D5);
        let ids: Vec<i32> = if case.all_rows {
            (0..case.n_rows as i32).collect()
        } else {
            (0..case.n_rows)
                .map(|r| match r {
                    0 => case.vocab as i32 - 1,
                    1 => 0,
                    _ => (rng.next() % case.vocab as u64) as i32,
                })
                .collect()
        };
        let (ids_buf, offset) = if case.decode_ctx {
            (vec![1234, ids[0]], 1)
        } else {
            (ids.clone(), 0)
        };
        let d_table = alloc_and_upload(&dev, &raw);
        let got = device_rows::<B>(&dev, &kernel, d_table, &ids_buf, offset, case.n_rows, case.hidden)?;
        // SAFETY: `d_table` was allocated with `raw.len()` bytes above.
        unsafe { dev.dealloc(d_table, raw.len())? };
        let reference = host_rows(&raw, case.hidden, &ids)?;
        let bad = mismatches(&got, &reference);
        results.push(ShapeResult {
            m: case.n_rows,
            k: case.vocab,
            n: case.hidden,
            seed,
            max_rel_err: bad as f32,
            tolerance: 0.0,
            pass: bad == 0,
        });
    }

    let seed = 0x000E_4BE7;
    let (raw, rows) = extreme_table(seed);
    let ids: Vec<i32> = (0..rows as i32).collect();
    let d_table = alloc_and_upload(&dev, &raw);
    let got = device_rows::<B>(&dev, &kernel, d_table, &ids, 0, rows, QK_K)?;
    // SAFETY: `d_table` was allocated with `raw.len()` bytes above.
    unsafe { dev.dealloc(d_table, raw.len())? };
    let bad = mismatches(&got, &host_rows(&raw, QK_K, &ids)?);
    results.push(ShapeResult {
        m: rows,
        k: rows,
        n: QK_K,
        seed,
        max_rel_err: bad as f32,
        tolerance: 0.0,
        pass: bad == 0,
    });

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: STEM.to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "embed".to_string(),
        dtype_weight: "Q4_K".to_string(),
        dtype_activation: "I32".to_string(),
        tolerance_formula: TOL_FORMULA.to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

/// Gathers every row of `token_embd.weight` (Q4_K) from the GGUF at `path`
/// on device 0 and compares it with the host load path. Returns
/// `(rows checked, differing F16 elements)`.
/// # Errors
/// Fails when the tensor is missing or not Q4_K, or on device errors.
pub fn check_gguf_token_embd<B: BenchBackend>(path: &Path) -> Result<(usize, usize)> {
    const CHUNK: usize = 4096;
    let file = GgufFile::open(path)?;
    let info = file.info("token_embd.weight")?;
    if info.dtype != GgmlDType::Q4K || info.dims.len() != 2 {
        bail!("token_embd.weight: expected 2-D Q4_K, got {:?} {:?}", info.dtype, info.dims);
    }
    let hidden = info.dims[0] as usize;
    let vocab = info.dims[1] as usize;
    let raw = file.tensor_raw("token_embd.weight")?;

    let dev = B::new_device(0)?;
    let image = B::image(STEM).ok_or_else(|| anyhow!("{STEM} not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, ENTRY)?;
    let d_table = alloc_and_upload(&dev, raw);
    let mut bad = 0;
    for start in (0..vocab).step_by(CHUNK) {
        let ids: Vec<i32> = (start as i32..(start + CHUNK).min(vocab) as i32).collect();
        let got = device_rows::<B>(&dev, &kernel, d_table, &ids, 0, ids.len(), hidden)?;
        bad += mismatches(&got, &host_rows(raw, hidden, &ids)?);
    }
    // SAFETY: `d_table` was allocated with `raw.len()` bytes above.
    unsafe { dev.dealloc(d_table, raw.len())? };
    Ok((vocab, bad))
}
