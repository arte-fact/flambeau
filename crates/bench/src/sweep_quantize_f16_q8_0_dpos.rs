//! Device-position F16 -> Q8_0 quantize-and-append correctness sweep.
//! The `_dpos` kernel reads the destination cache row from a device buffer
//! (`pos_ptr[0]`) instead of a host-folded destination pointer, so a captured
//! decode graph advances the Q8 KV write cursor with no rebind. Covers both the
//! absolute (ring_depth == 0) and ring-wrapped write-position cases across the
//! gemma-4 KV widths. Read back at the on-device-computed row and checked
//! against the F32 source within Q8_0 quant tolerance — validates both the
//! quantization and the row placement.
//!
//! Backend-generic; emits the CUDA (sm_86) cert. CUDA-only kernel.

#![cfg(any(feature = "hip", feature = "cuda"))]

use std::path::Path;

use anyhow::{anyhow, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, LaunchCfg, Stream};
use flambeau_quant::{BlockQ8_0, QK8_0};
use half::f16;

use crate::backend::{BenchBackend, LaunchArgs};
use crate::cert::{now_utc_iso8601, Cert, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig_for, seeded_f32_range};

const TOL: f32 = 2e-2;
const TOL_FORMULA: &str = "|err| <= 2e-2 * max(|ref|, 1)  (Q8_0 quant round-trip)";
const Q8_0_BLOCK_BYTES: usize = 34;

struct Case {
    /// KV width = one decode row of F16 elements.
    kv_width: usize,
    position: usize,
    /// Ring slab depth in rows (0 = absolute write position).
    ring_depth: usize,
}

const CASES: &[Case] = &[
    Case { kv_width: 256, position: 300, ring_depth: 0 },
    Case { kv_width: 512, position: 511, ring_depth: 0 },
    Case { kv_width: 2048, position: 1641, ring_depth: 0 },
    // Ring-wrapped: write lands at position % ring_depth.
    Case { kv_width: 512, position: 4000, ring_depth: 1152 },
    Case { kv_width: 512, position: 1200, ring_depth: 1152 },
];

fn dequantize_row_q8_0(xs: &[BlockQ8_0]) -> Vec<f32> {
    let mut out = vec![0.0f32; xs.len() * QK8_0];
    for (i, b) in xs.iter().enumerate() {
        let d = b.d.to_f32();
        for j in 0..QK8_0 {
            out[i * QK8_0 + j] = (b.qs[j] as f32) * d;
        }
    }
    out
}

/// Backend-generic entry point.
/// # Errors
/// Propagates device / launch / cert-write failures.
pub fn run_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let bytes = B::image("quantize_f16_q8_0_dpos")
        .ok_or_else(|| anyhow!("quantize_f16_q8_0_dpos not compiled (CUDA-only)"))?;
    let module = B::load(&dev, bytes)?;
    let kernel = B::kernel(&module, "flambeau_quantize_row_f16_q8_0_dpos")?;
    let pmc = B::attributes(&kernel)?;

    let mut results = Vec::new();
    for case in CASES {
        let seed = 0x9B0F ^ (case.kv_width as u64 * 7919) ^ (case.position as u64 * 101)
            ^ (case.ring_depth as u64 * 13);
        let (got, reference) = run_case::<B>(&dev, &kernel, case, seed)?;
        let max_rel = max_rel_err_with_floor(&got, &reference, 1.0);
        results.push(ShapeResult {
            m: 1,
            k: case.position,
            n: case.kv_width,
            seed,
            max_rel_err: max_rel,
            tolerance: TOL,
            pass: max_rel <= TOL,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: format!("quantize_f16_q8_0_dpos_{arch}"),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "quantize".to_string(),
        dtype_weight: "Q8_0".to_string(),
        dtype_activation: "F16".to_string(),
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

#[allow(clippy::undocumented_unsafe_blocks)]
fn run_case<B: BenchBackend>(
    dev: &B::Device,
    kernel: &B::Kernel<'_>,
    case: &Case,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let Case { kv_width, position, ring_depth } = *case;
    let blocks_per_row = kv_width / QK8_0;
    let write_row = if ring_depth > 0 { position % ring_depth } else { position };
    let cache_rows = if ring_depth > 0 { ring_depth } else { position + 1 };

    let src_f32 = seeded_f32_range(seed ^ 0x11, kv_width, -0.5, 0.5);
    let src: Vec<f16> = src_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let pos_i: Vec<i32> = vec![position as i32];

    // Cache initialised to a sentinel; only the written row should change.
    let cache_bytes = cache_rows * blocks_per_row * Q8_0_BLOCK_BYTES;
    let cache_init: Vec<u8> = vec![0xAB; cache_bytes];

    let d_src = alloc_and_upload(dev, &src);
    let d_pos = alloc_and_upload(dev, &pos_i);
    let d_cache = alloc_and_upload(dev, &cache_init);

    {
        let stream = dev.default_stream();
        let n_i = kv_width as i32;
        let ring_i = ring_depth as i32;
        let x = d_src.as_usize() as u64;
        let y = d_cache.as_usize() as u64;
        let pp = d_pos.as_usize() as u64;
        let mut a = <B::Args<'_>>::default();
        a.push_arg(&x);
        a.push_arg(&y);
        a.push_arg(&n_i);
        a.push_arg(&ring_i);
        a.push_arg(&pp);
        let cfg = LaunchCfg {
            grid: (blocks_per_row as u32, 1, 1),
            block: (QK8_0 as u32, 1, 1),
            shared_bytes: 0,
        };
        B::launch(kernel, stream, cfg, a)?;
        stream.synchronize()?;
    }

    // Read back the written row's Q8 blocks at the on-device-computed offset.
    let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;
    let mut row_raw: Vec<u8> = vec![0u8; row_bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(row_raw.as_mut_ptr() as usize),
            d_cache.offset_bytes(write_row * row_bytes),
            row_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_src, src.len() * 2)?;
        dev.dealloc(d_pos, pos_i.len() * 4)?;
        dev.dealloc(d_cache, cache_bytes)?;
    }

    // Reconstruct BlockQ8_0 from the raw bytes and dequantize.
    let mut blocks: Vec<BlockQ8_0> = Vec::with_capacity(blocks_per_row);
    for b in 0..blocks_per_row {
        let off = b * Q8_0_BLOCK_BYTES;
        let d = f16::from_bits(u16::from_le_bytes([row_raw[off], row_raw[off + 1]]));
        let mut qs = [0i8; QK8_0];
        for (j, q) in qs.iter_mut().enumerate() {
            *q = row_raw[off + 2 + j] as i8;
        }
        blocks.push(BlockQ8_0 { d, qs });
    }
    let got = dequantize_row_q8_0(&blocks);
    let reference: Vec<f32> = src.iter().map(|v| v.to_f32()).collect();
    Ok((got, reference))
}
