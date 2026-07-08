//! Device-position split-K (flash-decoding) decode attention correctness sweep,
//! Q8_0 KV. Q8 sibling of `sweep_attention_splitk_dpos`: the `_dpos` chunk
//! kernel reads `position` from a device buffer and derives `n_tokens_kv` /
//! `chunk_base` / the slid K-V pointer on-device from the baked
//! (window, ring_depth, slide_row_stride) regime; the combine pass is the shared
//! `attention_decode_q8_kv_splitk` kernel. Covers the three gemma-4 decode
//! regimes plus the mid-bucket case where the baked grid over-provisions
//! `n_chunks` and trailing chunks run the empty-sentinel body:
//!   * global / pre-window  — window == 0, base pointers, eff = position+1
//!   * SWA ring             — window > 0, ring_depth > 0, rows mod ring_depth
//!   * SWA in-band slid      — window > 0, slide_row_stride > 0, base slid on-device
//! K/V are Q8_0-quantized; the reference dequantizes exactly what the kernel
//! reads and runs the F32 softmax over the live rows, so any delta is kernel
//! arithmetic, not quant noise.
//!
//! Backend-generic so it emits the CUDA (sm_86) cert; the chunk kernel is
//! CUDA-only (gfx906 keeps the host-scalar `attention_decode_q8_kv_splitk`).

#![cfg(any(feature = "hip", feature = "cuda"))]

use std::path::Path;

use anyhow::{anyhow, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, LaunchCfg, Stream};
use flambeau_quant::{BlockQ8_0, QK8_0};
use half::f16;

use crate::backend::{BenchBackend, LaunchArgs};
use crate::cert::{now_utc_iso8601, Cert, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig_for, seeded_f32_range};

const MAX_SPLITK_CHUNKS: usize = 32;
const TOL: f32 = 5e-2;
const TOL_FORMULA: &str =
    "|err| <= 5e-2 * max(|ref|, sqrt(head_dim))  (correctness; delta-ppl quality cert)";
const Q8_0_BLOCK_BYTES: usize = 34;

#[derive(Clone, Copy)]
enum Regime {
    /// window == 0, ring_depth == 0, slide == 0. eff = position+1 from row 0.
    Global,
    /// window > 0, ring_depth = slab rows, slide == 0. eff = position+1,
    /// chunk_base = eff-window, rows addressed mod ring_depth.
    Ring,
    /// window > 0, ring_depth == 0, slide > 0. eff = window; base slid to row
    /// (position+1-window) on-device.
    Slid,
}

struct Case {
    head_dim: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    position: usize,
    window: usize,
    /// Ring slab depth in rows (Ring regime only).
    slab_depth: usize,
    regime: Regime,
    /// When `Some`, override the baked grid `n_chunks` (must be >= the derived
    /// value) to exercise the empty-trailing-chunk sentinel.
    bake_n_chunks: Option<usize>,
}

const CASES: &[Case] = &[
    // Global just past the single-pass threshold (n_chunks small).
    Case { head_dim: 128, n_heads_q: 4, n_heads_kv: 1, position: 300, window: 0, slab_depth: 0, regime: Regime::Global, bake_n_chunks: None },
    // Global mid-context, head_dim 256 (gemma-4 SWA / global GQA 16/2).
    Case { head_dim: 256, n_heads_q: 16, n_heads_kv: 2, position: 1641, window: 0, slab_depth: 0, regime: Regime::Global, bake_n_chunks: None },
    // Global mid-bucket: grid baked for the top of the bucket, run at a lower
    // eff so the last chunks run the empty sentinel.
    Case { head_dim: 256, n_heads_q: 16, n_heads_kv: 2, position: 900, window: 0, slab_depth: 0, regime: Regime::Global, bake_n_chunks: Some(MAX_SPLITK_CHUNKS) },
    // SWA ring (window 1024, slab 1152 = window + 128 ubatch), several eff.
    Case { head_dim: 256, n_heads_q: 16, n_heads_kv: 2, position: 1300, window: 1024, slab_depth: 1152, regime: Regime::Ring, bake_n_chunks: None },
    Case { head_dim: 256, n_heads_q: 16, n_heads_kv: 2, position: 4000, window: 1024, slab_depth: 1152, regime: Regime::Ring, bake_n_chunks: None },
    // SWA in-band slid (window 1024, positions inside the band).
    Case { head_dim: 256, n_heads_q: 16, n_heads_kv: 2, position: 1100, window: 1024, slab_depth: 1152, regime: Regime::Slid, bake_n_chunks: None },
    Case { head_dim: 256, n_heads_q: 16, n_heads_kv: 2, position: 1151, window: 1024, slab_depth: 1152, regime: Regime::Slid, bake_n_chunks: None },
];

fn splitk_chunk_size(n: usize) -> usize {
    if n <= 256 {
        return n;
    }
    n.div_ceil(MAX_SPLITK_CHUNKS).max(32).next_multiple_of(4)
}

/// CPU-side Q8_0 quantise — matches the on-device quantize_f16_q8_0 path.
fn quantize_row_q8_0(xs: &[f32]) -> Vec<BlockQ8_0> {
    assert_eq!(xs.len() % QK8_0, 0);
    let nb = xs.len() / QK8_0;
    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        let block = &xs[i * QK8_0..(i + 1) * QK8_0];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; QK8_0];
        for (j, &v) in block.iter().enumerate() {
            qs[j] = (v * id).round().clamp(-127.0, 127.0) as i8;
        }
        out.push(BlockQ8_0 { d: f16::from_f32(d), qs });
    }
    out
}

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
    let bytes = B::image("attention_decode_q8_kv_splitk_dpos")
        .ok_or_else(|| anyhow!("attention_decode_q8_kv_splitk_dpos not compiled (CUDA-only)"))?;
    let module = B::load(&dev, bytes)?;
    let k_chunk = B::kernel(&module, "flambeau_attention_decode_q8_kv_splitk_dpos_chunk")?;
    let combine_bytes = B::image("attention_decode_q8_kv_splitk")
        .ok_or_else(|| anyhow!("attention_decode_q8_kv_splitk not compiled"))?;
    let combine_module = B::load(&dev, combine_bytes)?;
    let k_combine = B::kernel(&combine_module, "flambeau_attention_decode_q8_kv_splitk_combine")?;
    let pmc = B::attributes(&k_chunk)?;

    let mut results = Vec::new();
    for case in CASES {
        let seed = 0xD9051 ^ (case.head_dim as u64 * 7919) ^ (case.position as u64 * 101)
            ^ (case.window as u64 * 13);
        let (got, reference) = run_case::<B>(&dev, &k_chunk, &k_combine, case, seed)?;
        let max_rel = max_rel_err_with_floor(&got, &reference, (case.head_dim as f32).sqrt() * 0.01);
        results.push(ShapeResult {
            m: case.n_heads_q,
            k: case.position + 1,
            n: case.head_dim,
            seed,
            max_rel_err: max_rel,
            tolerance: TOL,
            pass: max_rel <= TOL,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: format!("attention_decode_q8_kv_splitk_dpos_{arch}"),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "attention_decode_q8_kv".to_string(),
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

#[allow(clippy::undocumented_unsafe_blocks, clippy::too_many_lines)]
fn run_case<B: BenchBackend>(
    dev: &B::Device,
    k_chunk: &B::Kernel<'_>,
    k_combine: &B::Kernel<'_>,
    case: &Case,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let Case { head_dim, n_heads_q, n_heads_kv, position, window, slab_depth, regime, bake_n_chunks } =
        *case;
    let group = n_heads_q / n_heads_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_width = n_heads_kv * head_dim;
    let nb_per_row = head_dim / QK8_0;
    // Q8 row stride in BYTES: one full KV row = n_heads_kv * (head_dim/32) blocks.
    let q8_row_stride = (kv_width / QK8_0) * Q8_0_BLOCK_BYTES;

    // Regime-specific: baked kernel params, cache row count, and the ordered
    // list of live physical KV rows the reference softmaxes over.
    let (eff, chunk_base, kernel_window, ring_depth, slide_row_stride, cache_rows, live_rows): (
        usize,
        usize,
        i32,
        i32,
        i32,
        usize,
        Vec<usize>,
    ) = match regime {
        Regime::Global => {
            let eff = position + 1;
            (eff, 0, 0, 0, 0, eff, (0..eff).collect())
        }
        Regime::Ring => {
            let eff = position + 1;
            let cb = eff - window;
            let rows = (cb..eff).map(|t| t % slab_depth).collect();
            (eff, cb, window as i32, slab_depth as i32, 0, slab_depth, rows)
        }
        Regime::Slid => {
            let base_row = position + 1 - window;
            let rows = (0..window).map(|i| base_row + i).collect();
            (window, 0, 0, 0, q8_row_stride as i32, position + 1, rows)
        }
    };

    let chunk_len = eff - chunk_base;
    let chunk_size = splitk_chunk_size(chunk_len);
    let n_chunks = bake_n_chunks.unwrap_or_else(|| chunk_len.div_ceil(chunk_size).max(1));

    // Inputs.
    let q_host: Vec<f16> = seeded_f32_range(seed, n_heads_q * head_dim, -0.5, 0.5)
        .iter().map(|v| f16::from_f32(*v)).collect();
    let cache_n = cache_rows * kv_width;
    let k_f32 = seeded_f32_range(seed ^ 0xA1, cache_n, -0.5, 0.5);
    let v_f32 = seeded_f32_range(seed ^ 0xA2, cache_n, -0.5, 0.5);
    let pos_i: Vec<i32> = vec![position as i32];

    // Quantise K/V to Q8_0, layout [cache_rows * n_heads_kv, head_dim] → blocks.
    let n_block_rows = cache_rows * n_heads_kv;
    let mut k_blocks: Vec<BlockQ8_0> = Vec::with_capacity(n_block_rows * nb_per_row);
    let mut v_blocks: Vec<BlockQ8_0> = Vec::with_capacity(n_block_rows * nb_per_row);
    for row in 0..n_block_rows {
        k_blocks.extend(quantize_row_q8_0(&k_f32[row * head_dim..(row + 1) * head_dim]));
        v_blocks.extend(quantize_row_q8_0(&v_f32[row * head_dim..(row + 1) * head_dim]));
    }

    let d_q = alloc_and_upload(dev, &q_host);
    let d_k = alloc_and_upload(dev, &k_blocks);
    let d_v = alloc_and_upload(dev, &v_blocks);
    let d_pos = alloc_and_upload(dev, &pos_i);

    let out_len = n_heads_q * head_dim;
    let d_out = dev.alloc(out_len * 2)?;
    let part_ms = n_heads_q * n_chunks;
    let part_o = part_ms * head_dim;
    let d_m = dev.alloc(part_ms * 4)?;
    let d_s = dev.alloc(part_ms * 4)?;
    let d_o = dev.alloc(part_o * 4)?;

    {
        let stream = dev.default_stream();
        let n_heads_q_i = n_heads_q as i32;
        let n_heads_kv_i = n_heads_kv as i32;
        let head_dim_i = head_dim as i32;
        let n_chunks_i = n_chunks as i32;
        let chunk_size_i = chunk_size as i32;
        let window_i = kernel_window;
        let ring_i = ring_depth;
        let slide_i = slide_row_stride;
        let q_ptr = d_q.as_usize() as u64;
        let k_ptr = d_k.as_usize() as u64;
        let v_ptr = d_v.as_usize() as u64;
        let m_ptr = d_m.as_usize() as u64;
        let s_ptr = d_s.as_usize() as u64;
        let o_ptr = d_o.as_usize() as u64;
        let out_ptr = d_out.as_usize() as u64;
        let pos_ptr = d_pos.as_usize() as u64;

        let mut a1 = <B::Args<'_>>::default();
        a1.push_arg(&q_ptr);
        a1.push_arg(&k_ptr);
        a1.push_arg(&v_ptr);
        a1.push_arg(&m_ptr);
        a1.push_arg(&s_ptr);
        a1.push_arg(&o_ptr);
        a1.push_arg(&n_heads_q_i);
        a1.push_arg(&n_heads_kv_i);
        a1.push_arg(&head_dim_i);
        a1.push_arg(&n_chunks_i);
        a1.push_arg(&chunk_size_i);
        a1.push_arg(&scale);
        a1.push_arg(&window_i);
        a1.push_arg(&ring_i);
        a1.push_arg(&slide_i);
        a1.push_arg(&pos_ptr);
        // Chunk pass: block = head_dim/4 (one thread per int32-packed Q8 quad).
        let cfg1 = LaunchCfg {
            grid: (n_heads_q as u32, n_chunks as u32, 1),
            block: ((head_dim / 4) as u32, 1, 1),
            shared_bytes: 0,
        };
        B::launch(k_chunk, stream, cfg1, a1)?;

        let mut a2 = <B::Args<'_>>::default();
        a2.push_arg(&m_ptr);
        a2.push_arg(&s_ptr);
        a2.push_arg(&o_ptr);
        a2.push_arg(&out_ptr);
        a2.push_arg(&n_heads_q_i);
        a2.push_arg(&n_chunks_i);
        a2.push_arg(&head_dim_i);
        let cfg2 = LaunchCfg {
            grid: (n_heads_q as u32, 1, 1),
            block: (head_dim as u32, 1, 1),
            shared_bytes: 0,
        };
        B::launch(k_combine, stream, cfg2, a2)?;
        stream.synchronize()?;
    }

    let mut out_f16: Vec<f16> = vec![f16::from_f32(0.0); out_len];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out_f16.as_mut_ptr() as usize),
            d_out,
            out_len * 2,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_q, q_host.len() * 2)?;
        dev.dealloc(d_k, k_blocks.len() * std::mem::size_of::<BlockQ8_0>())?;
        dev.dealloc(d_v, v_blocks.len() * std::mem::size_of::<BlockQ8_0>())?;
        dev.dealloc(d_pos, pos_i.len() * 4)?;
        dev.dealloc(d_out, out_len * 2)?;
        dev.dealloc(d_m, part_ms * 4)?;
        dev.dealloc(d_s, part_ms * 4)?;
        dev.dealloc(d_o, part_o * 4)?;
    }
    let got: Vec<f32> = out_f16.iter().map(|v| v.to_f32()).collect();

    // Reference: dequantise K/V (exactly what the kernel reads) and run the F32
    // softmax over the live physical KV rows.
    let mut k_rt = vec![0.0f32; cache_n];
    let mut v_rt = vec![0.0f32; cache_n];
    for row in 0..n_block_rows {
        let bk = &k_blocks[row * nb_per_row..(row + 1) * nb_per_row];
        let bv = &v_blocks[row * nb_per_row..(row + 1) * nb_per_row];
        k_rt[row * head_dim..(row + 1) * head_dim].copy_from_slice(&dequantize_row_q8_0(bk));
        v_rt[row * head_dim..(row + 1) * head_dim].copy_from_slice(&dequantize_row_q8_0(bv));
    }
    let q_inputs: Vec<f32> = q_host.iter().map(|v| v.to_f32()).collect();
    let mut reference = vec![0.0f32; out_len];
    for qh in 0..n_heads_q {
        let kvh = qh / group;
        let q_off = qh * head_dim;
        let mut scores: Vec<f32> = Vec::with_capacity(live_rows.len());
        for &row in &live_rows {
            let base = (row * n_heads_kv + kvh) * head_dim;
            let mut dot = 0.0f64;
            for d in 0..head_dim {
                dot += (q_inputs[q_off + d] * k_rt[base + d]) as f64;
            }
            scores.push((dot as f32) * scale);
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        let mut w: Vec<f32> = Vec::with_capacity(scores.len());
        for &sc in &scores {
            let e = (sc - m).exp();
            w.push(e);
            sum += e as f64;
        }
        let inv = 1.0f32 / (sum as f32);
        let out_off = qh * head_dim;
        for d in 0..head_dim {
            let mut acc = 0.0f64;
            for (i, &row) in live_rows.iter().enumerate() {
                let base = (row * n_heads_kv + kvh) * head_dim;
                acc += (w[i] * inv * v_rt[base + d]) as f64;
            }
            reference[out_off + d] = f16::from_f32(acc as f32).to_f32();
        }
    }
    Ok((got, reference))
}
