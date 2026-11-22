//! Batched-slots split-K (flash-decoding) decode attention correctness sweep.
//! Same math as [`sweep_attention_splitk`] with the extra slot dimension:
//! per-slot K/V caches (own base pointer + `n_tokens`), SWA window + ring-wrap,
//! and variable per-slot lengths in one launch. Covers the gemma-4 decode shapes
//! (head_dim 256 SWA / 512 global, GQA 16/8) and the chunk counts that the
//! runtime hits (`n_chunks` 5–8, `chunk_size` 256/512), including the multi-slot
//! unequal-length case. Checked against a per-slot F32 windowed/ring reference.
//!
//! Backend-generic so it emits the CUDA (sm_86) cert; the kernel is CUDA-only
//! (gfx906 keeps the serial `attention_decode_f16_batched` fallback).

#![cfg(any(feature = "hip", feature = "cuda"))]

use std::path::Path;

use anyhow::{anyhow, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, LaunchCfg, Stream};
use half::f16;

use crate::backend::{BenchBackend, LaunchArgs};
use crate::cert::{now_utc_iso8601, Cert, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig_for, seeded_f32_range};

const MAX_SPLITK_CHUNKS: usize = 32;
const TOL: f32 = 2e-2;
const TOL_FORMULA: &str = "|err| <= 2e-2 * max(|ref|, sqrt(head_dim))";

/// One sweep case: a batch of slots sharing head geometry + window/ring, with
/// per-slot context lengths (unequal lengths exercise the empty-tail chunks).
struct Case {
    head_dim: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    /// Per-slot context length; `.len()` is `n_slots`.
    slot_lengths: &'static [usize],
    /// SWA window (0 = full attention).
    window: usize,
    /// Ring-cache depth (0 = linear cache indexed by absolute token).
    ring: usize,
}

const CASES: &[Case] = &[
    // Smoke / fallback: single chunk (n <= 256), full attention.
    Case { head_dim: 64, n_heads_q: 4, n_heads_kv: 1, slot_lengths: &[16], window: 0, ring: 0 },
    // gemma global (head_dim 512, no window): cs512/n_chunks5 and cs256/n_chunks7.
    Case { head_dim: 512, n_heads_q: 16, n_heads_kv: 8, slot_lengths: &[2127], window: 0, ring: 0 },
    Case { head_dim: 512, n_heads_q: 16, n_heads_kv: 8, slot_lengths: &[1642], window: 0, ring: 0 },
    // gemma SWA (head_dim 256, window == ring == 1024, real wrap): n_chunks 5/6/7.
    Case { head_dim: 256, n_heads_q: 16, n_heads_kv: 8, slot_lengths: &[1242], window: 1024, ring: 1024 },
    Case { head_dim: 256, n_heads_q: 16, n_heads_kv: 8, slot_lengths: &[1382], window: 1024, ring: 1024 },
    Case { head_dim: 256, n_heads_q: 16, n_heads_kv: 8, slot_lengths: &[1642], window: 1024, ring: 1024 },
    // Multi-slot unequal length — the concurrent variable-length decode shape.
    Case { head_dim: 256, n_heads_q: 16, n_heads_kv: 8, slot_lengths: &[2127, 1242], window: 1024, ring: 1024 },
    Case { head_dim: 512, n_heads_q: 16, n_heads_kv: 8, slot_lengths: &[2043, 545], window: 0, ring: 0 },
];

fn splitk_chunk_size(n: usize) -> usize {
    let base = if n <= 256 {
        return n;
    } else if n <= 1024 {
        128
    } else if n <= 2048 {
        256
    } else {
        512
    };
    if n.div_ceil(base) <= MAX_SPLITK_CHUNKS {
        base
    } else {
        n.div_ceil(MAX_SPLITK_CHUNKS).div_ceil(32) * 32
    }
}

/// Backend-generic entry point.
/// # Errors
/// Propagates device / launch / cert-write failures.
pub fn run_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let bytes = B::image("attention_decode_batched_splitk")
        .ok_or_else(|| anyhow!("attention_decode_batched_splitk not compiled (CUDA-only)"))?;
    let module = B::load(&dev, bytes)?;
    let k_chunk = B::kernel(&module, "flambeau_attention_decode_f16_batched_splitk_chunk")?;
    let k_combine = B::kernel(&module, "flambeau_attention_decode_f16_batched_splitk_combine")?;
    let pmc = B::attributes(&k_chunk)?;

    let mut results = Vec::new();
    for case in CASES {
        let max_n = case.slot_lengths.iter().copied().max().unwrap_or(0);
        let seed = 0xB47C4ED
            ^ (case.head_dim as u64 * 7919)
            ^ (max_n as u64 * 101)
            ^ (case.slot_lengths.len() as u64 * 37)
            ^ (case.window as u64 * 13);
        let (got, reference) = run_case::<B>(&dev, &k_chunk, &k_combine, case, seed)?;
        let max_rel = max_rel_err_with_floor(&got, &reference, (case.head_dim as f32).sqrt() * 0.01);
        results.push(ShapeResult {
            m: case.n_heads_q * case.slot_lengths.len(),
            k: max_n,
            n: case.head_dim,
            seed,
            max_rel_err: max_rel,
            tolerance: TOL,
            pass: max_rel <= TOL,
        });
        tracing::info!(
            target: "flambeau_bench::sweep_attention_batched_splitk",
            head_dim = case.head_dim, n_slots = case.slot_lengths.len(),
            max_n, window = case.window, ring = case.ring, max_rel, tol = TOL,
            "batched split-K decode shape"
        );
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: format!("attention_decode_f16_batched_splitk_{arch}"),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "attention_decode".to_string(),
        dtype_weight: "F16".to_string(),
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
    let Case { head_dim, n_heads_q, n_heads_kv, slot_lengths, window, ring } = *case;
    let n_slots = slot_lengths.len();
    let group = n_heads_q / n_heads_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let max_n = slot_lengths.iter().copied().max().unwrap_or(0);
    let chunk_size = splitk_chunk_size(max_n);
    let n_chunks = max_n.div_ceil(chunk_size).max(1);

    // Per-slot Q + K/V caches. Cache rows = ring (wrap) or n_tokens (linear).
    let q_all: Vec<f16> = seeded_f32_range(seed, n_slots * n_heads_q * head_dim, -0.5, 0.5)
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let d_q = alloc_and_upload(dev, &q_all);

    let mut owned: Vec<(DevicePtr, usize)> = Vec::new();
    let mut k_ptrs: Vec<u64> = Vec::with_capacity(n_slots);
    let mut v_ptrs: Vec<u64> = Vec::with_capacity(n_slots);
    // Host copies of the caches for the reference pass.
    let mut k_host: Vec<Vec<f32>> = Vec::with_capacity(n_slots);
    let mut v_host: Vec<Vec<f32>> = Vec::with_capacity(n_slots);
    for (s, &n_tokens) in slot_lengths.iter().enumerate() {
        let rows = if ring > 0 { ring } else { n_tokens };
        let len = rows * n_heads_kv * head_dim;
        let k_f32 = seeded_f32_range(seed ^ (0xA1 + s as u64 * 131), len, -0.5, 0.5);
        let v_f32 = seeded_f32_range(seed ^ (0xA2 + s as u64 * 131), len, -0.5, 0.5);
        let k_f16: Vec<f16> = k_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let v_f16: Vec<f16> = v_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let d_k = alloc_and_upload(dev, &k_f16);
        let d_v = alloc_and_upload(dev, &v_f16);
        k_ptrs.push(d_k.as_usize() as u64);
        v_ptrs.push(d_v.as_usize() as u64);
        owned.push((d_k, k_f16.len() * 2));
        owned.push((d_v, v_f16.len() * 2));
        // Reference reads the f16-round-tripped values.
        k_host.push(k_f16.iter().map(|v| v.to_f32()).collect());
        v_host.push(v_f16.iter().map(|v| v.to_f32()).collect());
    }
    let n_tokens_i: Vec<i32> = slot_lengths.iter().map(|&n| n as i32).collect();
    let d_k_ptrs = alloc_and_upload(dev, &k_ptrs);
    let d_v_ptrs = alloc_and_upload(dev, &v_ptrs);
    let d_n_tokens = alloc_and_upload(dev, &n_tokens_i);

    let out_len = n_slots * n_heads_q * head_dim;
    let d_out = dev.alloc(out_len * 2)?;
    let part_ms = n_heads_q * n_slots * n_chunks;
    let part_o = part_ms * head_dim;
    let d_m = dev.alloc(part_ms * 4)?;
    let d_s = dev.alloc(part_ms * 4)?;
    let d_o = dev.alloc(part_o * 4)?;

    {
        let stream = dev.default_stream();
        let n_heads_q_i = n_heads_q as i32;
        let n_heads_kv_i = n_heads_kv as i32;
        let head_dim_i = head_dim as i32;
        let n_slots_i = n_slots as i32;
        let n_chunks_i = n_chunks as i32;
        let chunk_size_i = chunk_size as i32;
        let window_i = window as i32;
        let ring_i = ring as i32;
        let q_ptr = d_q.as_usize() as u64;
        let kp_ptr = d_k_ptrs.as_usize() as u64;
        let vp_ptr = d_v_ptrs.as_usize() as u64;
        let m_ptr = d_m.as_usize() as u64;
        let s_ptr = d_s.as_usize() as u64;
        let o_ptr = d_o.as_usize() as u64;
        let nkv_ptr = d_n_tokens.as_usize() as u64;
        let out_ptr = d_out.as_usize() as u64;

        let mut a1 = <B::Args<'_>>::default();
        a1.push_arg(&q_ptr);
        a1.push_arg(&kp_ptr);
        a1.push_arg(&vp_ptr);
        a1.push_arg(&m_ptr);
        a1.push_arg(&s_ptr);
        a1.push_arg(&o_ptr);
        a1.push_arg(&nkv_ptr);
        a1.push_arg(&n_heads_q_i);
        a1.push_arg(&n_heads_kv_i);
        a1.push_arg(&head_dim_i);
        a1.push_arg(&n_slots_i);
        a1.push_arg(&n_chunks_i);
        a1.push_arg(&chunk_size_i);
        a1.push_arg(&scale);
        a1.push_arg(&window_i);
        a1.push_arg(&ring_i);
        let cfg1 = LaunchCfg {
            grid: (n_heads_q as u32, n_slots as u32, n_chunks as u32),
            block: (head_dim as u32, 1, 1),
            shared_bytes: 0,
        };
        B::launch(k_chunk, stream, cfg1, a1)?;

        let mut a2 = <B::Args<'_>>::default();
        a2.push_arg(&m_ptr);
        a2.push_arg(&s_ptr);
        a2.push_arg(&o_ptr);
        a2.push_arg(&out_ptr);
        a2.push_arg(&n_heads_q_i);
        a2.push_arg(&n_slots_i);
        a2.push_arg(&n_chunks_i);
        a2.push_arg(&head_dim_i);
        let cfg2 = LaunchCfg {
            grid: (n_heads_q as u32, n_slots as u32, 1),
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
        dev.dealloc(d_q, q_all.len() * 2)?;
        dev.dealloc(d_k_ptrs, k_ptrs.len() * 8)?;
        dev.dealloc(d_v_ptrs, v_ptrs.len() * 8)?;
        dev.dealloc(d_n_tokens, n_tokens_i.len() * 4)?;
        dev.dealloc(d_out, out_len * 2)?;
        dev.dealloc(d_m, part_ms * 4)?;
        dev.dealloc(d_s, part_ms * 4)?;
        dev.dealloc(d_o, part_o * 4)?;
        for (ptr, bytes) in owned {
            dev.dealloc(ptr, bytes)?;
        }
    }
    let got: Vec<f32> = out_f16.iter().map(|v| v.to_f32()).collect();

    // Per-slot F32 windowed / ring reference (matches attention_decode_f16_batched_body).
    let q_inputs: Vec<f32> = q_all.iter().map(|v| v.to_f32()).collect();
    let mut reference = vec![0.0f32; out_len];
    for (s, &n_tokens) in slot_lengths.iter().enumerate() {
        let t_lo = if window > 0 { n_tokens.saturating_sub(window) } else { 0 };
        let k_cache = &k_host[s];
        let v_cache = &v_host[s];
        for qh in 0..n_heads_q {
            let kvh = qh / group;
            let q_off = (s * n_heads_q + qh) * head_dim;
            let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n_tokens - t_lo);
            for t in t_lo..n_tokens {
                let t_phys = if ring > 0 { t % ring } else { t };
                let base = (t_phys * n_heads_kv + kvh) * head_dim;
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += (q_inputs[q_off + d] * k_cache[base + d]) as f64;
                }
                scores.push((t_phys, (dot as f32) * scale));
            }
            let m = scores.iter().map(|&(_, v)| v).fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            let mut w: Vec<f32> = Vec::with_capacity(scores.len());
            for &(_, sc) in &scores {
                let e = (sc - m).exp();
                w.push(e);
                sum += e as f64;
            }
            let inv = 1.0f32 / (sum as f32);
            let out_off = (s * n_heads_q + qh) * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for (i, &(t_phys, _)) in scores.iter().enumerate() {
                    let base = (t_phys * n_heads_kv + kvh) * head_dim;
                    acc += (w[i] * inv * v_cache[base + d]) as f64;
                }
                reference[out_off + d] = f16::from_f32(acc as f32).to_f32();
            }
        }
    }
    Ok((got, reference))
}
