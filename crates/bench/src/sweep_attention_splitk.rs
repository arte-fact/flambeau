//! 9.b — attention_decode_f16 split-K (flash-decoding) correctness
//! sweep. Same math as [`sweep_attention`] but launches the two-pass
//! split-K kernel pair and checks against the same F32 reference.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "sweep harness — every unsafe block is a kernel launch or a memcpy_async \
              over buffers allocated locally in the same function and freed before \
              return; invariant is uniform across all sites."
)]

use std::path::Path;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{
    device_count, FuncAttributes, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use half::f16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig, seeded_f32_range};

/// Same shape family as the non-split-K cert.
const SHAPES: &[(usize, usize, usize)] = &[
    (64, 4, 1),   // synthetic smoke
    (128, 32, 4), // Qwen3.5 / GQA-8
    (256, 16, 2), // Qwen3.6 / GQA-8
];

fn pick_chunk(n_tokens: usize) -> usize {
    if n_tokens <= 128 {
        128
    } else if n_tokens <= 1024 {
        128
    } else if n_tokens <= 2048 {
        256
    } else {
        512
    }
}

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("attention_decode_f16_splitk").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let k_chunk: HipKernel<'_> = module.kernel("flambeau_attention_decode_f16_splitk_chunk")?;
    let k_combine: HipKernel<'_> = module.kernel("flambeau_attention_decode_f16_splitk_combine")?;
    let attrs_chunk: FuncAttributes = k_chunk.attributes()?;

    // Split-K targets the long-context decode sweet spot. 16 is there to
    // cover the fallback path (single chunk).
    let contexts = [16usize, 128, 1024, 4096];
    let mut results = Vec::new();
    for &(head_dim, n_heads_q, n_heads_kv) in SHAPES {
        for n_tokens in contexts {
            let seed = 0xDECADE ^ (head_dim as u64 * 7919) ^ (n_tokens as u64 * 101);
            let (got, reference) = run_shape(
                &dev, &k_chunk, &k_combine, head_dim, n_heads_q, n_heads_kv, n_tokens, seed,
            )?;
            let max_rel = max_rel_err_with_floor(&got, &reference, (head_dim as f32).sqrt() * 0.01);
            // Same tolerance as the single-pass cert: 2e-2 covers F16 round-trip
            // noise + online-softmax exp chain at n_tokens=4096.
            let tol = 2e-2;
            results.push(ShapeResult {
                m: n_heads_q,
                k: n_tokens,
                n: head_dim,
                seed,
                max_rel_err: max_rel,
                tolerance: tol,
                pass: max_rel <= tol,
            });
            tracing::info!(
                target: "flambeau_bench::sweep_attention_splitk",
                head_dim, n_heads_q, n_heads_kv,
                n_tokens, max_rel, tol,
                "attention decode split-K shape"
            );
        }
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = rig();
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "attention_decode_f16_splitk_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "attention_decode".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: "|err| <= 2e-2 * max(|ref|, sqrt(head_dim))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig,
        pmc: Some(PmcSnapshot {
            vgpr_count: Some(attrs_chunk.num_regs),
            sgpr_count: None,
            waves_per_simd: Some(attrs_chunk.gfx906_waves_per_simd()),
            mem_busy_pct: None,
            valu_busy_pct: None,
        }),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn run_shape(
    dev: &HipDevice,
    k_chunk: &HipKernel<'_>,
    k_combine: &HipKernel<'_>,
    head_dim: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    n_tokens: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let q_len = n_heads_q * head_dim;
    let kv_len = n_tokens * n_heads_kv * head_dim;
    let q_f32 = seeded_f32_range(seed, q_len, -0.5, 0.5);
    let k_f32 = seeded_f32_range(seed.wrapping_add(0xA1), kv_len, -0.5, 0.5);
    let v_f32 = seeded_f32_range(seed.wrapping_add(0xA2), kv_len, -0.5, 0.5);
    let q_f16: Vec<f16> = q_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let k_f16: Vec<f16> = k_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let v_f16: Vec<f16> = v_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_q = alloc_and_upload(dev, &q_f16);
    let d_k = alloc_and_upload(dev, &k_f16);
    let d_v = alloc_and_upload(dev, &v_f16);
    let out_bytes = q_len * 2;
    let d_out = dev.alloc(out_bytes)?;

    let chunk_size = pick_chunk(n_tokens);
    let n_chunks = n_tokens.div_ceil(chunk_size);
    let part_ms_floats = n_heads_q * n_chunks;
    let part_o_floats = n_heads_q * n_chunks * head_dim;
    let d_part_m = dev.alloc(part_ms_floats * 4)?;
    let d_part_s = dev.alloc(part_ms_floats * 4)?;
    let d_part_o = dev.alloc(part_o_floats * 4)?;

    let scale = 1.0 / (head_dim as f32).sqrt();
    {
        let stream = dev.default_stream();
        let n_heads_q_i = n_heads_q as i32;
        let n_heads_kv_i = n_heads_kv as i32;
        let head_dim_i = head_dim as i32;
        let n_tokens_i = n_tokens as i32;
        let n_chunks_i = n_chunks as i32;
        let chunk_size_i = chunk_size as i32;
        let d_q_ptr: u64 = d_q.as_usize() as u64;
        let d_k_ptr: u64 = d_k.as_usize() as u64;
        let d_v_ptr: u64 = d_v.as_usize() as u64;
        let d_m_ptr: u64 = d_part_m.as_usize() as u64;
        let d_s_ptr: u64 = d_part_s.as_usize() as u64;
        let d_o_ptr: u64 = d_part_o.as_usize() as u64;
        let d_out_ptr: u64 = d_out.as_usize() as u64;
        let scale_f = scale;
        let window_i: i32 = 0;

        let mut a1 = KernelArgs::new();
        a1.push(&d_q_ptr);
        a1.push(&d_k_ptr);
        a1.push(&d_v_ptr);
        a1.push(&d_m_ptr);
        a1.push(&d_s_ptr);
        a1.push(&d_o_ptr);
        a1.push(&n_heads_q_i);
        a1.push(&n_heads_kv_i);
        a1.push(&head_dim_i);
        a1.push(&n_tokens_i);
        a1.push(&n_chunks_i);
        a1.push(&chunk_size_i);
        a1.push(&scale_f);
        a1.push(&window_i);
        let cfg1 = LaunchCfg {
            grid: (n_heads_q as u32, n_chunks as u32, 1),
            block: (head_dim as u32, 1, 1),
            shared_bytes: 0,
        };
        unsafe { k_chunk.launch(stream, cfg1, a1)? };

        let mut a2 = KernelArgs::new();
        a2.push(&d_m_ptr);
        a2.push(&d_s_ptr);
        a2.push(&d_o_ptr);
        a2.push(&d_out_ptr);
        a2.push(&n_heads_q_i);
        a2.push(&n_chunks_i);
        a2.push(&head_dim_i);
        let cfg2 = LaunchCfg {
            grid: (n_heads_q as u32, 1, 1),
            block: (head_dim as u32, 1, 1),
            shared_bytes: 0,
        };
        unsafe { k_combine.launch(stream, cfg2, a2)? };
        stream.synchronize()?;
    }

    let mut out_f16: Vec<f16> = vec![f16::from_f32(0.0); q_len];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out_f16.as_mut_ptr() as usize),
            d_out,
            out_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_q, q_f16.len() * 2)?;
        dev.dealloc(d_k, k_f16.len() * 2)?;
        dev.dealloc(d_v, v_f16.len() * 2)?;
        dev.dealloc(d_out, out_bytes)?;
        dev.dealloc(d_part_m, part_ms_floats * 4)?;
        dev.dealloc(d_part_s, part_ms_floats * 4)?;
        dev.dealloc(d_part_o, part_o_floats * 4)?;
    }
    let got: Vec<f32> = out_f16.iter().map(|v| v.to_f32()).collect();

    // F32 reference.
    let group = n_heads_q / n_heads_kv;
    let mut reference = vec![0.0f32; q_len];
    let q_inputs: Vec<f32> = q_f16.iter().map(|v| v.to_f32()).collect();
    let k_inputs: Vec<f32> = k_f16.iter().map(|v| v.to_f32()).collect();
    let v_inputs: Vec<f32> = v_f16.iter().map(|v| v.to_f32()).collect();
    for qh in 0..n_heads_q {
        let kvh = qh / group;
        let mut scores = vec![0.0f32; n_tokens];
        for t in 0..n_tokens {
            let mut dot = 0.0f64;
            for d in 0..head_dim {
                let qv = q_inputs[qh * head_dim + d];
                let kv = k_inputs[(t * n_heads_kv + kvh) * head_dim + d];
                dot += (qv * kv) as f64;
            }
            scores[t] = (dot as f32) * scale;
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            sum += *s as f64;
        }
        let inv = 1.0f32 / (sum as f32);
        for s in scores.iter_mut() {
            *s *= inv;
        }
        for d in 0..head_dim {
            let mut acc = 0.0f64;
            for t in 0..n_tokens {
                let vv = v_inputs[(t * n_heads_kv + kvh) * head_dim + d];
                acc += (scores[t] * vv) as f64;
            }
            reference[qh * head_dim + d] = f16::from_f32(acc as f32).to_f32();
        }
    }
    Ok((got, reference))
}
