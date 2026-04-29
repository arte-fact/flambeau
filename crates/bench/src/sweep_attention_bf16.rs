//! MTP-4-C-4 — BF16 attention decode (GQA) correctness sweep.
//!
//! Mirrors the F16 sweep across the same head-dim / GQA shapes, but
//! all storage is BF16 (Q / K / V / out). Internal F32 math; reference
//! is the same decomposed F32 attention with BF16 quantisation applied
//! to inputs and output.
//!
//! Shapes:
//!   (head_dim, n_heads_q, n_heads_kv)
//!     (64, 4, 1)     — smoke (1 wave64 warp)
//!     (128, 32, 4)   — Qwen3.5 GQA-8
//!     (256, 16, 2)   — Qwen3.6 GQA-8 / MTP target

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
use half::bf16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig, seeded_f32_range};

const SHAPES: &[(usize, usize, usize)] = &[
    (64, 4, 1),
    (128, 32, 4),
    (256, 16, 2),
];

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    if device_count().context("hipGetDeviceCount")? < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("attention_decode_bf16")
        .ok_or_else(|| anyhow::anyhow!("attention_decode_bf16 not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_attention_decode_bf16")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    let contexts = [16usize, 128, 1024, 4096];
    let mut results = Vec::new();
    for &(head_dim, n_heads_q, n_heads_kv) in SHAPES {
        for n_tokens in contexts {
            let seed = 0xBF16_DECAu64
                ^ (head_dim as u64 * 7919)
                ^ (n_tokens as u64 * 101);
            let (got, reference) =
                run_shape(&dev, &kernel, head_dim, n_heads_q, n_heads_kv, n_tokens, seed)?;
            let max_rel = max_rel_err_with_floor(&got, &reference, (head_dim as f32).sqrt() * 0.01);
            // BF16's 7-bit mantissa (~7.8e-3 relative noise) compounds with
            // online softmax + sum chain. F16 sweep uses 2e-2 — BF16 has 3
            // fewer mantissa bits than F16, so we widen to 5e-2 (still well
            // under a usable bar; structural bugs would show 10×+).
            let tol = 5e-2;
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
                target: "flambeau_bench::sweep_attention_bf16",
                head_dim, n_heads_q, n_heads_kv,
                n_tokens, max_rel, tol,
                "attention decode bf16 shape"
            );
        }
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "attention_decode_bf16_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "attention_decode".to_string(),
        dtype_weight: "BF16".to_string(),
        dtype_activation: "BF16".to_string(),
        tolerance_formula: "|err| <= 5e-2 * max(|ref|, sqrt(head_dim) * 0.01)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig(),
        pmc: Some(PmcSnapshot {
            vgpr_count: Some(attrs.num_regs),
            sgpr_count: None,
            waves_per_simd: Some(attrs.gfx906_waves_per_simd()),
            mem_busy_pct: None,
            valu_busy_pct: None,
        }),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn run_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
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
    let q_bf16: Vec<bf16> = q_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    let k_bf16: Vec<bf16> = k_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    let v_bf16: Vec<bf16> = v_f32.iter().map(|v| bf16::from_f32(*v)).collect();

    let d_q = alloc_and_upload(dev, &q_bf16);
    let d_k = alloc_and_upload(dev, &k_bf16);
    let d_v = alloc_and_upload(dev, &v_bf16);
    let out_bytes = n_heads_q * head_dim * 2;
    let d_out = dev.alloc(out_bytes)?;

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    {
        let stream = dev.default_stream();
        let n_heads_q_i = n_heads_q as i32;
        let n_heads_kv_i = n_heads_kv as i32;
        let head_dim_i = head_dim as i32;
        let n_tokens_i = n_tokens as i32;
        let d_q_ptr: u64 = d_q.as_usize() as u64;
        let d_k_ptr: u64 = d_k.as_usize() as u64;
        let d_v_ptr: u64 = d_v.as_usize() as u64;
        let d_out_ptr: u64 = d_out.as_usize() as u64;
        let scale_f = scale;
        let mut args = KernelArgs::new();
        args.push(&d_q_ptr);
        args.push(&d_k_ptr);
        args.push(&d_v_ptr);
        args.push(&d_out_ptr);
        args.push(&n_heads_q_i);
        args.push(&n_heads_kv_i);
        args.push(&head_dim_i);
        args.push(&n_tokens_i);
        args.push(&scale_f);
        let cfg = LaunchCfg::one_d(n_heads_q as u32, head_dim as u32);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut out_bf16 = vec![bf16::from_f32(0.0); n_heads_q * head_dim];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out_bf16.as_mut_ptr() as usize),
            d_out,
            out_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_q, q_bf16.len() * 2)?;
        dev.dealloc(d_k, k_bf16.len() * 2)?;
        dev.dealloc(d_v, v_bf16.len() * 2)?;
        dev.dealloc(d_out, out_bytes)?;
    }
    let got: Vec<f32> = out_bf16.iter().map(|v| v.to_f32()).collect();

    // Reference: decomposed F32 attention with BF16 quantisation on
    // inputs (matches what the kernel reads from HBM) and output.
    let group = n_heads_q / n_heads_kv;
    let q_inputs: Vec<f32> = q_bf16.iter().map(|v| v.to_f32()).collect();
    let k_inputs: Vec<f32> = k_bf16.iter().map(|v| v.to_f32()).collect();
    let v_inputs: Vec<f32> = v_bf16.iter().map(|v| v.to_f32()).collect();
    let mut reference = vec![0.0f32; n_heads_q * head_dim];
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
            reference[qh * head_dim + d] = bf16::from_f32(acc as f32).to_f32();
        }
    }
    Ok((got, reference))
}
