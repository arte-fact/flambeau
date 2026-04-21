//! V1.6.5 attention prefill (F16 KV, GQA, causal mask) correctness sweep.
//!
//! Covers both V1 head_dim values:
//!   - head_dim=128, GQA-32/4 — Qwen3.5 family.
//!   - head_dim=256, GQA-16/2 — Qwen3.6 family.
//! Varies Q token count (the novel axis vs decode) and KV cache size.
//! Causal mask is applied inside the kernel via a per-(q_token) context
//! limit.

#![cfg(feature = "hip")]

use std::path::Path;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{
    device_count, FuncAttributes, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use half::f16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};

/// (head_dim, n_heads_q, n_heads_kv) — V1 target families.
const SHAPES: &[(usize, usize, usize)] = &[
    (128, 32, 4), // Qwen3.5
    (256, 16, 2), // Qwen3.6
];

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("attention_prefill_f16").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_attention_prefill_f16")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // (n_q_tokens, n_k_tokens, q_offset) combinations covering:
    //  - first prefill batch (q_offset=0, n_k=n_q).
    //  - follow-on prefill batch into an existing cache (q_offset>0).
    let cases = [
        (8usize, 8usize, 0usize),   // short first batch
        (128, 128, 0),              // canonical 128-token prefill
        (128, 640, 512),            // 2nd 128-batch into 512-cache
        (512, 512, 0),              // larger prefill
    ];
    let mut results = Vec::new();
    for &(head_dim, n_heads_q, n_heads_kv) in SHAPES {
        for (n_q, n_k, q_off) in cases {
            let seed = 0xDECADE
                ^ (head_dim as u64 * 7919)
                ^ (n_q as u64 * 101)
                ^ (n_k as u64 * 31)
                ^ (q_off as u64);
            let (got, reference) = run_shape(
                &dev, &kernel, head_dim, n_heads_q, n_heads_kv, n_q, n_k, q_off, seed,
            )?;
            let max_rel = max_rel_err(&got, &reference, head_dim);
            let tol = 2e-2;
            results.push(ShapeResult {
                m: n_q,
                k: n_k,
                n: head_dim,
                seed,
                max_rel_err: max_rel,
                tolerance: tol,
                pass: max_rel <= tol,
            });
            tracing::info!(
                target: "flambeau_bench::sweep_attention_prefill",
                head_dim, n_heads_q, n_heads_kv,
                n_q, n_k, q_off, max_rel, tol,
                "prefill shape"
            );
        }
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = format!("{}-gfx906", hostname().unwrap_or_else(|| "unknown".into()));
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "attention_prefill_f16_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "attention_prefill".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: "|err| <= 2e-2 * max(|ref|, sqrt(head_dim))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig,
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

#[allow(clippy::too_many_arguments)]
fn run_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    head_dim: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    n_q_tokens: usize,
    n_k_tokens: usize,
    q_offset: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let q_len = n_q_tokens * n_heads_q * head_dim;
    let kv_len = n_k_tokens * n_heads_kv * head_dim;
    let q_f32 = seeded_f32(seed, q_len);
    let k_f32 = seeded_f32(seed.wrapping_add(0xA1), kv_len);
    let v_f32 = seeded_f32(seed.wrapping_add(0xA2), kv_len);
    let q_f16: Vec<f16> = q_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let k_f16: Vec<f16> = k_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let v_f16: Vec<f16> = v_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_q = alloc_and_upload(dev, &q_f16);
    let d_k = alloc_and_upload(dev, &k_f16);
    let d_v = alloc_and_upload(dev, &v_f16);
    let out_bytes = q_len * 2;
    let d_out = dev.alloc(out_bytes)?;

    let scale = 1.0 / (head_dim as f32).sqrt();
    {
        let stream = dev.default_stream();
        let n_q_i = n_q_tokens as i32;
        let n_heads_q_i = n_heads_q as i32;
        let n_heads_kv_i = n_heads_kv as i32;
        let head_dim_i = head_dim as i32;
        let n_k_i = n_k_tokens as i32;
        let q_off_i = q_offset as i32;
        let scale_f = scale;
        let d_q_ptr: u64 = d_q.as_usize() as u64;
        let d_k_ptr: u64 = d_k.as_usize() as u64;
        let d_v_ptr: u64 = d_v.as_usize() as u64;
        let d_out_ptr: u64 = d_out.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_q_ptr);
        args.push(&d_k_ptr);
        args.push(&d_v_ptr);
        args.push(&d_out_ptr);
        args.push(&n_q_i);
        args.push(&n_heads_q_i);
        args.push(&n_heads_kv_i);
        args.push(&head_dim_i);
        args.push(&n_k_i);
        args.push(&q_off_i);
        args.push(&scale_f);
        let cfg = LaunchCfg {
            grid: (n_q_tokens as u32, n_heads_q as u32, 1),
            block: (head_dim as u32, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
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
    }
    let got: Vec<f32> = out_f16.iter().map(|v| v.to_f32()).collect();

    // CPU F32 reference with causal mask.
    let group = n_heads_q / n_heads_kv;
    let mut reference = vec![0.0f32; q_len];
    let q_in: Vec<f32> = q_f16.iter().map(|v| v.to_f32()).collect();
    let k_in: Vec<f32> = k_f16.iter().map(|v| v.to_f32()).collect();
    let v_in: Vec<f32> = v_f16.iter().map(|v| v.to_f32()).collect();
    for qt in 0..n_q_tokens {
        let limit = usize::min(q_offset + qt + 1, n_k_tokens);
        if limit == 0 {
            continue;
        }
        for qh in 0..n_heads_q {
            let kvh = qh / group;
            let mut scores = vec![0.0f32; limit];
            for t in 0..limit {
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    let qv = q_in[(qt * n_heads_q + qh) * head_dim + d];
                    let kv = k_in[(t * n_heads_kv + kvh) * head_dim + d];
                    dot += (qv * kv) as f64;
                }
                scores[t] = (dot as f32) * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s as f64;
            }
            let inv = 1.0f32 / sum as f32;
            for s in scores.iter_mut() {
                *s *= inv;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for t in 0..limit {
                    let vv = v_in[(t * n_heads_kv + kvh) * head_dim + d];
                    acc += (scores[t] * vv) as f64;
                }
                reference[(qt * n_heads_q + qh) * head_dim + d] =
                    f16::from_f32(acc as f32).to_f32();
            }
        }
    }
    Ok((got, reference))
}

fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (s >> 32) as u32;
            ((u as f32 / u32::MAX as f32) - 0.5)
        })
        .collect()
}

fn alloc_and_upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(data.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

fn max_rel_err(got: &[f32], reference: &[f32], head_dim: usize) -> f32 {
    let abs_floor = (head_dim as f32).sqrt() * 0.01;
    got.iter()
        .zip(reference)
        .map(|(g, r)| (g - r).abs() / r.abs().max(abs_floor))
        .fold(0.0f32, f32::max)
}

fn hostname() -> Option<String> {
    std::env::var("HOSTNAME").ok().or_else(|| {
        let mut buf = vec![0u8; 256];
        let rv = unsafe { libc_gethostname(buf.as_mut_ptr() as *mut _, buf.len()) };
        if rv != 0 {
            return None;
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        buf.truncate(end);
        String::from_utf8(buf).ok()
    })
}

extern "C" {
    #[link_name = "gethostname"]
    fn libc_gethostname(name: *mut std::os::raw::c_char, len: usize) -> i32;
}
