//! V1.6.3 attention decode (F16 KV, GQA) correctness sweep.
//!
//! Covers the two V1 head_dim values:
//! - head_dim=128, GQA-32/4 — Qwen3.5 family.
//! - head_dim=256, GQA-16/2 — Qwen3.6 family.
//!
//! Sequence lengths cover early context (16) through long context
//! (4096 — the usual decode sweet spot).

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

/// (head_dim, n_heads_q, n_heads_kv) — V1 target families + tests.
const SHAPES: &[(usize, usize, usize)] = &[
    (64, 4, 1),    // synthetic smoke-test shape (head_dim=64, 1 warp path)
    (128, 32, 4), // Qwen3.5 / GQA-8
    (256, 16, 2), // Qwen3.6 / GQA-8
];

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("attention_decode_f16").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_attention_decode_f16")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // Sequence lengths representative of Qwen3.x decode: short / medium / long.
    let contexts = [16usize, 128, 1024, 4096];
    let mut results = Vec::new();
    for &(head_dim, n_heads_q, n_heads_kv) in SHAPES {
        for n_tokens in contexts {
            let seed = 0xDECADE
                ^ (head_dim as u64 * 7919)
                ^ (n_tokens as u64 * 101);
            let (got, reference) =
                run_shape(&dev, &kernel, head_dim, n_heads_q, n_heads_kv, n_tokens, seed)?;
            let max_rel = max_rel_err(&got, &reference, head_dim);
            // Online softmax + F16 output round-trip + F32 reference comparison.
            // 2e-2 at n_tokens=4096 is on the order of F16's 2^-10 precision
            // compounded by the exp/sum chain; well below a usable bar.
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
                target: "flambeau_bench::sweep_attention",
                head_dim, n_heads_q, n_heads_kv,
                n_tokens, max_rel, tol,
                "attention decode shape"
            );
        }
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = format!("{}-gfx906", hostname().unwrap_or_else(|| "unknown".into()));
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "attention_decode_f16_gfx906".to_string(),
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
    // Random Q / K / V.
    let q_len = n_heads_q * head_dim;
    let kv_len = n_tokens * n_heads_kv * head_dim;
    let q_f32 = seeded_f32(seed, q_len);
    let k_f32 = seeded_f32(seed.wrapping_add(0xA1), kv_len);
    let v_f32 = seeded_f32(seed.wrapping_add(0xA2), kv_len);
    let q_f16: Vec<f16> = q_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let k_f16: Vec<f16> = k_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let v_f16: Vec<f16> = v_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_q = alloc_and_upload(dev, &q_f16);
    let d_k = alloc_and_upload(dev, &k_f16);
    let d_v = alloc_and_upload(dev, &v_f16);
    let out_bytes = n_heads_q * head_dim * 2;
    let d_out = dev.alloc(out_bytes)?;

    let scale = 1.0 / (head_dim as f32).sqrt();
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

    let mut out_f16: Vec<f16> = vec![f16::from_f32(0.0); n_heads_q * head_dim];
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

    // Reference: decomposed F32 attention in 3 passes (score, softmax, V-mul).
    let group = n_heads_q / n_heads_kv;
    let mut reference = vec![0.0f32; n_heads_q * head_dim];
    let q_inputs: Vec<f32> = q_f16.iter().map(|v| v.to_f32()).collect();
    let k_inputs: Vec<f32> = k_f16.iter().map(|v| v.to_f32()).collect();
    let v_inputs: Vec<f32> = v_f16.iter().map(|v| v.to_f32()).collect();
    for qh in 0..n_heads_q {
        let kvh = qh / group;
        // Scores.
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
        // Softmax.
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
        // Sum_t w_t * V[t].
        for d in 0..head_dim {
            let mut acc = 0.0f64;
            for t in 0..n_tokens {
                let vv = v_inputs[(t * n_heads_kv + kvh) * head_dim + d];
                acc += (scores[t] * vv) as f64;
            }
            reference[qh * head_dim + d] =
                f16::from_f32(acc as f32).to_f32();
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
            (u as f32 / u32::MAX as f32) - 0.5
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
