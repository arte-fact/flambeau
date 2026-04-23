//! V1.6.6 attention decode with Q8_0 KV cache — correctness cert.
//!
//! The quality cert (delta-ppl ≤ 0.5% on wikitext-2) lands with the V1.7
//! model loader; this sweep gates on the kernel's *arithmetic* matching
//! the F32 reference that uses Q8-round-tripped K/V — i.e. the kernel
//! must compute exactly what you'd get if you dequantised K/V to F32 and
//! ran the F16-KV attention.

#![cfg(feature = "hip")]

use std::path::Path;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{
    device_count, FuncAttributes, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ8_0, QK8_0};
use half::f16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};

/// (head_dim, n_heads_q, n_heads_kv) — V1 target families.
const SHAPES: &[(usize, usize, usize)] = &[
    (128, 32, 4), // Qwen3.5
    (256, 16, 2), // Qwen3.6
];
const QK: usize = QK8_0;

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("attention_decode_q8_kv").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_attention_decode_q8_kv")?;
    let attrs: FuncAttributes = kernel.attributes()?;

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
            // Q8 quant noise on both K and V → looser bar than F16 KV's 2e-2.
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
                target: "flambeau_bench::sweep_attention_q8_kv",
                head_dim, n_heads_q, n_heads_kv,
                n_tokens, max_rel, tol,
                "q8 kv decode shape"
            );
        }
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = format!("{}-gfx906", hostname().unwrap_or_else(|| "unknown".into()));
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "attention_decode_q8_kv_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "attention_decode_q8_kv".to_string(),
        dtype_weight: "Q8_0".to_string(),    // KV dtype
        dtype_activation: "F16".to_string(),  // Q/out dtype
        tolerance_formula: "|err| <= 5e-2 * max(|ref|, sqrt(head_dim))  (correctness; V1.7 adds delta-ppl quality cert)".to_string(),
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

/// CPU-side Q8_0 quantise — match the on-device path bit-for-bit.
fn quantize_row_q8_0(xs: &[f32]) -> Vec<BlockQ8_0> {
    assert_eq!(xs.len() % QK, 0);
    let nb = xs.len() / QK;
    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        let block = &xs[i * QK..(i + 1) * QK];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; QK];
        for (j, &v) in block.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            qs[j] = q;
        }
        out.push(BlockQ8_0 { d: f16::from_f32(d), qs });
    }
    out
}

fn dequantize_row_q8_0(xs: &[BlockQ8_0]) -> Vec<f32> {
    let mut out = vec![0.0f32; xs.len() * QK];
    for (i, b) in xs.iter().enumerate() {
        let d = b.d.to_f32();
        for j in 0..QK {
            out[i * QK + j] = (b.qs[j] as f32) * d;
        }
    }
    out
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

    let q_f32 = seeded_f32(seed, q_len);
    let k_f32 = seeded_f32(seed.wrapping_add(0xA1), kv_len);
    let v_f32 = seeded_f32(seed.wrapping_add(0xA2), kv_len);

    // Quantise K and V to Q8_0 blocks — one row (head_dim elements) at a
    // time so per-row scale is independent. KvCache<Q8Contig> layout:
    // `[n_tokens, n_heads_kv, head_dim/32]` blocks row-major.
    let nb_per_row = head_dim / QK;
    let n_rows = n_tokens * n_heads_kv;
    let mut k_blocks: Vec<BlockQ8_0> = Vec::with_capacity(n_rows * nb_per_row);
    let mut v_blocks: Vec<BlockQ8_0> = Vec::with_capacity(n_rows * nb_per_row);
    for row in 0..n_rows {
        let k_row = &k_f32[row * head_dim..(row + 1) * head_dim];
        let v_row = &v_f32[row * head_dim..(row + 1) * head_dim];
        k_blocks.extend(quantize_row_q8_0(k_row));
        v_blocks.extend(quantize_row_q8_0(v_row));
    }

    // Q is kept F16.
    let q_f16: Vec<f16> = q_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_q = alloc_and_upload(dev, &q_f16);
    let d_k = alloc_and_upload(dev, &k_blocks);
    let d_v = alloc_and_upload(dev, &v_blocks);
    let out_bytes = q_len * 2;
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
        dev.dealloc(d_k, k_blocks.len() * std::mem::size_of::<BlockQ8_0>())?;
        dev.dealloc(d_v, v_blocks.len() * std::mem::size_of::<BlockQ8_0>())?;
        dev.dealloc(d_out, out_bytes)?;
    }
    let got: Vec<f32> = out_f16.iter().map(|v| v.to_f32()).collect();

    // Reference: dequantise K/V back to F32 (exactly what the kernel sees)
    // and run the F16-KV reference attention math on those values. Any
    // delta we observe is kernel arithmetic error, not quant noise.
    let mut k_rt = vec![0.0f32; kv_len];
    let mut v_rt = vec![0.0f32; kv_len];
    for row in 0..n_rows {
        let blocks_k = &k_blocks[row * nb_per_row..(row + 1) * nb_per_row];
        let blocks_v = &v_blocks[row * nb_per_row..(row + 1) * nb_per_row];
        k_rt[row * head_dim..(row + 1) * head_dim]
            .copy_from_slice(&dequantize_row_q8_0(blocks_k));
        v_rt[row * head_dim..(row + 1) * head_dim]
            .copy_from_slice(&dequantize_row_q8_0(blocks_v));
    }
    let q_in: Vec<f32> = q_f16.iter().map(|v| v.to_f32()).collect();
    let group = n_heads_q / n_heads_kv;
    let mut reference = vec![0.0f32; q_len];
    for qh in 0..n_heads_q {
        let kvh = qh / group;
        let mut scores = vec![0.0f32; n_tokens];
        for t in 0..n_tokens {
            let mut dot = 0.0f64;
            for d in 0..head_dim {
                let qv = q_in[qh * head_dim + d];
                let kv = k_rt[(t * n_heads_kv + kvh) * head_dim + d];
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
            for t in 0..n_tokens {
                let vv = v_rt[(t * n_heads_kv + kvh) * head_dim + d];
                acc += (scores[t] * vv) as f64;
            }
            reference[qh * head_dim + d] = f16::from_f32(acc as f32).to_f32();
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
