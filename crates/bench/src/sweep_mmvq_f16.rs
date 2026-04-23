//! V2.21.b — F16 weight × Q8_1 activation MMVQ correctness sweep.
//!
//! Unsloth's UD-Q8_K_XL GGUFs reserve F16 for i-matrix-flagged layers. Sweep
//! validates `flambeau_mmvq_f16_q8_1` against an F32 reference that
//! dequantises the Q8_1 activation back to F32, multiplies in F32 with the
//! F16 weight upcast, and sums. Shapes cover 27B-UD-Q8_K_XL:
//!   * GDN attn_gate: n=6144, k=5120
//!   * GDN ssm_out:   n=5120, k=6144
//!   * synthetic narrow shape for block-count coverage.

#![cfg(feature = "hip")]

use std::path::Path;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{
    device_count, FuncAttributes, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ8_1, QK8_1};
use half::f16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};

const SHAPES: &[(usize, usize)] = &[
    (256, 256),         // smoke
    (6144, 5120),       // Qwen3.6-27B-UD-Q8_K_XL attn_gate
    (5120, 6144),       // Qwen3.6-27B-UD-Q8_K_XL ssm_out
    (248320, 5120),     // Qwen3.6-27B LM head / embeddings (F16 in UD)
];

/// V2.25.a — multi-row variant sweep: exercises `flambeau_mmq_f16_q8_1` at
/// `n_tokens > 1` and compares each row to the single-row F32 reference.
pub fn run_mmq_sweep(repo_root: &Path) -> Result<Cert> {
    if device_count().context("hipGetDeviceCount")? < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("mmq_f16_q8_1").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_mmq_f16_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // (n_rows, k, n_tokens)
    let cases = [
        (256usize, 256usize, 1usize),   // collapse to mmvq case
        (256, 256, 8),                   // small smoke
        (6144, 5120, 16),                // 27B-UD-Q8_K_XL attn_gate, prefill-like
        (5120, 6144, 8),                 // 27B-UD-Q8_K_XL ssm_out
    ];
    let mut results = Vec::new();
    for (n, k, n_tokens) in cases {
        let seed = 0x25A0u64 ^ (n as u64 * 7919) ^ (k as u64 * 101) ^ (n_tokens as u64 * 37);
        let (got, reference) = run_mmq_shape(&dev, &kernel, n, k, n_tokens, seed)?;
        let max_rel = max_rel_err_multi(&got, &reference, k);
        let tol = 3e-2;
        results.push(ShapeResult {
            m: n_tokens,
            k,
            n,
            seed,
            max_rel_err: max_rel,
            tolerance: tol,
            pass: max_rel <= tol,
        });
    }
    let pass = results.iter().all(|r| r.pass);
    let rig = format!("{}-gfx906", hostname().unwrap_or_else(|| "unknown".into()));
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "mmq_f16_q8_1_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "qmatmul".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k) * q8_step)".to_string(),
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

fn run_mmq_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    n: usize,
    k: usize,
    n_tokens: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let w_f32 = seeded_f32(seed, n * k);
    let x_f32 = seeded_f32(seed.wrapping_add(0xA1), n_tokens * k);
    let w_f16: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();
    // Quantise each activation row independently.
    let mut x_q: Vec<BlockQ8_1> = Vec::with_capacity(n_tokens * (k / QK8_1));
    for row in 0..n_tokens {
        x_q.extend(quantize_row_q8_1(&x_f32[row * k..(row + 1) * k]));
    }

    let d_w = alloc_upload(dev, &w_f16);
    let d_x = alloc_upload(dev, &x_q);
    let d_out = dev.alloc(n_tokens * n * 4)?;

    {
        let stream = dev.default_stream();
        let n_rows_i = n as i32;
        let n_tokens_i = n_tokens as i32;
        let n_blocks_i = (k / 32) as i32;
        let w_ptr: u64 = d_w.as_usize() as u64;
        let y_ptr: u64 = d_x.as_usize() as u64;
        let o_ptr: u64 = d_out.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&w_ptr);
        args.push(&y_ptr);
        args.push(&o_ptr);
        args.push(&n_rows_i);
        args.push(&n_tokens_i);
        args.push(&n_blocks_i);
        let cfg = LaunchCfg {
            grid: (n as u32, n_tokens as u32, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_tokens * n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_out,
            n_tokens * n * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_w, w_f16.len() * 2)?;
        dev.dealloc(d_x, x_q.len() * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_out, n_tokens * n * 4)?;
    }

    // Reference: F16 weight × Q8_1-dequant activation in F32, per row.
    let mut reference = vec![0.0f32; n_tokens * n];
    let w_back: Vec<f32> = w_f16.iter().map(|v| v.to_f32()).collect();
    for tok in 0..n_tokens {
        let blocks = &x_q[tok * (k / QK8_1)..(tok + 1) * (k / QK8_1)];
        let x_dequant: Vec<f32> = blocks
            .iter()
            .flat_map(|b| {
                let d = b.d.to_f32();
                (0..QK8_1).map(move |i| d * b.qs[i] as f32)
            })
            .collect();
        for row in 0..n {
            let mut acc = 0.0f64;
            for j in 0..k {
                acc += (w_back[row * k + j] * x_dequant[j]) as f64;
            }
            reference[tok * n + row] = acc as f32;
        }
    }
    Ok((got, reference))
}

fn max_rel_err_multi(got: &[f32], reference: &[f32], k: usize) -> f32 {
    let abs_floor = (k as f32).sqrt() * 0.01;
    got.iter()
        .zip(reference)
        .map(|(g, r)| (g - r).abs() / r.abs().max(abs_floor))
        .fold(0.0f32, f32::max)
}

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    if device_count().context("hipGetDeviceCount")? < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;
    let kb = kernels::hsaco("mmvq_f16_q8_1").unwrap();
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_mmvq_f16_q8_1")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    let mut results = Vec::new();
    for &(n, k) in SHAPES {
        let seed = 0xF16FACE ^ (n as u64 * 7919) ^ (k as u64 * 101);
        let (got, reference) = run_shape(&dev, &kernel, n, k, seed)?;
        let max_rel = max_rel_err(&got, &reference, k);
        // F16 × Q8_1 dequant → F32: error floor = Q8 step (1/127) × k-sum
        // noise + F16 rounding on weight side. 3e-2 is comfortably above the
        // measured noise at k=5120/6144 and far below a usable signal bar.
        let tol = 3e-2;
        results.push(ShapeResult {
            m: 1,
            k,
            n,
            seed,
            max_rel_err: max_rel,
            tolerance: tol,
            pass: max_rel <= tol,
        });
        tracing::info!(
            target: "flambeau_bench::sweep_mmvq_f16",
            n, k, max_rel, tol,
            "mmvq f16 × q8_1 shape"
        );
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = format!("{}-gfx906", hostname().unwrap_or_else(|| "unknown".into()));
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "mmvq_f16_q8_1_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "qmatmul_mmvq".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k) * q8_step)".to_string(),
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

fn quantize_row_q8_1(xs: &[f32]) -> Vec<BlockQ8_1> {
    assert_eq!(xs.len() % QK8_1, 0);
    let nb = xs.len() / QK8_1;
    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        let b = &xs[i * QK8_1..(i + 1) * QK8_1];
        let amax = b.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; QK8_1];
        let mut sum_i: i32 = 0;
        for (j, &v) in b.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            qs[j] = q;
            sum_i += q as i32;
        }
        out.push(BlockQ8_1 {
            d: f16::from_f32(d),
            s: f16::from_f32(d * sum_i as f32),
            qs,
        });
    }
    out
}

fn run_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    n: usize,
    k: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let w_f32 = seeded_f32(seed, n * k);
    let x_f32 = seeded_f32(seed.wrapping_add(0xA1), k);
    let w_f16: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let x_q = quantize_row_q8_1(&x_f32);

    let d_w = alloc_upload(dev, &w_f16);
    let d_x = alloc_upload(dev, &x_q);
    let d_out = dev.alloc(n * 4)?;

    {
        let stream = dev.default_stream();
        let n_rows_i = n as i32;
        let n_blocks_i = (k / 32) as i32;
        let w_ptr: u64 = d_w.as_usize() as u64;
        let y_ptr: u64 = d_x.as_usize() as u64;
        let o_ptr: u64 = d_out.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&w_ptr);
        args.push(&y_ptr);
        args.push(&o_ptr);
        args.push(&n_rows_i);
        args.push(&n_blocks_i);
        let cfg = LaunchCfg::one_d(n as u32, 256);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_out,
            n * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_w, w_f16.len() * 2)?;
        dev.dealloc(d_x, x_q.len() * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_out, n * 4)?;
    }

    // Reference: F16 weight × Q8_1-dequant activation in F32.
    let mut reference = vec![0.0f32; n];
    let w_back: Vec<f32> = w_f16.iter().map(|v| v.to_f32()).collect();
    let x_dequant: Vec<f32> = x_q.iter()
        .flat_map(|b| {
            let d = b.d.to_f32();
            (0..QK8_1).map(move |i| d * b.qs[i] as f32)
        })
        .collect();
    for row in 0..n {
        let mut acc = 0.0f64;
        for j in 0..k {
            acc += (w_back[row * k + j] * x_dequant[j]) as f64;
        }
        reference[row] = acc as f32;
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

fn alloc_upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
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

fn max_rel_err(got: &[f32], reference: &[f32], k: usize) -> f32 {
    let abs_floor = (k as f32).sqrt() * 0.01;
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
