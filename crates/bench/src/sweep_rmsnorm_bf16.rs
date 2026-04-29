//! MTP-4-C-3 — BF16 RMSNorm correctness sweep.
//!
//! Computation per row of length k:
//!   mean_sq = Σ x² / k
//!   y       = bf16(x * weight * rsqrt(mean_sq + eps))
//!
//! Reference matches the kernel: F32 reduction of BF16-lifted inputs,
//! F16 weight, BF16 output. Tolerance covers the BF16 output rounding
//! plus the F32 vs parallel-reduction summation order delta.

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
use half::{bf16, f16};

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig, seeded_f32_range};

/// Shapes covering every MTP-norm size on Qwen3.6-27B + a prefill case:
///   `[1, 5120]` — pre_fc_norm_*, input_layernorm, post_attention_layernorm,
///                 mtp.norm
///   `[24, 256]` — q_norm (n_q heads × head_dim)
///   `[4, 256]`  — k_norm
///   `[8, 5120]` / `[128, 5120]` — prefill seq lengths
const SHAPES: &[(usize, usize)] = &[
    (1, 5120),
    (24, 256),
    (4, 256),
    (8, 5120),
    (128, 5120),
];

const EPS: f32 = 1e-6;

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    if device_count().context("hipGetDeviceCount")? < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let kb = kernels::hsaco("rmsnorm_bf16")
        .ok_or_else(|| anyhow::anyhow!("rmsnorm_bf16 not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_rmsnorm_bf16")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    let mut results = Vec::new();
    for &(m, k) in SHAPES {
        let seed = 0xBF16_C0FFu64
            .wrapping_add((m as u64).wrapping_mul(0x1234567))
            .wrapping_add((k as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let (got, reference) = run_shape(&dev, &kernel, m, k, seed)?;
        let max_rel = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt() * 0.01);
        // BF16 output has ~2^-7 (=7.8e-3) relative noise per element. The
        // F32 sum-of-squares reduction adds O(eps * sqrt(k)) on top. 2e-2
        // is the same comfort margin the F16 sweep uses (1e-2 there) lifted
        // to absorb BF16's wider mantissa rounding.
        let tol = 2e-2;
        results.push(ShapeResult {
            m,
            k,
            n: k,
            seed,
            max_rel_err: max_rel,
            tolerance: tol,
            pass: max_rel <= tol,
        });
        tracing::info!(
            target: "flambeau_bench::sweep_rmsnorm_bf16",
            m, k, max_rel, tol,
            "rmsnorm bf16 shape"
        );
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "rmsnorm_bf16_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "rmsnorm".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "BF16".to_string(),
        tolerance_formula: "|err| <= 2e-2 * max(|ref|, sqrt(k) * 0.01)".to_string(),
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
    m: usize,
    k: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let x_f32 = seeded_f32_range(seed, m * k, -0.5, 0.5);
    let w_f32 = seeded_f32_range(seed.wrapping_add(0x5A5A_5A5A), k, -0.5, 0.5);
    // x baked at BF16 precision (matches HBM); weight stays F16.
    let x_bf16: Vec<bf16> = x_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    let w_f16:  Vec<f16>  = w_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_x = alloc_and_upload(dev, &x_bf16);
    let d_w = alloc_and_upload(dev, &w_f16);
    let d_y = dev.alloc(m * k * 2)?;

    {
        let stream = dev.default_stream();
        let m_i = m as i32;
        let k_i = k as i32;
        let eps = EPS;
        let x_ptr: u64 = d_x.as_usize() as u64;
        let w_ptr: u64 = d_w.as_usize() as u64;
        let y_ptr: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&x_ptr);
        args.push(&w_ptr);
        args.push(&y_ptr);
        args.push(&m_i);
        args.push(&k_i);
        args.push(&eps);
        let cfg = LaunchCfg::one_d(m as u32, 256);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut y_bf16 = vec![bf16::from_f32(0.0); m * k];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(y_bf16.as_mut_ptr() as usize),
            d_y,
            m * k * 2,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_x, m * k * 2)?;
        dev.dealloc(d_w, k * 2)?;
        dev.dealloc(d_y, m * k * 2)?;
    }

    let got: Vec<f32> = y_bf16.iter().map(|v| v.to_f32()).collect();

    // Reference: same arithmetic, F32 sum, BF16 output cast (same as kernel).
    let x_lifted: Vec<f32> = x_bf16.iter().map(|v| v.to_f32()).collect();
    let w_lifted: Vec<f32> = w_f16.iter().map(|v| v.to_f32()).collect();
    let mut reference = vec![0.0f32; m * k];
    for row in 0..m {
        let xr = &x_lifted[row * k..(row + 1) * k];
        let mut sum_sq = 0.0f64;
        for v in xr {
            sum_sq += (*v as f64) * (*v as f64);
        }
        let mean_sq = (sum_sq / k as f64) as f32;
        let rsqrt = 1.0f32 / (mean_sq + EPS).sqrt();
        for i in 0..k {
            let y = xr[i] * w_lifted[i] * rsqrt;
            reference[row * k + i] = bf16::from_f32(y).to_f32();
        }
    }

    Ok((got, reference))
}
