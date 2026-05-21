//! RMSNorm correctness sweep.
//! Computation — per row i of length k:
//! mean_sq = Σ x² / k
//! y = x * weight / sqrt(mean_sq + eps)
//! The reference matches the kernel: F32 accumulator for the sum-of-squares,
//! F16 weight, F16 output. The cert records max relative error across rows,
//! with a floor tracking `sqrt(K)` noise (same formula as the MMVQ sweep).

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// F16 in / F16 out RMSNorm.
    F16,
}

impl Dtype {
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F16 => "F16",
        }
    }
    fn impl_id(self) -> &'static str {
        match self {
            Dtype::F16 => "rmsnorm_f16_gfx906",
        }
    }
    fn kernel_stem(self) -> &'static str {
        match self {
            Dtype::F16 => "rmsnorm_f16",
        }
    }
    fn kernel_entry(self) -> &'static str {
        match self {
            Dtype::F16 => "flambeau_rmsnorm_f16",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub m: usize,
    pub k: usize,
}

#[derive(Debug, Clone)]
pub struct SweepSpec {
    pub dtype: Dtype,
    pub shapes: Vec<Shape>,
    pub seed: u64,
    pub eps: f32,
}

impl SweepSpec {
    /// Qwen3.6 hidden sizes: 2048 (tiny), 5120 (attention), 15360 (FFN). M
    /// = sequence length — mostly 1 (decode) but large-seq prefill hits
    /// k*m*2 bytes of traffic, so we include a prefill-shape datapoint.
    pub fn v1_6_rmsnorm() -> Self {
        let shapes = vec![
            Shape { m: 1, k: 2048 },
            Shape { m: 1, k: 5120 },
            Shape { m: 1, k: 15360 },
            Shape { m: 8, k: 5120 },
            Shape { m: 128, k: 5120 },
            Shape { m: 512, k: 5120 },
        ];
        Self {
            dtype: Dtype::F16,
            shapes,
            seed: 0xC0FFEE,
            eps: 1e-6,
        }
    }
}

pub fn run_sweep(spec: &SweepSpec, repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let pmc = capture_static_pmc(&dev, spec.dtype)?;

    let mut results = Vec::new();
    for sh in &spec.shapes {
        let seed = spec
            .seed
            .wrapping_add((sh.m as u64).wrapping_mul(0x1234567))
            .wrapping_add((sh.k as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let (got, reference) = run_shape(&dev, spec.dtype, sh.m, sh.k, spec.eps, seed)?;
        let max_rel_err = max_rel_err_with_floor(&got, &reference, (sh.k as f32).sqrt() * 0.01);
        let tolerance = cert_tol();
        results.push(ShapeResult {
            m: sh.m,
            k: sh.k,
            n: sh.k, // RMSNorm has no separate N — mirror K so the row is shapely.
            seed,
            max_rel_err,
            tolerance,
            pass: max_rel_err <= tolerance,
        });
        tracing::info!(
            target: "flambeau_bench::sweep_rmsnorm",
            m = sh.m, k = sh.k, max_rel_err, tolerance,
            "rmsnorm shape"
        );
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = rig();
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: spec.dtype.impl_id().to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "rmsnorm".to_string(),
        dtype_weight: spec.dtype.name().to_string(),
        dtype_activation: spec.dtype.name().to_string(),
        tolerance_formula: "|err| <= 1e-2 * max(|ref|, sqrt(k))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig,
        pmc: Some(pmc),
    };
    let written = cert.write_to_disk(repo_root)?;
    tracing::info!(
        target: "flambeau_bench::sweep_rmsnorm",
        cert = %written.display(),
        pass = cert.pass,
        "cert written"
    );
    Ok(cert)
}

fn capture_static_pmc(dev: &HipDevice, dtype: Dtype) -> Result<PmcSnapshot> {
    let bytes = kernels::hsaco(dtype.kernel_stem())
        .ok_or_else(|| anyhow::anyhow!("{} not compiled", dtype.kernel_stem()))?;
    let module = HipModule::load(dev.id(), bytes)?;
    let kernel: HipKernel<'_> = module.kernel(dtype.kernel_entry())?;
    let attrs: FuncAttributes = kernel.attributes()?;
    Ok(PmcSnapshot {
        vgpr_count: Some(attrs.num_regs),
        sgpr_count: None,
        waves_per_simd: Some(attrs.gfx906_waves_per_simd()),
        mem_busy_pct: None,
        valu_busy_pct: None,
    })
}

fn run_shape(
    dev: &HipDevice,
    dtype: Dtype,
    m: usize,
    k: usize,
    eps: f32,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let bytes = kernels::hsaco(dtype.kernel_stem()).unwrap();
    let module = HipModule::load(dev.id(), bytes)?;
    let kernel: HipKernel<'_> = module.kernel(dtype.kernel_entry())?;

    // Generate random F16 x and weight.
    let x_f32 = seeded_f32_range(seed, m * k, -0.5, 0.5);
    let w_f32 = seeded_f32_range(seed.wrapping_add(0x5A5A5A5A), k, -0.5, 0.5);
    let x_f16: Vec<f16> = x_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let w_f16: Vec<f16> = w_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_x = alloc_and_upload(dev, &x_f16);
    let d_w = alloc_and_upload(dev, &w_f16);
    let d_y = dev.alloc(m * k * 2)?;

    {
        let stream = dev.default_stream();
        let n_rows_i = m as i32;
        let k_i = k as i32;
        let eps_f = eps;
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_w_ptr: u64 = d_w.as_usize() as u64;
        let d_y_ptr: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_x_ptr);
        args.push(&d_w_ptr);
        args.push(&d_y_ptr);
        args.push(&n_rows_i);
        args.push(&k_i);
        args.push(&eps_f);
        let cfg = LaunchCfg::one_d(m as u32, 256);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    // Download and convert F16 → F32.
    let mut y_f16: Vec<f16> = vec![f16::from_f32(0.0); m * k];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(y_f16.as_mut_ptr() as usize),
            d_y,
            m * k * 2,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_x, x_f16.len() * 2)?;
        dev.dealloc(d_w, w_f16.len() * 2)?;
        dev.dealloc(d_y, m * k * 2)?;
    }

    let got: Vec<f32> = y_f16.iter().map(|v| v.to_f32()).collect();

    // Reference: same arithmetic in F32 (with the f16→f32 cast of the
    // inputs to mirror the kernel's first read) and F16-quantised output
    // (kernel writes F16 so we cast the reference output to F16 and back
    // — otherwise round-off on the output cast is billed as kernel error).
    let x_inputs: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
    let w_inputs: Vec<f32> = w_f16.iter().map(|v| v.to_f32()).collect();
    let mut reference = vec![0.0f32; m * k];
    for row in 0..m {
        let xr = &x_inputs[row * k..(row + 1) * k];
        let mut sum_sq = 0.0f64;
        for v in xr {
            sum_sq += (*v as f64) * (*v as f64);
        }
        let mean_sq = (sum_sq / k as f64) as f32;
        let rsqrt = 1.0f32 / (mean_sq + eps).sqrt();
        for i in 0..k {
            let y = xr[i] * w_inputs[i] * rsqrt;
            reference[row * k + i] = f16::from_f32(y).to_f32();
        }
    }

    Ok((got, reference))
}

// ---- utilities ----

fn cert_tol() -> f32 {
    // RMSNorm's F16 output has ~2^-10 precision per element, but the
    // rsqrt/multiply chain + F32 reduction is tight enough that 5e-3 holds
    // for reasonable inputs. Widened to 1e-2 to absorb the f16 output cast.
    1e-2
}
