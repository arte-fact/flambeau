//! SwiGLU correctness sweep — pure pointwise silu(gate) * up.

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

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let kernel_bytes =
        kernels::hsaco("swiglu_f16").ok_or_else(|| anyhow::anyhow!("swiglu_f16 not compiled"))?;
    let module = HipModule::load(dev.id(), kernel_bytes)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_swiglu_f16")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // Shapes are flat lengths: `n = m * intermediate_size`. Qwen3.6's
    // intermediate size for FFN expert is 15360; a batch of 128 tokens hits
    // n ≈ 2M. We cover a decode-shaped small case and a prefill-shaped
    // large case.
    let shapes = [(1usize, 2048usize), (1, 15360), (8, 15360), (128, 15360)];

    let mut results = Vec::new();
    for (m, hidden) in shapes {
        let n_elems = m * hidden;
        let seed = 0xC0FFEE ^ (n_elems as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let (got, reference) = run_shape(&dev, &kernel, n_elems, seed)?;
        let max_rel = max_rel_err_with_floor(&got, &reference, 1e-2);
        let tol = 5e-3;
        results.push(ShapeResult {
            m,
            k: hidden,
            n: hidden,
            seed,
            max_rel_err: max_rel,
            tolerance: tol,
            pass: max_rel <= tol,
        });
        tracing::info!(
            target: "flambeau_bench::sweep_swiglu",
            m, hidden, max_rel, tol,
            "swiglu shape"
        );
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = rig();
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "swiglu_f16_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "swiglu".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: "|err| <= 5e-3 * max(|ref|, 1)".to_string(),
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
    n: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let gate_f32 = seeded_f32_range(seed, n, -2.0, 2.0);
    let up_f32 = seeded_f32_range(seed.wrapping_add(0xA5A5A5A5), n, -2.0, 2.0);
    let gate_f16: Vec<f16> = gate_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let up_f16: Vec<f16> = up_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_g = alloc_and_upload(dev, &gate_f16);
    let d_u = alloc_and_upload(dev, &up_f16);
    let d_y = dev.alloc(n * 2)?;

    let stream = dev.default_stream();
    let n_i = n as i32;
    let d_g_ptr: u64 = d_g.as_usize() as u64;
    let d_u_ptr: u64 = d_u.as_usize() as u64;
    let d_y_ptr: u64 = d_y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&d_g_ptr);
    args.push(&d_u_ptr);
    args.push(&d_y_ptr);
    args.push(&n_i);
    let cfg = LaunchCfg::one_d(n.div_ceil(256) as u32, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    stream.synchronize()?;

    let mut y_f16: Vec<f16> = vec![f16::from_f32(0.0); n];
    unsafe {
        dev.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(y_f16.as_mut_ptr() as usize),
            d_y,
            n * 2,
        )?;
    }
    stream.synchronize()?;
    unsafe {
        dev.dealloc(d_g, n * 2)?;
        dev.dealloc(d_u, n * 2)?;
        dev.dealloc(d_y, n * 2)?;
    }

    let got: Vec<f32> = y_f16.iter().map(|v| v.to_f32()).collect();

    // Reference: F16→F32 cast, silu in F32, multiply, F16 round-trip.
    let reference: Vec<f32> = (0..n)
        .map(|i| {
            let g = gate_f16[i].to_f32();
            let u = up_f16[i].to_f32();
            let sig = 1.0f32 / (1.0f32 + (-g).exp());
            let silu = g * sig;
            f16::from_f32(silu * u).to_f32()
        })
        .collect();
    Ok((got, reference))
}
