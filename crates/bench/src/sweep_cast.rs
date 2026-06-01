//! b cast_f32_f16 correctness sweep — single-kernel pointwise cast.
//! Reference: host F32→f16 round-trip (`half::f16::from_f32`). We expect
//! bit-exact match on every element: both the device and the host cast
//! go through round-to-nearest-even with identical rounding modes on the
//! values produced by `seeded_f32`.
//! extends this with four BF16 cast sweeps (F32↔BF16, F16↔BF16),
//! same bit-exact-vs-host pattern.

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
use crate::harness::{alloc_and_upload, rig, seeded_f32_range};

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let kb = kernels::hsaco("cast_f32_f16")
        .ok_or_else(|| anyhow::anyhow!("cast_f32_f16 not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel("flambeau_cast_f32_f16")?;
    let attrs: FuncAttributes = kernel.attributes()?;

    // Shapes covering Qwen3.6 decode-path tensor sizes:
    // - 2048: hidden
    // - 4096: per-head projection × n_heads (post-split gate size)
    // - 8192: fused (Q, gate) output width
    // - 64: odd small, checks tail-handling
    // - 1024: covers shapes that don't divide 256 cleanly
    let shapes = [64usize, 1024, 2048, 4096, 8192];
    let mut results = Vec::new();
    for n_elems in shapes {
        let seed = 0xC0FFEE ^ ((n_elems as u64) * 31 + 7);
        let err = run_shape(&dev, &kernel, n_elems, seed)?;
        results.push(ShapeResult {
            m: n_elems,
            k: 1,
            n: 1,
            seed,
            max_rel_err: err,
            tolerance: 0.0,
            pass: err == 0.0,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = rig();
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "cast_f32_f16_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "cast_f32_f16".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: "exact bit match vs half::f16::from_f32".to_string(),
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

fn run_shape(dev: &HipDevice, kernel: &HipKernel<'_>, n_elems: usize, seed: u64) -> Result<f32> {
    let x = seeded_f32_range(seed, n_elems, -0.5, 0.5);
    let reference: Vec<f16> = x.iter().map(|v| f16::from_f32(*v)).collect();

    let d_x = alloc_and_upload(dev, &x);
    let d_y = dev.alloc(n_elems * 2)?;
    {
        let stream = dev.default_stream();
        let n_i = n_elems as i32;
        let x_ptr: u64 = d_x.as_usize() as u64;
        let y_ptr: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&x_ptr);
        args.push(&y_ptr);
        args.push(&n_i);
        let cfg = LaunchCfg::one_d((n_elems as u32).div_ceil(256), 256);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }
    let mut got = vec![f16::from_f32(0.0); n_elems];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_y,
            n_elems * 2,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_x, n_elems * 4)?;
        dev.dealloc(d_y, n_elems * 2)?;
    }

    // Bit-exact comparison — any mismatch lifts the score above 0.
    let mut diff_bits = 0u32;
    for (a, b) in got.iter().zip(&reference) {
        if a.to_bits() != b.to_bits() {
            diff_bits += 1;
        }
    }
    Ok(diff_bits as f32)
}

// ── BF16 cast sweeps ────────────────────────────────────────

/// Common shape set for BF16 cast sweeps; mirrors `run_sweep` above.
const BF16_SHAPES: [usize; 5] = [64, 1024, 2048, 4096, 8192];

/// Generic single-kernel cast harness. The two type parameters are the
/// input/output element types as on-device, with companion `convert`
/// closure producing the host reference. Bit-exact vs host.
fn run_cast_sweep<TIn, TOut>(
    repo_root: &Path,
    stem: &'static str,
    entry: &'static str,
    impl_id: &'static str,
    op: &'static str,
    dtype_weight: &'static str,
    dtype_activation: &'static str,
    tolerance_formula: &'static str,
    seed_pepper: u64,
    inputs: impl Fn(u64, usize) -> Vec<TIn>,
    reference: impl Fn(&[TIn]) -> Vec<TOut>,
    bits_in: impl Fn(&TIn) -> u32,
    bits_out: impl Fn(&TOut) -> u32,
) -> Result<Cert>
where
    TIn: Copy + Default,
    TOut: Copy + Default,
{
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let kb = kernels::hsaco(stem).ok_or_else(|| anyhow::anyhow!("{stem} not compiled"))?;
    let module = HipModule::load(dev.id(), kb)?;
    let kernel: HipKernel<'_> = module.kernel(entry)?;
    let attrs: FuncAttributes = kernel.attributes()?;

    let in_size = std::mem::size_of::<TIn>();
    let out_size = std::mem::size_of::<TOut>();

    let mut results = Vec::new();
    for n_elems in BF16_SHAPES {
        let seed = seed_pepper ^ ((n_elems as u64) * 31 + 7);
        let x = inputs(seed, n_elems);
        let ref_out = reference(&x);

        // Upload input.
        let d_x = dev.alloc(n_elems * in_size)?;
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::HostToDevice,
                d_x,
                DevicePtr(x.as_ptr() as usize),
                n_elems * in_size,
            )?;
        }
        let d_y = dev.alloc(n_elems * out_size)?;

        let stream = dev.default_stream();
        let n_i = n_elems as i32;
        let x_ptr: u64 = d_x.as_usize() as u64;
        let y_ptr: u64 = d_y.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&x_ptr);
        args.push(&y_ptr);
        args.push(&n_i);
        let cfg = LaunchCfg::one_d((n_elems as u32).div_ceil(256), 256);
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;

        let mut got: Vec<TOut> = vec![TOut::default(); n_elems];
        unsafe {
            dev.memcpy_async(
                stream,
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                d_y,
                n_elems * out_size,
            )?;
        }
        stream.synchronize()?;
        unsafe {
            dev.dealloc(d_x, n_elems * in_size)?;
            dev.dealloc(d_y, n_elems * out_size)?;
        }
        let _ = (&bits_in,); // unused but kept for symmetry / future debug

        let mut diff_bits = 0u32;
        for (g, r) in got.iter().zip(&ref_out) {
            if bits_out(g) != bits_out(r) {
                diff_bits += 1;
            }
        }
        results.push(ShapeResult {
            m: n_elems,
            k: 1,
            n: 1,
            seed,
            max_rel_err: diff_bits as f32,
            tolerance: 0.0,
            pass: diff_bits == 0,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: impl_id.to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: op.to_string(),
        dtype_weight: dtype_weight.to_string(),
        dtype_activation: dtype_activation.to_string(),
        tolerance_formula: tolerance_formula.to_string(),
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
