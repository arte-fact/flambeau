//! Dtype cast kernels — F32 → F16 today, others as needed.
//! Purpose: decode-path glue. MMVQ accumulates in F32; attention / rmsnorm /
//! swiglu consume F16. Keeping a one-kernel cast here avoids writing an F16
//! accumulator variant of every MMVQ kernel.
//!
//! Backend-generic over [`KernelBackend`]: one body serves HIP and CUDA.

use anyhow::Result;
use flambeau_core::{Device, DevicePtr, KernelBackend, LaunchArgs, LaunchCfg};

use crate::Registry;

/// Pointwise `y[i] = (fp16) x[i]`. Round-to-nearest-even per the compiler's
/// default on the backend's fp16 type.
pub fn cast_f32_to_f16<B: KernelBackend>(
    reg: &Registry<B>,
    stream: &<B::Device as Device>::Stream,
    x_f32: DevicePtr,
    y_f16: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("cast_f32_f16")?;
    let kernel = B::kernel(module, "flambeau_cast_f32_f16")?;
    let n_i = n as i32;
    let x_ptr: u64 = x_f32.as_usize() as u64;
    let y_ptr: u64 = y_f16.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&x_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&n_i);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}

/// Pointwise `y[i] = (float) x[i]`. Exact up to the F16 source's precision.
pub fn cast_f16_to_f32<B: KernelBackend>(
    reg: &Registry<B>,
    stream: &<B::Device as Device>::Stream,
    x_f16: DevicePtr,
    y_f32: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("cast_f16_f32")?;
    let kernel = B::kernel(module, "flambeau_cast_f16_f32")?;
    let n_i = n as i32;
    let x_ptr: u64 = x_f16.as_usize() as u64;
    let y_ptr: u64 = y_f32.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&x_ptr);
    args.push_arg(&y_ptr);
    args.push_arg(&n_i);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}
