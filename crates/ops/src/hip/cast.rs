//! Dtype cast kernels — F32 → F16 today, others as needed.
//!
//! Purpose: decode-path glue. MMVQ accumulates in F32; attention / rmsnorm /
//! swiglu consume F16. Keeping a one-kernel cast here avoids writing an F16
//! accumulator variant of every MMVQ kernel.

use anyhow::Result;
use flambeau_backend_hip::{HipStream, KernelArgs, LaunchCfg};
use flambeau_core::DevicePtr;

use super::OpsRegistry;

/// Pointwise `y[i] = (fp16) x[i]`. No rounding guarantees beyond the HIP
/// compiler's default round-to-nearest-even on `fb_fp16_t`.
pub fn cast_f32_to_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    x_f32: DevicePtr,
    y_f16: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("cast_f32_f16")?;
    let kernel = module.kernel("flambeau_cast_f32_f16")?;
    let n_i = n as i32;
    let x_ptr: u64 = x_f32.as_usize() as u64;
    let y_ptr: u64 = y_f16.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Pointwise `y[i] = (float) x[i]`. Exact up to the F16 source's precision.
pub fn cast_f16_to_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    x_f16: DevicePtr,
    y_f32: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("cast_f16_f32")?;
    let kernel = module.kernel("flambeau_cast_f16_f32")?;
    let n_i = n as i32;
    let x_ptr: u64 = x_f16.as_usize() as u64;
    let y_ptr: u64 = y_f32.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
