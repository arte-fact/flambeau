//! Dtype cast kernels — F32 → F16 today, others as needed.
//! Purpose: decode-path glue. MMVQ accumulates in F32; attention / rmsnorm /
//! swiglu consume F16. Keeping a one-kernel cast here avoids writing an F16
//! accumulator variant of every MMVQ kernel.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "op wrapper — every unsafe block is a kernel.launch or memcpy_async over \
              DevicePtrs validated by the caller; the kernel stem + entry are resolved \
              through the validated registry and the ABI matches the kernels extern-C \
              signature."
)]

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

/// pointwise `y[i] = (bfloat16) x[i]` with round-to-nearest-even
/// on the dropped F32 mantissa bits. NaN preserved as quiet NaN.
pub fn cast_f32_to_bf16(
    reg: &OpsRegistry,
    stream: &HipStream,
    x_f32: DevicePtr,
    y_bf16: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("cast_f32_bf16")?;
    let kernel = module.kernel("flambeau_cast_f32_bf16")?;
    let n_i = n as i32;
    let x_ptr: u64 = x_f32.as_usize() as u64;
    let y_ptr: u64 = y_bf16.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// pointwise `y[i] = (float) x[i]`. Lossless bit-shift; BF16
/// fits exactly into F32's upper half.
pub fn cast_bf16_to_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    x_bf16: DevicePtr,
    y_f32: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("cast_bf16_f32")?;
    let kernel = module.kernel("flambeau_cast_bf16_f32")?;
    let n_i = n as i32;
    let x_ptr: u64 = x_bf16.as_usize() as u64;
    let y_ptr: u64 = y_f32.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// pointwise `y[i] = (bfloat16) (float) x[i]` via F32. F16 fits
/// inside BF16's exponent range, so no overflow; BF16 has 3 fewer mantissa
/// bits than F16, so the conversion rounds.
pub fn cast_f16_to_bf16(
    reg: &OpsRegistry,
    stream: &HipStream,
    x_f16: DevicePtr,
    y_bf16: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("cast_f16_bf16")?;
    let kernel = module.kernel("flambeau_cast_f16_bf16")?;
    let n_i = n as i32;
    let x_ptr: u64 = x_f16.as_usize() as u64;
    let y_ptr: u64 = y_bf16.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// pointwise `y[i] = (fp16) (float) x[i]` via F32. BF16's
/// 8-bit exponent saturates F16's 5-bit exponent: `|x| > 65504` → ±Inf,
/// `|x| < 6.1e-5` → subnormal/zero.
pub fn cast_bf16_to_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    x_bf16: DevicePtr,
    y_f16: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("cast_bf16_f16")?;
    let kernel = module.kernel("flambeau_cast_bf16_f16")?;
    let n_i = n as i32;
    let x_ptr: u64 = x_bf16.as_usize() as u64;
    let y_ptr: u64 = y_f16.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
