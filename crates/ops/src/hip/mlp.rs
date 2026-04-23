//! MLP building blocks — SwiGLU today.
//!
//! Dense gate/up/down matmul itself lives under [`super::qmatmul`]; this
//! module is for the non-matmul pointwise piece.

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

/// F32 standalone `y[i] = silu(x[i]) = x / (1 + exp(-x))`. Used by the GDN
/// path on `silu(conv_out)` before QKV split — everything stays in F32 so
/// the recurrence doesn't lose precision on the conv output.
pub fn silu_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    x: DevicePtr,
    y: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("silu_f32")?;
    let kernel = module.kernel("flambeau_silu_f32")?;
    let n_i = n as i32;
    let x_ptr: u64 = x.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// F32 variant of [`swiglu_f16`]. `y = silu(a) * b`. Used by the GDN path
/// for `gated = silu(z) * out_normed` where both inputs stay F32 from the
/// recurrent state step.
pub fn swiglu_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    a: DevicePtr,
    b: DevicePtr,
    y: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("swiglu_f32")?;
    let kernel = module.kernel("flambeau_swiglu_f32")?;
    let n_i = n as i32;
    let a_ptr: u64 = a.as_usize() as u64;
    let b_ptr: u64 = b.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&a_ptr);
    args.push(&b_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Pointwise `y[i] = x[i] * scale`. Used by the GDN path to apply the
/// attention scale `1 / sqrt(head_k_dim)` to Q before the state step.
pub fn scale_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    x: DevicePtr,
    y: DevicePtr,
    n: usize,
    scale: f32,
) -> Result<()> {
    let module = reg.expect_module("scale_f32")?;
    let kernel = module.kernel("flambeau_scale_f32")?;
    let n_i = n as i32;
    let x_ptr: u64 = x.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    args.push(&scale);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Pointwise `y = a + b` in F16. Residual fan-in primitive — used between
/// layers in `forward_one_token` to sum the pre-layer residual with each
/// per-layer delta, and to merge the shared-expert contribution into the
/// routed-MoE output.
pub fn add_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    a: DevicePtr,
    b: DevicePtr,
    y: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("add_f16")?;
    let kernel = module.kernel("flambeau_add_f16")?;
    let n_i = n as i32;
    let a_ptr: u64 = a.as_usize() as u64;
    let b_ptr: u64 = b.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&a_ptr);
    args.push(&b_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// `y = silu(gate) * up`, pointwise. All three tensors are F16, flat length
/// `n` (usually `m * hidden`). In-place-safe if the caller wants `y == gate`
/// or `y == up` at the cost of the kernel reading before it writes that lane.
///
/// Kernel: `flambeau_swiglu_f16`. One thread per output element, 256
/// threads/block, 1D grid.
pub fn swiglu_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    gate: DevicePtr,
    up: DevicePtr,
    y: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("swiglu_f16")?;
    let kernel = module.kernel("flambeau_swiglu_f16")?;

    let n_i = n as i32;
    let g_ptr: u64 = gate.as_usize() as u64;
    let u_ptr: u64 = up.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Pointwise `y[i] = sigmoid(gate[i]) * x[i]`, F16 in/out.
///
/// Qwen3.5/3.6 full-attention output gate: plain logistic sigmoid applied
/// to the gate half of the fused Q-projection, then element-wise multiply
/// against the attention output. **Not** SiLU — SiLU is
/// `gate * sigmoid(gate)`, so using `swiglu_f16` here adds an extra factor
/// of `gate` vs llama.cpp's graph (V1.7.4.b root cause).
pub fn sigmoid_mul_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    gate: DevicePtr,
    x: DevicePtr,
    y: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("sigmoid_mul_f16")?;
    let kernel = module.kernel("flambeau_sigmoid_mul_f16")?;

    let n_i = n as i32;
    let g_ptr: u64 = gate.as_usize() as u64;
    let x_ptr: u64 = x.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&g_ptr);
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
