//! MLP building blocks — SwiGLU today.
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

/// 3.b.2 — fused `y_f16[i] = (fp16)(silu(a[i]) * b[i])`. Replaces the
/// swiglu_f32 + cast_f32_f16 pair on MoE activation paths.
pub fn swiglu_f32_to_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    a: DevicePtr,
    b: DevicePtr,
    y: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("swiglu_f32_to_f16")?;
    let kernel = module.kernel("flambeau_swiglu_f32_to_f16")?;
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

/// /d — fused `y_q8_1 = quantize_row_q8_1(silu(a) * b)` for F32
/// inputs. Replaces the unfused chain (swiglu_f32 → quantize_q8_1) used by
/// the GDN tail (`forward/gdn.rs`), and the swiglu_f32_to_f16 →
/// quantize_f16_q8_1 chain used by the shared-expert decode
/// (`forward/moe.rs`). One launch + one HBM round-trip saved per layer
/// per token.
/// Grid: one thread block per 32-element Q8_1 block, 32 threads/block.
/// `n` must be a multiple of 32.
pub fn swiglu_f32_to_q8_1(
    reg: &OpsRegistry,
    stream: &HipStream,
    a: DevicePtr,
    b: DevicePtr,
    y_q8_1: DevicePtr,
    n: usize,
) -> Result<()> {
    assert_eq!(n % 32, 0, "swiglu_f32_to_q8_1 expects n % 32 == 0");
    let module = reg.expect_module("swiglu_f32_to_q8_1")?;
    let kernel = module.kernel("flambeau_swiglu_f32_to_q8_1")?;
    let n_i = n as i32;
    let a_ptr: u64 = a.as_usize() as u64;
    let b_ptr: u64 = b.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&a_ptr);
    args.push(&b_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let n_blocks = (n / 32) as u32;
    let cfg = LaunchCfg::one_d(n_blocks, 32);
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

/// F32 sibling of `add_f16`. `y[i] = a[i] + b[i]` in F32,
/// no precision loss. Used for the MTP block's residual additions to
/// keep the residual stream in F32 between matmul output and the next
/// norm input.
pub fn add_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    a: DevicePtr,
    b: DevicePtr,
    y: DevicePtr,
    n: usize,
) -> Result<()> {
    let module = reg.expect_module("add_f32")?;
    let kernel = module.kernel("flambeau_add_f32")?;
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
/// Qwen3.5/3.6 full-attention output gate: plain logistic sigmoid applied
/// to the gate half of the fused Q-projection, then element-wise multiply
/// against the attention output. **Not** SiLU — SiLU is
/// `gate * sigmoid(gate)`, so using `swiglu_f16` here adds an extra factor
/// of `gate` vs llama.cpp's graph (root cause).
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

