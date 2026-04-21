//! RMSNorm — F16→F16 and fused F16→Q8_1.
//!
//! Both kernels use 256 threads/row, one block per row. Layout of the Q8_1
//! output is the GGUF-standard `[d (fp16), s (fp16), qs[32] (i8)]` block.

use anyhow::Result;
use flambeau_backend_hip::{HipStream, KernelArgs, LaunchCfg};
use flambeau_core::DevicePtr;

use super::OpsRegistry;

/// `y[i] = (x[i] / sqrt(mean(x*x) + eps)) * weight[i]`. Row-wise over `m`
/// rows of `k` elements each. All tensors F16, row-major, contiguous.
pub fn rmsnorm_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    x: DevicePtr,
    weight: DevicePtr,
    y: DevicePtr,
    m: usize,
    k: usize,
    eps: f32,
) -> Result<()> {
    let module = reg.expect_module("rmsnorm_f16")?;
    let kernel = module.kernel("flambeau_rmsnorm_f16")?;

    let m_i = m as i32;
    let k_i = k as i32;
    let eps_f = eps;
    let x_ptr: u64 = x.as_usize() as u64;
    let w_ptr: u64 = weight.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&m_i);
    args.push(&k_i);
    args.push(&eps_f);
    let cfg = LaunchCfg::one_d(m as u32, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Fused RMSNorm + Q8_1-quantize. Saves the round-trip to HBM between
/// norm-out and the next matmul's activation-quantize step (candle D1).
/// Output is `m * (k / 32)` `flambeau_block_q8_1` blocks, layout matching
/// `flambeau_quant::BlockQ8_1`.
pub fn rmsnorm_quant_q8_1(
    reg: &OpsRegistry,
    stream: &HipStream,
    x: DevicePtr,
    weight: DevicePtr,
    y_q8_1: DevicePtr,
    m: usize,
    k: usize,
    eps: f32,
) -> Result<()> {
    let module = reg.expect_module("rmsnorm_q8_1_fused")?;
    let kernel = module.kernel("flambeau_rmsnorm_q8_1_fused")?;

    let m_i = m as i32;
    let k_i = k as i32;
    let eps_f = eps;
    let x_ptr: u64 = x.as_usize() as u64;
    let w_ptr: u64 = weight.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&m_i);
    args.push(&k_i);
    args.push(&eps_f);
    let cfg = LaunchCfg::one_d(m as u32, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// F32 per-row RMSNorm — F32 weight, F32 input, F32 output. Same math as
/// [`rmsnorm_f16`] but keeps precision in F32 across the GDN ssm_norm step
/// (applied per-head on the F32 state-step output).
pub fn rmsnorm_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    x: DevicePtr,
    weight: DevicePtr,
    y: DevicePtr,
    m: usize,
    k: usize,
    eps: f32,
) -> Result<()> {
    let module = reg.expect_module("rmsnorm_f32")?;
    let kernel = module.kernel("flambeau_rmsnorm_f32")?;
    let m_i = m as i32;
    let k_i = k as i32;
    let x_ptr: u64 = x.as_usize() as u64;
    let w_ptr: u64 = weight.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&m_i);
    args.push(&k_i);
    args.push(&eps);
    let cfg = LaunchCfg::one_d(m as u32, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// L2 normalization along the last dimension. `y[i] = x[i] / sqrt(sum(x^2) + eps)`.
/// F32 in/out. Used by GDN on Q and K before the recurrent state update.
///
/// Launch: one block/row, 256 threads.
pub fn l2_norm_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    x: DevicePtr,
    y: DevicePtr,
    n_rows: usize,
    k: usize,
    eps: f32,
) -> Result<()> {
    let module = reg.expect_module("l2_norm_f32")?;
    let kernel = module.kernel("flambeau_l2_norm_f32")?;
    let n_rows_i = n_rows as i32;
    let k_i = k as i32;
    let eps_f = eps;
    let x_ptr: u64 = x.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_rows_i);
    args.push(&k_i);
    args.push(&eps_f);
    let cfg = LaunchCfg::one_d(n_rows as u32, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Stand-alone activation quantise: F32 → Q8_1 blocks. Used for the
/// first-layer input before RMSNorm's fused variant is in play, and any time
/// we need to quantise an existing F32 tensor on device.
///
/// `n_elems` must be a multiple of 32 (QK8_1). Launches one block per Q8_1
/// super-block, 32 threads/block.
pub fn quantize_q8_1(
    reg: &OpsRegistry,
    stream: &HipStream,
    x_f32: DevicePtr,
    y_q8_1: DevicePtr,
    n_elems: usize,
) -> Result<()> {
    assert_eq!(n_elems % 32, 0, "quantize_q8_1 expects n_elems % 32 == 0");
    let module = reg.expect_module("quantize_q8_1")?;
    let kernel = module.kernel("flambeau_quantize_row_q8_1")?;

    let n_i = n_elems as i32;
    let x_ptr: u64 = x_f32.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let n_blocks = (n_elems / 32) as u32;
    let cfg = LaunchCfg::one_d(n_blocks, 32);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// F16 sibling of [`quantize_q8_1`]. Used by the full-attention forward to
/// skip the host-roundtrip F16→F32 placeholder between swiglu and the
/// output-projection MMVQ.
pub fn quantize_f16_q8_1(
    reg: &OpsRegistry,
    stream: &HipStream,
    x_f16: DevicePtr,
    y_q8_1: DevicePtr,
    n_elems: usize,
) -> Result<()> {
    assert_eq!(n_elems % 32, 0, "quantize_f16_q8_1 expects n_elems % 32 == 0");
    let module = reg.expect_module("quantize_f16_q8_1")?;
    let kernel = module.kernel("flambeau_quantize_row_f16_q8_1")?;
    let n_i = n_elems as i32;
    let x_ptr: u64 = x_f16.as_usize() as u64;
    let y_ptr: u64 = y_q8_1.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let n_blocks = (n_elems / 32) as u32;
    let cfg = LaunchCfg::one_d(n_blocks, 32);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
