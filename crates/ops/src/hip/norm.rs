//! RMSNorm — F16→F16 and fused F16→Q8_1.
//!
//! Both kernels use 256 threads/row, one block per row. Layout of the Q8_1
//! output is the GGUF-standard `[d (fp16), s (fp16), qs[32] (i8)]` block.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "op wrapper — every unsafe block is a `kernel.launch` or `memcpy_async` \
              over `DevicePtr`s validated by the caller; the kernel stem + entry are \
              resolved through the validated registry and the ABI matches the kernels \
              extern-C signature."
)]

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

/// V2.23.a.1 — fused `mid = x_in + delta; mid_norm = rmsnorm(mid) * weight`.
/// Replaces `add_f16` + `rmsnorm_f16` pair at the attention-residual epilogue.
/// Both `mid` and `mid_norm` are needed downstream.
pub fn rmsnorm_f16_add_residual(
    reg: &OpsRegistry,
    stream: &HipStream,
    x_in: DevicePtr,
    delta: DevicePtr,
    weight: DevicePtr,
    mid: DevicePtr,
    mid_norm: DevicePtr,
    m: usize,
    k: usize,
    eps: f32,
) -> Result<()> {
    let module = reg.expect_module("rmsnorm_f16_add_residual")?;
    let kernel = module.kernel("flambeau_rmsnorm_f16_add_residual")?;

    let m_i = m as i32;
    let k_i = k as i32;
    let eps_f = eps;
    let x_ptr: u64 = x_in.as_usize() as u64;
    let d_ptr: u64 = delta.as_usize() as u64;
    let w_ptr: u64 = weight.as_usize() as u64;
    let m_ptr: u64 = mid.as_usize() as u64;
    let n_ptr: u64 = mid_norm.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&d_ptr);
    args.push(&w_ptr);
    args.push(&m_ptr);
    args.push(&n_ptr);
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

/// V2.2.d.P8 — F32 activation → BlockQ8_1Mmq (DS4 layout) for the 4-warp
/// LDS-tiled MMQ prefill path. Output is `[n_big_blocks, total_b]`
/// row-major of 144 B blocks (128 F32 elements per block). Grid =
/// `(ncols/128, total_b)`, block = 128 threads.
pub fn quantize_q8_1_mmq(
    reg: &OpsRegistry,
    stream: &HipStream,
    x_f32: DevicePtr,
    y_q8_1_mmq: DevicePtr,
    ncols: usize,
    total_b: usize,
) -> Result<()> {
    assert_eq!(ncols % 128, 0, "quantize_q8_1_mmq expects ncols % 128 == 0");
    let module = reg.expect_module("quantize_q8_1_mmq")?;
    let kernel = module.kernel("flambeau_quantize_q8_1_mmq")?;
    let ncols_i = ncols as i32;
    let total_b_i = total_b as i32;
    let x_ptr: u64 = x_f32.as_usize() as u64;
    let y_ptr: u64 = y_q8_1_mmq.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&ncols_i);
    args.push(&total_b_i);
    let cfg = LaunchCfg {
        grid: ((ncols / 128) as u32, total_b as u32, 1),
        block: (128, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// F16 sibling of [`quantize_q8_1_mmq`]. Used by the forward prefill path
/// after rmsnorm (which emits F16) to produce the DS4 layout for
/// `mmq_q4_1_4warp_lds` and future K-quant turbo kernels.
pub fn quantize_f16_q8_1_mmq(
    reg: &OpsRegistry,
    stream: &HipStream,
    x_f16: DevicePtr,
    y_q8_1_mmq: DevicePtr,
    ncols: usize,
    total_b: usize,
) -> Result<()> {
    assert_eq!(ncols % 128, 0, "quantize_f16_q8_1_mmq expects ncols % 128 == 0");
    let module = reg.expect_module("quantize_f16_q8_1_mmq")?;
    let kernel = module.kernel("flambeau_quantize_f16_q8_1_mmq")?;
    let ncols_i = ncols as i32;
    let total_b_i = total_b as i32;
    let x_ptr: u64 = x_f16.as_usize() as u64;
    let y_ptr: u64 = y_q8_1_mmq.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&ncols_i);
    args.push(&total_b_i);
    let cfg = LaunchCfg {
        grid: ((ncols / 128) as u32, total_b as u32, 1),
        block: (128, 1, 1),
        shared_bytes: 0,
    };
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

/// V1-BENCH-#116a — F16 → Q8_0 row-quantise. Used to convert K/V
/// projection output (F16) to Q8_0 blocks for `KvCache<Q8Contig>`. Same
/// shape as [`quantize_f16_q8_1`] but writes 18 B / 32 elems (no `s`
/// field).
pub fn quantize_f16_q8_0(
    reg: &OpsRegistry,
    stream: &HipStream,
    x_f16: DevicePtr,
    y_q8_0: DevicePtr,
    n_elems: usize,
) -> Result<()> {
    assert_eq!(n_elems % 32, 0, "quantize_f16_q8_0 expects n_elems % 32 == 0");
    let module = reg.expect_module("quantize_f16_q8_0")?;
    let kernel = module.kernel("flambeau_quantize_row_f16_q8_0")?;
    let n_i = n_elems as i32;
    let x_ptr: u64 = x_f16.as_usize() as u64;
    let y_ptr: u64 = y_q8_0.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    let n_blocks = (n_elems / 32) as u32;
    let cfg = LaunchCfg::one_d(n_blocks, 32);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
