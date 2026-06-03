//! Masked + scaled softmax.

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

/// Row-wise softmax with fused `scale` and optional F16 additive `mask`.
/// Shape: `scores[m, k]`, `mask[m, k]` (F16, `-inf` in excluded positions)
/// or `DevicePtr::NULL` for no mask. Output `out[m, k]` F16.
/// Launch: one block per row, 256 threads/row (two-pass online softmax).
pub fn softmax_masked_f16(
    ctx: crate::OpCtx<'_>,
    bufs: crate::SoftmaxMaskedBuffers,
    knobs: crate::SoftmaxMaskedKnobs,
) -> Result<()> {
    let crate::OpCtx { reg, stream } = ctx;
    let crate::SoftmaxMaskedBuffers { scores, mask, out } = bufs;
    let crate::SoftmaxMaskedKnobs { m, k, scale } = knobs;
    let module = reg.expect_module("softmax_masked_f16")?;
    let kernel = module.kernel("flambeau_softmax_masked_f16")?;

    let m_i = m as i32;
    let k_i = k as i32;
    let scale_f = scale;
    let s_ptr: u64 = scores.as_usize() as u64;
    let m_ptr: u64 = mask.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&s_ptr);
    args.push(&m_ptr);
    args.push(&o_ptr);
    args.push(&m_i);
    args.push(&k_i);
    args.push(&scale_f);
    let cfg = LaunchCfg::one_d(m as u32, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
