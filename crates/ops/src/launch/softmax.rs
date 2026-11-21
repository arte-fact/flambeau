//! Masked + scaled softmax.

use anyhow::Result;
use flambeau_core::{LaunchArgs, LaunchCfg};


/// Row-wise softmax with fused `scale` and optional F16 additive `mask`.
/// Shape: `scores[m, k]`, `mask[m, k]` (F16, `-inf` in excluded positions)
/// or `DevicePtr::NULL` for no mask. Output `out[m, k]` F16.
/// Launch: one block per row, 256 threads/row (two-pass online softmax).
pub fn softmax_masked_f16<B: flambeau_core::KernelBackend>(
    reg: &crate::Registry<B>,
    stream: &<B::Device as flambeau_core::Device>::Stream,
    bufs: crate::SoftmaxMaskedBuffers,
    knobs: crate::SoftmaxMaskedKnobs,
) -> Result<()> {
    let crate::SoftmaxMaskedBuffers { scores, mask, out } = bufs;
    let crate::SoftmaxMaskedKnobs { m, k, scale } = knobs;
    let module = reg.expect_module("softmax_masked_f16")?;
    let kernel = B::kernel(module, "flambeau_softmax_masked_f16")?;

    let m_i = m as i32;
    let k_i = k as i32;
    let scale_f = scale;
    let s_ptr: u64 = scores.as_usize() as u64;
    let m_ptr: u64 = mask.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let mut args = <B::Args<'_>>::default();
    args.push_arg(&s_ptr);
    args.push_arg(&m_ptr);
    args.push_arg(&o_ptr);
    args.push_arg(&m_i);
    args.push_arg(&k_i);
    args.push_arg(&scale_f);
    let cfg = LaunchCfg::one_d(m as u32, 256);
    B::launch(&kernel, stream, cfg, args)?;
    Ok(())
}
