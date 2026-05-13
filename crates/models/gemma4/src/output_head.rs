//! Output-head composition: `rmsnorm(output_norm) + LM-head matmul +
//! softcap`. Gemma 4 has a tied LM head — all 5 audited GGUFs lack
//! `output.weight`, so the LM head reuses `token_embd.weight`
//! (`{vocab, hidden}`).

#![cfg(feature = "hip")]

use anyhow::{Context, Result};
use flambeau_blocks::WeightHandle;
use flambeau_core::DevicePtr;
use flambeau_ops::Ops;

use crate::softcap::apply_logit_softcap;

/// Scratch buffers for the output-head step.
pub struct OutputHeadScratch {
    /// F16 [hidden] — `rmsnorm(output_norm)(x_final)`.
    pub x_norm_f16: DevicePtr,
    /// Q8_1 staging of `x_norm` for the LM-head MMVQ.
    pub x_q8_1: DevicePtr,
    /// F32 [vocab] — logits pre-softcap.
    pub logits_f32: DevicePtr,
}

/// Compute logits and optionally apply softcap. Mirrors llama.cpp's
/// `gemma4-iswa.cpp:240-258`:
/// ```text
/// cur = build_norm(cur, output_norm, LLM_NORM_RMS, -1);
/// cur = build_lora_mm(model.output, cur);
/// if (f_final_logit_softcapping) {
///     cur = scale(cur, 1 / cap);
///     cur = tanh(cur);
///     cur = scale(cur, cap);
/// }
/// ```
///
/// `lm_head` is the tied `token_embd` weight handle for gemma4 (all 5
/// audited GGUFs have no separate `output.weight`).
#[allow(clippy::too_many_arguments)]
pub fn forward_output_head<O: Ops>(
    ops: &O,
    x_in: DevicePtr,
    output_norm: DevicePtr,
    lm_head: WeightHandle,
    softcap: f32,
    scratch: &mut OutputHeadScratch,
    hidden: usize,
    vocab: usize,
    rms_norm_eps: f32,
) -> Result<DevicePtr> {
    ops.rmsnorm_f16(x_in, output_norm, scratch.x_norm_f16, 1, hidden, rms_norm_eps)
        .context("output_norm")?;
    ops.quantize_f16_q8_1(scratch.x_norm_f16, scratch.x_q8_1, hidden)
        .context("quantize output_norm → Q8_1")?;
    ops.mmvq(
        lm_head.ptr,
        scratch.x_q8_1,
        scratch.logits_f32,
        vocab,
        hidden,
        lm_head.dtype,
    )
    .context("lm_head mmvq")?;
    apply_logit_softcap(ops, scratch.logits_f32, scratch.logits_f32, vocab, softcap)?;
    Ok(scratch.logits_f32)
}
