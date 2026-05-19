//! Gemma-4 forward. Every layer is FullAttn. FFN is the dense variant
//! (`model.ffn[li]`, 31B / 9B); the MoE 26B-A4B variant loads weights
//! but the forward path is not yet wired — gemma4 MoE requires a
//! 5-norm F32 cascade (legacy `crates/models/gemma4/src/moe.rs`)
//! that the shared `moe_ffn` composite does not yet implement.
//! Surface a clear error instead of silently routing through the
//! qwen-shape `moe_ffn` and emitting garbage tokens.

use anyhow::{bail, Result};
use flambeau_forward::ctx::ForwardCtx;

use crate::loader::Gemma4V2Model;

pub fn forward<C: ForwardCtx>(
    model: &Gemma4V2Model,
    ctx: &mut C,
    tokens: &[u32],
    positions: &[usize],
    slot_ids: &[usize],
) -> Result<()> {
    if model.config.moe.is_some() {
        bail!(
            "gemma4-v2 MoE (26B-A4B) forward path not yet wired — gemma4 MoE \
             requires a 5-norm F32 cascade (pre_router_weight + ffn_norm + \
             pre_ffw_norm_2 + post_ffw_norm_1 + post_ffw_norm_2 + post_ffw_norm) \
             which the qwen-shape moe_ffn composite does not implement. See \
             crates/models/gemma4/src/moe.rs forward_ffn_moe for the math."
        );
    }
    let n = tokens.len();
    let mut resid = ctx.embed(&model.embedding, tokens)?;
    let layers: Vec<usize> = ctx.layer_range(&model.layout).collect();
    for li in layers {
        let attn_w = model.attn[li]
            .as_ref()
            .expect("attn weights missing for owned layer (PP slice mismatch)");
        let ffn_w = model.ffn[li]
            .as_ref()
            .expect("ffn weights missing for owned dense layer (PP slice mismatch)");

        let delta = ctx.standard_attn(&resid, attn_w, li, positions, slot_ids, None)?;
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }

        let delta = ctx.dense_ffn(&resid, ffn_w, n, None)?;
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }

        if let Some(scale) = model.layer_output_scale[li] {
            resid = ctx.scale_inplace_f16(resid, scale, n)?;
        }
    }
    ctx.output_head(&resid, &model.lm_head, slot_ids)?;
    Ok(())
}
