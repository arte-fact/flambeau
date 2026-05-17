//! qwen3 forward, `<C: ForwardCtx>`-generic. Topology placement is
//! the ctx's job; this file only sequences composites.

use anyhow::Result;
use flambeau_forward::ctx::ForwardCtx;

use crate::loader::Qwen3V2Model;

/// One decode step. Logits land in `ctx.logits()`.
pub fn forward_one_token<C: ForwardCtx>(
    model: &Qwen3V2Model,
    ctx: &mut C,
    token_id: u32,
    position: usize,
) -> Result<()> {
    let mut resid = ctx.embed(&model.embedding, token_id)?;

    let layers: Vec<usize> = ctx.layer_range(&model.layout).collect();
    for li in layers {
        let attn_w = &model.attn[li];
        let ffn_w = &model.ffn[li];

        let normed = ctx.rmsnorm(&resid, &attn_w.attn_norm, attn_w.rms_eps)?;
        let delta = ctx.standard_attn(&normed, attn_w, li, position)?;
        resid = ctx.residual_add(resid, delta)?;

        let normed = ctx.rmsnorm(&resid, &ffn_w.ffn_norm, ffn_w.rms_eps)?;
        let delta = ctx.dense_ffn(&normed, ffn_w)?;
        resid = ctx.residual_add(resid, delta)?;
    }

    ctx.output_head(&resid, &model.lm_head)?;
    Ok(())
}
