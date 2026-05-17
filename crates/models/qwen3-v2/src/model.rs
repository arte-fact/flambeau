//! Qwen3 dense forward function.
//!
//! Generic over `ForwardCtx` — the same code runs under single-device,
//! PP, TP, or hybrid topology executors. Topology placement (AR after
//! row-parallel ops, peer-copy at PP stage boundaries) is the ctx's
//! responsibility; this file only sequences the composites.

use anyhow::Result;
use flambeau_forward::ctx::ForwardCtx;

use crate::loader::Qwen3V2Model;

/// Run one decode-step forward for `token_id` at sequence position
/// `position`. Result lives in `ctx.logits()`.
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
