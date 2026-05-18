//! Gemma-4 forward. Every layer is FullAttn (dense variant). Per-layer
//! SWA/window comes from `AttnWeights.window_size`, picked up by
//! standard_attn's kernel arg.

use anyhow::Result;
use flambeau_forward::ctx::ForwardCtx;

use crate::loader::Gemma4V2Model;

pub fn forward_one_token<C: ForwardCtx>(
    model: &Gemma4V2Model,
    ctx: &mut C,
    token_id: u32,
    position: usize,
) -> Result<()> {
    let mut resid = ctx.embed(&model.embedding, token_id)?;
    let layers: Vec<usize> = ctx.layer_range(&model.layout).collect();
    for li in layers {
        let attn_w = &model.attn[li];
        let ffn_w = &model.ffn[li];

        let delta = ctx.standard_attn(&resid, attn_w, li, position)?;
        resid = ctx.residual_add(resid, delta)?;

        let delta = ctx.dense_ffn(&resid, ffn_w)?;
        resid = ctx.residual_add(resid, delta)?;
    }
    ctx.output_head(&resid, &model.lm_head)?;
    Ok(())
}
