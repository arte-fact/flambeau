//! Gemma-4 forward. Every layer is FullAttn (dense variant). Per-layer
//! SWA/window comes from `AttnWeights.window_size`, picked up by
//! standard_attn's kernel arg.

use anyhow::Result;
use flambeau_forward::ctx::ForwardCtx;

use crate::loader::Gemma4V2Model;

pub fn forward<C: ForwardCtx>(
    model: &Gemma4V2Model,
    ctx: &mut C,
    tokens: &[u32],
    start_position: usize,
) -> Result<()> {
    let n = tokens.len();
    let mut resid = ctx.embed(&model.embedding, tokens)?;
    let layers: Vec<usize> = ctx.layer_range(&model.layout).collect();
    for li in layers {
        let attn_w = model.attn[li]
            .as_ref()
            .expect("attn weights missing for owned layer (PP slice mismatch)");
        let ffn_w = model.ffn[li]
            .as_ref()
            .expect("ffn weights missing for owned layer (PP slice mismatch)");

        let delta = ctx.standard_attn(&resid, attn_w, li, start_position, n)?;
        resid = ctx.residual_add(resid, delta, n)?;

        let delta = ctx.dense_ffn(&resid, ffn_w, n)?;
        resid = ctx.residual_add(resid, delta, n)?;
    }
    ctx.output_head(&resid, &model.lm_head, n)?;
    Ok(())
}
