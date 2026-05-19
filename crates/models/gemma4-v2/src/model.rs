//! Gemma-4 forward. Every layer is FullAttn. FFN dispatches dense vs
//! MoE based on `model.config.moe`. The MoE path uses the 5-norm F32
//! cascade in `flambeau-forward::core::composites::moe_ffn` (gemma4
//! branch), which is activated when `MoeWeights::pre_router_weight_f16`
//! is `Some`.

use anyhow::Result;
use flambeau_forward::ctx::ForwardCtx;

use crate::loader::Gemma4V2Model;

pub fn forward<C: ForwardCtx>(
    model: &Gemma4V2Model,
    ctx: &mut C,
    tokens: &[u32],
    positions: &[usize],
    slot_ids: &[usize],
) -> Result<()> {
    let n = tokens.len();
    let is_moe = model.config.moe.is_some();
    let mut resid = ctx.embed(&model.embedding, tokens)?;
    let layers: Vec<usize> = ctx.layer_range(&model.layout).collect();
    for li in layers {
        let attn_w = model.attn[li]
            .as_ref()
            .expect("attn weights missing for owned layer (PP slice mismatch)");

        let delta = ctx.standard_attn(&resid, attn_w, li, positions, slot_ids, None)?;
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }

        let delta = if is_moe {
            let moe_w = model.moe[li]
                .as_ref()
                .expect("moe weights missing for owned MoE layer (PP slice mismatch)");
            ctx.moe_ffn(&resid, moe_w, n, None)?
        } else {
            let ffn_w = model.ffn[li]
                .as_ref()
                .expect("ffn weights missing for owned dense layer (PP slice mismatch)");
            ctx.dense_ffn(&resid, ffn_w, n, None)?
        };
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
