//! qwen35moe forward. Per-layer dispatch on `LayerKind`. FFN is routed MoE.

use anyhow::Result;
use flambeau_forward::ctx::{ForwardCtx, LayerKind};

use crate::loader::Qwen35MoeV2Model;

pub fn forward<C: ForwardCtx>(
    model: &Qwen35MoeV2Model,
    ctx: &mut C,
    tokens: &[u32],
    positions: &[usize],
    slot_ids: &[usize],
) -> Result<()> {
    let n = tokens.len();
    let mut resid = ctx.embed(&model.embedding, tokens)?;
    let layers: Vec<usize> = ctx.layer_range(&model.layout).collect();
    for li in layers {
        let ffn_w = model.ffn[li]
            .as_ref()
            .expect("ffn weights missing for owned layer (PP slice mismatch)");
        let delta = match model.layer_kinds[li] {
            LayerKind::FullAttn => {
                let w = model.full_attn[li]
                    .as_ref()
                    .expect("layer_kinds says FullAttn but full_attn[li] is None");
                ctx.standard_attn(&resid, w, li, positions, slot_ids)?
            }
            LayerKind::Gdn => {
                let w = model.gdn[li]
                    .as_ref()
                    .expect("layer_kinds says Gdn but gdn[li] is None");
                ctx.gdn_layer(&resid, w, li, slot_ids)?
            }
        };
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }

        let delta = ctx.moe_ffn(&resid, ffn_w, n)?;
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }
    }
    ctx.output_head(&resid, &model.lm_head, slot_ids)?;
    Ok(())
}
