//! qwen35moe forward. Per-layer dispatch on `LayerKind`: GDN layers
//! call `gdn_layer`; full-attn layers call `standard_attn`. FFN is
//! routed MoE everywhere.

use anyhow::Result;
use flambeau_forward::ctx::{ForwardCtx, LayerKind};

use crate::loader::Qwen35MoeV2Model;

pub fn forward<C: ForwardCtx>(
    model: &Qwen35MoeV2Model,
    ctx: &mut C,
    tokens: &[u32],
    start_position: usize,
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
                ctx.standard_attn(&resid, w, li, start_position, n)?
            }
            LayerKind::Gdn => {
                let w = model.gdn[li]
                    .as_ref()
                    .expect("layer_kinds says Gdn but gdn[li] is None");
                ctx.gdn_layer(&resid, w, li, n)?
            }
        };
        resid = ctx.residual_add(resid, delta, n)?;

        let delta = ctx.moe_ffn(&resid, ffn_w, n)?;
        resid = ctx.residual_add(resid, delta, n)?;
    }
    ctx.output_head(&resid, &model.lm_head, n)?;
    Ok(())
}
