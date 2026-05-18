//! qwen35 forward. Per-layer dispatch on `LayerKind`: GDN layers call
//! `gdn_layer`; full-attn layers call `standard_attn`. Both share the
//! same dense-FFN path afterward.

use anyhow::Result;
use flambeau_forward::ctx::{ForwardCtx, LayerKind};

use crate::loader::Qwen35V2Model;

pub fn forward_one_token<C: ForwardCtx>(
    model: &Qwen35V2Model,
    ctx: &mut C,
    token_id: u32,
    position: usize,
) -> Result<()> {
    let mut resid = ctx.embed(&model.embedding, token_id)?;
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
                ctx.standard_attn(&resid, w, li, position)?
            }
            LayerKind::Gdn => {
                let w = model.gdn[li]
                    .as_ref()
                    .expect("layer_kinds says Gdn but gdn[li] is None");
                ctx.gdn_layer(&resid, w, li)?
            }
        };
        resid = ctx.residual_add(resid, delta)?;

        let delta = ctx.dense_ffn(&resid, ffn_w)?;
        resid = ctx.residual_add(resid, delta)?;
    }
    ctx.output_head(&resid, &model.lm_head)?;
    Ok(())
}
