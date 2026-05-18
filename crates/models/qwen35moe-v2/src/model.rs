//! qwen35moe forward. Per-layer dispatch on `LayerKind`: GDN layers
//! call `gdn_layer`; full-attn layers call `standard_attn`. FFN is
//! routed MoE everywhere.

use anyhow::Result;
use flambeau_forward::ctx::{ForwardCtx, LayerKind};

use crate::loader::Qwen35MoeV2Model;

pub fn forward_one_token<C: ForwardCtx>(
    model: &Qwen35MoeV2Model,
    ctx: &mut C,
    token_id: u32,
    position: usize,
) -> Result<()> {
    let mut resid = ctx.embed(&model.embedding, token_id)?;
    let layers: Vec<usize> = ctx.layer_range(&model.layout).collect();
    for li in layers {
        let ffn_w = &model.ffn[li];
        let delta = match model.layer_kinds[li] {
            LayerKind::FullAttn => {
                let w = model.full_attn[li]
                    .as_ref()
                    .expect("layer_kinds says FullAttn but full_attn[li] is None");
                let normed = ctx.rmsnorm(&resid, &w.attn_norm, w.rms_eps)?;
                ctx.standard_attn(&normed, w, li, position)?
            }
            LayerKind::Gdn => {
                let w = model.gdn[li]
                    .as_ref()
                    .expect("layer_kinds says Gdn but gdn[li] is None");
                let normed = ctx.rmsnorm(&resid, &w.attn_norm, w.rms_eps)?;
                ctx.gdn_layer(&normed, w, li)?
            }
        };
        resid = ctx.residual_add(resid, delta)?;

        let normed = ctx.rmsnorm(&resid, &ffn_w.ffn_norm, ffn_w.rms_eps)?;
        let delta = ctx.moe_ffn(&normed, ffn_w)?;
        resid = ctx.residual_add(resid, delta)?;
    }
    ctx.output_head(&resid, &model.lm_head)?;
    Ok(())
}
