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
    let last_layer_idx = layers.last().copied();
    for li in layers.iter().copied() {
        let ffn_w = model.ffn[li]
            .as_ref()
            .expect("ffn weights missing for owned layer (PP slice mismatch)");
        let next_norm_after_attn: Option<&_> = Some(&ffn_w.ffn_norm);
        let delta = match model.layer_kinds[li] {
            LayerKind::FullAttn => {
                let w = model.full_attn[li]
                    .as_ref()
                    .expect("layer_kinds says FullAttn but full_attn[li] is None");
                ctx.standard_attn(&resid, w, li, positions, slot_ids, next_norm_after_attn)?
            }
            LayerKind::Gdn => {
                let w = model.gdn[li]
                    .as_ref()
                    .expect("layer_kinds says Gdn but gdn[li] is None");
                ctx.gdn_layer(&resid, w, li, slot_ids, next_norm_after_attn)?
            }
        };
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }

        let next_norm_after_ffn: Option<&_> = if Some(li) == last_layer_idx {
            None
        } else {
            let nli = li + 1;
            match model.layer_kinds[nli] {
                LayerKind::FullAttn => model.full_attn[nli].as_ref().map(|w| &w.attn_norm),
                LayerKind::Gdn => model.gdn[nli].as_ref().map(|w| &w.attn_norm),
            }
        };
        let delta = ctx.moe_ffn(&resid, ffn_w, n, next_norm_after_ffn)?;
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }
    }
    ctx.output_head(&resid, &model.lm_head, slot_ids)?;
    Ok(())
}

/// Sarathi-Serve mixed-batch forward (Phase K3). See
/// `flambeau_qwen35_v2::forward_mixed` for the design notes — this is
/// the MoE sibling that routes FFN through `moe_ffn` at `n = K + N`.
pub fn forward_mixed<C: ForwardCtx>(
    model: &Qwen35MoeV2Model,
    ctx: &mut C,
    tokens: &[u32],
    positions: &[usize],
    slot_ids: &[usize],
    prefill_rows: usize,
) -> Result<()> {
    let n = tokens.len();
    if prefill_rows == 0 || prefill_rows >= n {
        anyhow::bail!(
            "forward_mixed: prefill_rows must satisfy 0 < K < n (got K={prefill_rows}, n={n})"
        );
    }

    let mut resid = ctx.embed(&model.embedding, tokens)?;
    let layers: Vec<usize> = ctx.layer_range(&model.layout).collect();
    let last_layer_idx = layers.last().copied();
    for li in layers.iter().copied() {
        let ffn_w = model.ffn[li]
            .as_ref()
            .expect("ffn weights missing for owned layer (PP slice mismatch)");
        let next_norm_after_attn: Option<&_> = Some(&ffn_w.ffn_norm);
        let delta = match model.layer_kinds[li] {
            LayerKind::FullAttn => {
                let w = model.full_attn[li]
                    .as_ref()
                    .expect("layer_kinds says FullAttn but full_attn[li] is None");
                ctx.standard_attn_mixed(
                    &resid,
                    w,
                    li,
                    flambeau_forward::core::MixedBatch {
                        positions,
                        slot_ids,
                        prefill_rows,
                    },
                    next_norm_after_attn,
                )?
            }
            LayerKind::Gdn => {
                let w = model.gdn[li]
                    .as_ref()
                    .expect("layer_kinds says Gdn but gdn[li] is None");
                ctx.gdn_layer_mixed(
                    &resid,
                    w,
                    li,
                    flambeau_forward::core::GdnMixedBatch { slot_ids, prefill_rows },
                    next_norm_after_attn,
                )?
            }
        };
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }

        let next_norm_after_ffn: Option<&_> = if Some(li) == last_layer_idx {
            None
        } else {
            let nli = li + 1;
            match model.layer_kinds[nli] {
                LayerKind::FullAttn => model.full_attn[nli].as_ref().map(|w| &w.attn_norm),
                LayerKind::Gdn => model.gdn[nli].as_ref().map(|w| &w.attn_norm),
            }
        };
        let delta = ctx.moe_ffn(&resid, ffn_w, n, next_norm_after_ffn)?;
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }
    }

    let hidden = model.layout.hidden;
    let n_emit = n - prefill_rows + 1;
    let emit_ptr = resid.ptr.offset_bytes((prefill_rows - 1) * hidden * 2);
    let emit_input = unsafe {
        flambeau_model_ops::Tensor::<flambeau_model_ops::F16>::from_raw(emit_ptr, n_emit * hidden)
    };
    let mut emit_slots: Vec<usize> = Vec::with_capacity(n_emit);
    emit_slots.push(slot_ids[0]);
    emit_slots.extend_from_slice(&slot_ids[prefill_rows..]);
    ctx.output_head(&emit_input, &model.lm_head, &emit_slots)?;
    Ok(())
}
