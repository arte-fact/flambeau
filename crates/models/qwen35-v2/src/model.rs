//! qwen35 forward. Per-layer dispatch on `LayerKind`: GDN → `gdn_layer`,
//! FullAttn → `standard_attn`. Dense-FFN afterward. `tokens.len() == 1`
//! is decode; longer is prefill or batched-decode (decided by the
//! caller via `slot_ids` uniformity).

use anyhow::Result;
use flambeau_forward::ctx::{ForwardCtx, LayerKind};

use crate::loader::Qwen35V2Model;

pub fn forward<C: ForwardCtx>(
    model: &Qwen35V2Model,
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

        let delta = ctx.dense_ffn(&resid, ffn_w, n, None)?;
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }
    }
    ctx.output_head(&resid, &model.lm_head, slot_ids)?;
    Ok(())
}

/// Sarathi-Serve mixed-batch forward (Phase K3). Rows `[0..prefill_rows)`
/// are a prefill chunk for `slot_ids[0]` at contiguous positions;
/// rows `[prefill_rows..n)` are batched decodes across N distinct slots.
/// Each attention / GDN layer routes through the `_mixed` ctx methods
/// (one launch covering both phases) instead of two separate calls; FFN
/// + residual + embed run uniformly at `n = K + N`. The output head
///   emits N+1 rows: the (K-1)-th prefill row (slot_p's next-token logit)
///   followed by N decode rows (per decode slot).
///
/// See `doc/MIXED_BATCH_V2_PLAN.md` Phase K3.
pub fn forward_mixed<C: ForwardCtx>(
    model: &Qwen35V2Model,
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
    for li in layers {
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
                    positions,
                    slot_ids,
                    prefill_rows,
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
                    slot_ids,
                    prefill_rows,
                    next_norm_after_attn,
                )?
            }
        };
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }

        let delta = ctx.dense_ffn(&resid, ffn_w, n, None)?;
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }
    }

    // Output head emits the (K-1)-th prefill row + N decode rows.
    // The residual buffer is contiguous over n=K+N rows; we hand the
    // ctx a sliced tensor view starting at row (K-1), of length N+1.
    let hidden = model.layout.hidden;
    let n_emit = n - prefill_rows + 1;
    let emit_ptr = resid.ptr.offset_bytes((prefill_rows - 1) * hidden * 2);
    let emit_input =
        unsafe { flambeau_model_ops::Tensor::<flambeau_model_ops::F16>::from_raw(emit_ptr, n_emit * hidden) };
    let mut emit_slots: Vec<usize> = Vec::with_capacity(n_emit);
    emit_slots.push(slot_ids[0]);
    emit_slots.extend_from_slice(&slot_ids[prefill_rows..]);
    ctx.output_head(&emit_input, &model.lm_head, &emit_slots)?;
    Ok(())
}
