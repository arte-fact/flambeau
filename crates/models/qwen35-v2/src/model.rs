//! qwen35 forward. Per-layer dispatch on `LayerKind`: GDN → `gdn_layer`,
//! FullAttn → `standard_attn`. Dense-FFN afterward. `tokens.len() == 1`
//! is decode; longer is prefill or batched-decode (decided by the
//! caller via `slot_ids` uniformity).

use anyhow::Result;
use flambeau_forward::core::composites::MtpLead;
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
    if let Some(mtp) = model.mtp.as_ref() {
        let single_slot = slot_ids.iter().all(|&s| s == slot_ids[0]);
        if ctx.mtp_anchor_ptr() != flambeau_core::DevicePtr::NULL && single_slot {
            let slot = slot_ids[0];
            let lead = match positions[0].checked_sub(1) {
                Some(prev) if ctx.mtp_pending() == Some((slot, prev)) => MtpLead::Anchor,
                _ => MtpLead::Zero,
            };
            ctx.mtp_stage_hidden(&model.lm_head, &resid, n - 1, lead)?;
            capture_pending(model, ctx, &resid, n - 1, slot, positions[n - 1])?;
            ctx.mtp_catchup(mtp, &model.embedding, tokens, positions, slot, model.layout.num_layers)?;
        }
    }
    Ok(())
}

/// Write the post-`output_norm` hidden of `row` into the MTP anchor buffer
/// and record it as the pending hidden of `(slot, pos)`.
fn capture_pending<C: ForwardCtx>(
    model: &Qwen35V2Model,
    ctx: &mut C,
    resid: &flambeau_model_ops::Tensor<flambeau_model_ops::F16>,
    row: usize,
    slot: usize,
    pos: usize,
) -> Result<()> {
    let mut anchor = unsafe {
        flambeau_model_ops::Tensor::<flambeau_model_ops::F16>::from_raw(
            ctx.mtp_anchor_ptr(),
            model.layout.hidden,
        )
    };
    ctx.capture_hidden_from(
        resid,
        row,
        &model.lm_head.output_norm,
        model.lm_head.rms_eps,
        &mut anchor,
    )?;
    ctx.mtp_set_pending(Some((slot, pos)));
    Ok(())
}

/// All rows through the layers in one batched pass, no head. Returns the
/// pre-head residual.
fn verify_layers<C: ForwardCtx>(
    model: &Qwen35V2Model,
    ctx: &mut C,
    tokens: &[u32],
    positions: &[usize],
    slot_ids: &[usize],
) -> Result<flambeau_model_ops::Tensor<flambeau_model_ops::F16>> {
    let n = tokens.len();
    let mut resid = ctx.embed(&model.embedding, tokens)?;
    let layers: Vec<usize> = ctx.layer_range(&model.layout).collect();
    for li in layers {
        let ffn_w = model.ffn[li].as_ref().expect("ffn missing");
        let next_norm_after_attn: Option<&_> = Some(&ffn_w.ffn_norm);
        let delta = match model.layer_kinds[li] {
            LayerKind::FullAttn => {
                let w = model.full_attn[li].as_ref().expect("full_attn missing");
                ctx.standard_attn(&resid, w, li, positions, slot_ids, next_norm_after_attn)?
            }
            LayerKind::Gdn => {
                let w = model.gdn[li].as_ref().expect("gdn missing");
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
    Ok(resid)
}

/// One MTP cycle with rejection sampling (mirrors llama.cpp's
/// `draft-mtp` + `result_q` path).
///
/// `last_token` is the newest sampled token, not yet forwarded, sitting at
/// `pos`. The draft head pairs it with the target hidden of `pos - 1` (the
/// pending hidden every forward leaves behind) to draft `pos + 1..`; one
/// target forward over `[last_token, d1..dk]` then verifies all drafts.
/// Row `i` of that forward verifies draft `i + 1`, and the row after the
/// last accepted draft yields the bonus token.
///
/// `seed` drives the accept / residual draws: every TP rank runs this cycle
/// on identical logits and must reach identical verdicts.
///
/// Returns the tokens to emit: accepted drafts followed by the bonus
/// token (itself not yet forwarded). `out_logits` receives the target
/// logits row the bonus token was sampled from.
pub fn mtp_cycle<C: ForwardCtx>(
    model: &Qwen35V2Model,
    ctx: &mut C,
    last_token: u32,
    pos: usize,
    slot_id: usize,
    n_draft: usize,
    sample: &mut dyn FnMut(&[f32]) -> (u32, Vec<(u32, f32)>),
    seed: u64,
    out_logits: &mut Vec<f32>,
) -> Result<Vec<u32>> {
    let mut verdict_rng = flambeau_runtime::Rng::from_seed(seed ^ 0xA076_1D64_78BD_642F);
    let Some(mtp) = model.mtp.as_ref() else {
        anyhow::bail!("mtp_cycle: model has no MTP head loaded");
    };
    let vocab = model.lm_head.vocab_size;
    let mtp_layer_idx = model.layout.num_layers;
    let n_draft = n_draft.max(1);

    if pos == 0 || ctx.mtp_pending() != Some((slot_id, pos - 1)) {
        forward(model, ctx, &[last_token], &[pos], &[slot_id])?;
        let row = ctx.logits();
        anyhow::ensure!(row.len() >= vocab, "decode logits short");
        out_logits.clear();
        out_logits.extend_from_slice(&row[..vocab]);
        return Ok(vec![sample(out_logits).0]);
    }

    let mut drafts: Vec<flambeau_forward::ctx::MtpDraft> = Vec::with_capacity(n_draft);
    let pending_h = unsafe {
        flambeau_model_ops::Tensor::<flambeau_model_ops::F16>::from_raw(
            ctx.mtp_anchor_ptr(),
            model.layout.hidden,
        )
    };
    ctx.mtp_draft(
        mtp,
        &model.lm_head,
        &model.embedding,
        last_token,
        &pending_h,
        pos,
        slot_id,
        mtp_layer_idx,
        n_draft,
        sample,
        &mut drafts,
    )?;
    let k = drafts.len();

    let mut tokens: Vec<u32> = Vec::with_capacity(k + 1);
    tokens.push(last_token);
    tokens.extend(drafts.iter().map(|d| d.token));
    let positions: Vec<usize> = (pos..pos + k + 1).collect();
    let slots: Vec<usize> = vec![slot_id; k + 1];

    if k == 0 {
        forward(model, ctx, &tokens, &positions, &slots)?;
        let row = ctx.logits();
        out_logits.clear();
        out_logits.extend_from_slice(&row[..vocab]);
        return Ok(vec![sample(out_logits).0]);
    }
    ctx.mtp_set_pending(None);
    ctx.gdn_snapshot(slot_id)?;
    ctx.mtp_spec_rows(true);
    let verified = verify_layers(model, ctx, &tokens, &positions, &slots);
    ctx.mtp_spec_rows(false);
    let resid = verified?;

    // One LM-head pass over all k + 1 rows: the head is weight-bound, so a
    // batched pass reads its weights once instead of once per row.
    ctx.output_head_all(&resid, &model.lm_head, &slots, k + 1)?;
    let all_logits = ctx.logits();
    anyhow::ensure!(all_logits.len() >= (k + 1) * vocab, "verify logits short");
    let verify_logits = all_logits[..(k + 1) * vocab].to_vec();
    let row_at = |i: usize| &verify_logits[i * vocab..(i + 1) * vocab];

    let mut emitted: Vec<u32> = Vec::with_capacity(k + 1);
    let mut accepted = 0usize;
    let mut bonus: Option<u32> = None;
    let mut row = row_at(0);
    for d in &drafts {
        let (_pt, p_dist) = sample(row);
        let p_d = p_dist.iter().find(|(t, _)| *t == d.token).map(|(_, p)| *p).unwrap_or(0.0);
        // accept with prob min(1, p(d)/q(d)); greedy dists are one-hot, so
        // this reduces to argmax-match.
        if !(d.q > 0.0 && verdict_rng.next_f32() < (p_d / d.q).min(1.0)) {
            bonus = Some(sample_residual(&p_dist, &d.dist, &mut verdict_rng));
            break;
        }
        emitted.push(d.token);
        accepted += 1;
        row = row_at(accepted);
    }
    emitted.push(match bonus {
        Some(t) => t,
        None => sample(row).0,
    });

    if accepted < k {
        let layers: Vec<usize> = ctx.layer_range(&model.layout).collect();
        for li in layers {
            if let (LayerKind::Gdn, Some(w)) = (model.layer_kinds[li], model.gdn[li].as_ref()) {
                ctx.gdn_replay_rows(w, li, slot_id, accepted + 1)?;
            }
        }
    }
    ctx.mtp_stage_hidden(&model.lm_head, &resid, accepted, MtpLead::None)?;
    capture_pending(model, ctx, &resid, accepted, slot_id, pos + accepted)?;
    ctx.mtp_catchup(
        mtp,
        &model.embedding,
        &tokens[1..=accepted],
        &positions[1..=accepted],
        slot_id,
        mtp_layer_idx,
    )?;
    tracing::debug!(target: "mtp.cycle", drafted = k, accepted, "mtp cycle");

    *out_logits = row.to_vec();
    Ok(emitted)
}

/// Bonus-token distribution on rejection: normalize(max(p - q, 0)) over
/// the union of both supports, sampled with a uniform draw; falls back to
/// sampling from p when the residual is empty (p and q identical).
fn sample_residual(
    p_dist: &[(u32, f32)],
    q_dist: &[(u32, f32)],
    rng: &mut flambeau_runtime::Rng,
) -> u32 {
    let qmap: std::collections::HashMap<u32, f32> = q_dist.iter().cloned().collect();
    let mut resid: Vec<(u32, f32)> = Vec::new();
    for &(t, p) in p_dist {
        let q = qmap.get(&t).copied().unwrap_or(0.0);
        let r = (p - q).max(0.0);
        if r > 0.0 {
            resid.push((t, r));
        }
    }
    let total: f32 = resid.iter().map(|(_, r)| r).sum();
    if resid.is_empty() || total <= 0.0 {
        // fall back to p
        let mut acc = 0.0;
        let r = rng.next_f32();
        for &(t, p) in p_dist {
            acc += p;
            if r <= acc {
                return t;
            }
        }
        return p_dist.last().map(|(t, _)| *t).unwrap_or(0);
    }
    let mut acc = 0.0;
    let r = rng.next_f32() * total;
    for &(t, p) in &resid {
        acc += p;
        if r <= acc {
            return t;
        }
    }
    resid.last().map(|(t, _)| *t).unwrap_or(0)
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
