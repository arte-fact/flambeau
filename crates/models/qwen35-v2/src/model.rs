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

/// Forward like [`forward`] but captures the post-`output_norm` hidden of
/// `capture_row` into `ctx`'s MTP anchor buffer (via
/// [`ForwardCtx::capture_hidden_from`]).
pub fn forward_capture<C: ForwardCtx>(
    model: &Qwen35V2Model,
    ctx: &mut C,
    tokens: &[u32],
    positions: &[usize],
    slot_ids: &[usize],
    capture_row: usize,
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
    // capture the MTP anchor (post-output_norm hidden) before the head
    if model.mtp.is_some() {
        let mut anchor = unsafe {
            flambeau_model_ops::Tensor::<flambeau_model_ops::F16>::from_raw(
                ctx_anchor_ptr(ctx),
                model.layout.hidden,
            )
        };
        ctx.capture_hidden_from(
            &resid,
            capture_row,
            &model.lm_head.output_norm,
            model.lm_head.rms_eps,
            &mut anchor,
        )?;
    }
    ctx.output_head(&resid, &model.lm_head, slot_ids)?;
    Ok(())
}

/// Forward like [`forward`] (all rows through the layers in one batched
/// pass — that's where the MTP win lives), then the LM head is run PER ROW
/// (the head is weight-memory-bound, so batching rows buys nothing there and
/// would need a `k×vocab` device buffer). Returns per-row argmax tokens and
/// the full logits of every row on the host.
pub fn forward_verify<C: ForwardCtx>(
    model: &Qwen35V2Model,
    ctx: &mut C,
    tokens: &[u32],
    positions: &[usize],
    slot_ids: &[usize],
) -> Result<(Vec<u32>, Vec<f32>)> {
    let n = tokens.len();
    let hidden = model.layout.hidden;
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
    // Per-row head: emit one row at a time into the single-row logits
    // buffer, argmax on host, and keep the full row logits.
    let vocab = model.lm_head.vocab_size;
    let mut argmaxes = Vec::with_capacity(n);
    let mut all_logits = Vec::with_capacity(n * vocab);
    for i in 0..n {
        let row_view = unsafe {
            flambeau_model_ops::Tensor::<flambeau_model_ops::F16>::from_raw(
                resid.ptr.offset_bytes(i * hidden * 2),
                hidden,
            )
        };
        ctx.output_head(&row_view, &model.lm_head, &[slot_ids[i]])?;
        let row = ctx.logits();
        anyhow::ensure!(row.len() >= vocab, "verify head row short");
        let (mut bi, mut bv) = (0u32, f32::NEG_INFINITY);
        for (j, &l) in row.iter().take(vocab).enumerate() {
            if l > bv {
                bv = l;
                bi = j as u32;
            }
        }
        argmaxes.push(bi);
        all_logits.extend_from_slice(&row[..vocab]);
    }
    Ok((argmaxes, all_logits))
}

/// Raw device pointer of the ctx's MTP anchor buffer. The engine ctx
/// exposes it through a dedicated accessor; model crates use this helper.
fn ctx_anchor_ptr<C: ForwardCtx>(ctx: &mut C) -> flambeau_core::DevicePtr {
    ctx.mtp_anchor_ptr()
}

/// One MTP cycle with rejection sampling (mirrors llama.cpp's
/// `draft-mtp` + `result_q` path). `sample` draws a token from a logits
/// row and returns it with its full (truncated) distribution — for greedy
/// the distribution is one-hot, so the same code path reduces to the
/// validated argmax-match behavior.
///
/// Returns the tokens to emit: accepted drafts followed by the next
/// (bonus) token sampled from the target distribution. `out_logits`
/// receives the target logits row the bonus token was sampled from.
pub fn mtp_cycle<C: ForwardCtx>(
    model: &Qwen35V2Model,
    ctx: &mut C,
    anchor_token: u32,
    anchor_pos: usize,
    slot_id: usize,
    n_draft: usize,
    sample: &mut dyn FnMut(&[f32]) -> (u32, Vec<(u32, f32)>),
    out_logits: &mut Vec<f32>,
) -> Result<Vec<u32>> {
    let Some(mtp) = model.mtp.as_ref() else {
        anyhow::bail!("mtp_cycle: model has no MTP head loaded");
    };
    let vocab = model.lm_head.vocab_size;
    let mtp_layer_idx = model.layout.num_layers;
    let n_draft = n_draft.max(1);

    // 1. anchor forward: logits l_p + capture anchor hidden
    forward_capture(model, ctx, &[anchor_token], &[anchor_pos], &[slot_id], 0)?;
    let l_p: Vec<f32> = ctx.logits().to_vec();
    anyhow::ensure!(l_p.len() >= vocab, "anchor logits short");

    // 2. draft chain from the anchor (each draft carries its q dist)
    let mut drafts: Vec<flambeau_forward::ctx::MtpDraft> = Vec::with_capacity(n_draft);
    let anchor_h = unsafe {
        flambeau_model_ops::Tensor::<flambeau_model_ops::F16>::from_raw(
            ctx.mtp_anchor_ptr(),
            model.layout.hidden,
        )
    };
    ctx.mtp_draft(
        mtp,
        &model.lm_head,
        &model.embedding,
        anchor_token,
        &anchor_h,
        anchor_pos,
        slot_id,
        mtp_layer_idx,
        n_draft,
        sample,
        &mut drafts,
    )?;
    let k = drafts.len();
    if k == 0 {
        out_logits.clear();
        out_logits.extend_from_slice(&l_p);
        return Ok(Vec::new());
    }

    // 3. verify: one batched forward over the drafts; per-row head logits
    let snap = ctx.gdn_snapshot(slot_id)?;
    let d_tokens: Vec<u32> = drafts.iter().map(|d| d.token).collect();
    let positions: Vec<usize> = (anchor_pos + 1..anchor_pos + 1 + k).collect();
    let slots: Vec<usize> = vec![slot_id; k];
    let (_verify_amax, verify_logits) = forward_verify(model, ctx, &d_tokens, &positions, &slots)?;

    // target logits row that verifies draft i (1-based): l_p for i=1,
    // verify row i-2 for i>=2.
    let p_row = |i: usize| -> &[f32] {
        if i == 1 {
            &l_p
        } else {
            &verify_logits[(i - 2) * vocab..(i - 1) * vocab]
        }
    };

    // 4. rejection-sampling acceptance
    let mut emitted: Vec<u32> = Vec::with_capacity(k + 1);
    let mut accepted = 0usize;
    let mut next: Option<u32> = None;
    let mut next_row: &[f32] = &l_p;
    for i in 1..=k {
        let d = &drafts[i - 1];
        let row = p_row(i);
        let (_pt, p_dist) = sample(row);
        let p_d = p_dist.iter().find(|(t, _)| *t == d.token).map(|(_, p)| *p).unwrap_or(0.0);
        // accept with prob min(1, p(d)/q(d)); greedy => one-hot dists, so
        // this is exactly argmax-match.
        let accept = d.q > 0.0 && uniform01() < (p_d / d.q).min(1.0);
        next_row = row;
        if accept {
            emitted.push(d.token);
            accepted = i;
        } else {
            // bonus token from the residual normalize(max(p - q, 0))
            let bonus = sample_residual(&p_dist, &d.dist);
            next = Some(bonus);
            break;
        }
    }
    let next = match next {
        Some(t) => t,
        None => {
            // all drafts accepted: sample next from the last verify row
            let row = &verify_logits[(k - 1) * vocab..k * vocab];
            next_row = row;
            sample(row).0
        }
    };
    emitted.push(next);

    // 5. roll back GDN state if some drafts were rejected, then re-advance
    //    over the accepted prefix so recurrent state matches the tokens.
    if accepted < k {
        ctx.gdn_restore(slot_id, &snap)?;
        if accepted > 0 {
            let pos2: Vec<usize> = (anchor_pos + 1..anchor_pos + 1 + accepted).collect();
            let slots2 = vec![slot_id; accepted];
            forward(model, ctx, &d_tokens[..accepted], &pos2, &slots2)?;
        }
    }

    // 6. output logits = the target row the bonus/next token came from
    out_logits.clear();
    out_logits.extend_from_slice(next_row);
    Ok(emitted)
}

/// Thread-local uniform draw (rejection sampling). Deterministic enough for
/// spec-decode (each call advances a simple LCG seeded per call site).
fn uniform01() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0x9E3779B97F4A7C15) };
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 11) as f32 / (1u64 << 53) as f32
    })
}

/// Bonus-token distribution on rejection: normalize(max(p - q, 0)) over
/// the union of both supports, sampled with a uniform draw; falls back to
/// sampling from p when the residual is empty (p and q identical).
fn sample_residual(p_dist: &[(u32, f32)], q_dist: &[(u32, f32)]) -> u32 {
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
        let r = uniform01();
        for &(t, p) in p_dist {
            acc += p;
            if r <= acc {
                return t;
            }
        }
        return p_dist.last().map(|(t, _)| *t).unwrap_or(0);
    }
    let mut acc = 0.0;
    let r = uniform01() * total;
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
