//! MTP (multi-token prediction) draft chain for qwen35 nextn heads.
//!
//! Mirrors llama.cpp's `graph_mtp`: each draft step feeds the nextn layer
//! `eh_proj(concat(enorm(e_tok), hnorm(h)))`, pairing the token at position
//! `p + 1` with the hidden of position `p`: the target's post-`output_norm`
//! hidden for the first step, the draft layer's own post-`shared_head_norm`
//! hidden for chained steps. The draft layer is a
//! full-attention layer driven through its own KV slab
//! (`kv_caches[mtp_layer_idx]`).

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16, F32, Q8_1};
use flambeau_ops::Ops;

use crate::core::{CoreState, TokenBatch, TopologyHooks};
use crate::ctx::{EmbeddingWeights, LmHeadWeights, MtpDraft, MtpWeights};

use super::{dense_ffn_local, embed_local, residual_add_local, standard_attn_local};

fn slot_f16(ptr: DevicePtr, n: usize) -> Tensor<F16> {
    unsafe { Tensor::<F16>::from_raw(ptr, n) }
}
fn slot_f32(ptr: DevicePtr, n: usize) -> Tensor<F32> {
    unsafe { Tensor::<F32>::from_raw(ptr, n) }
}

/// Which hidden pairs with the first token of a catch-up batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpLead {
    /// The pool's anchor buffer (target hidden of the preceding position).
    Anchor,
    /// Zeros: the batch starts at position 0 or the anchor is stale.
    Zero,
    /// No lead row: every token pairs with a row of the staged residual.
    None,
}

/// Stage the target hiddens the draft layer's KV catch-up will consume into
/// `pool.mtp_rows_f16`: an optional lead row, then `output_norm` of the first
/// `n_rows` rows of `resid`. Must run before the anchor buffer or `resid` is
/// overwritten.
pub fn mtp_stage_hidden_local<B: flambeau_backend::Backend>(
    state: &mut CoreState<'_, B>,
    lm_head: &LmHeadWeights,
    resid: &Tensor<F16>,
    n_rows: usize,
    lead: MtpLead,
) -> Result<()> {
    let hidden = state.hidden();
    let row_bytes = hidden * 2;
    let rows = state.pool.mtp_rows_f16;
    if rows == DevicePtr::NULL {
        bail!("mtp_stage_hidden: pool has no MTP scratch (mtp disabled)");
    }
    let lead_src = match lead {
        MtpLead::Anchor => Some(state.pool.mtp_anchor_f16),
        MtpLead::Zero => Some(state.pool.mtp_zero_f16),
        MtpLead::None => None,
    };
    if let Some(src) = lead_src {
        // SAFETY: both buffers hold at least one `[hidden]` F16 row.
        unsafe {
            state
                .device
                .memcpy_async(state.stream, CopyDirection::DeviceToDevice, rows, src, row_bytes)
                .context("mtp lead hidden")?;
        }
    }
    if n_rows > 0 {
        let off = usize::from(lead_src.is_some());
        let mut dst = slot_f16(rows.offset_bytes(off * row_bytes), n_rows * hidden);
        let ops = state.ops();
        flambeau_model_ops::rmsnorm_f16(
            resid,
            &lm_head.output_norm,
            &mut dst,
            n_rows,
            hidden,
            lm_head.rms_eps,
            &ops,
        )
        .context("mtp stage output_norm")?;
    }
    Ok(())
}

/// The draft layer up to and including attention for `tokens.len()` rows:
/// `eh_proj(concat(enorm(embed(token)), hnorm(h)))`, then attention through
/// the layer's own KV slab at `positions` (which writes those KV rows).
/// Returns the post-attention residual.
#[allow(clippy::too_many_arguments)]
fn mtp_attn_rows<B: flambeau_backend::Backend, H: TopologyHooks<B>>(
    state: &mut CoreState<'_, B>,
    hooks: &mut H,
    w: &MtpWeights,
    embedding: &EmbeddingWeights,
    tokens: &[u32],
    h_in: DevicePtr,
    positions: &[usize],
    slot_id: usize,
    mtp_layer_idx: usize,
) -> Result<Tensor<F16>> {
    let hidden = state.hidden();
    let n = tokens.len();
    let ops = state.ops();
    let e = embed_local(state, hooks, embedding, tokens)?;
    let mut e_n = slot_f16(state.pool.mtp_e_f16, n * hidden);
    flambeau_model_ops::rmsnorm_f16(&e, &w.enorm, &mut e_n, n, hidden, w.rms_eps, &ops)
        .context("mtp enorm")?;
    let h_in = slot_f16(h_in, n * hidden);
    let mut h_n = slot_f16(state.pool.mtp_h_f16, n * hidden);
    flambeau_model_ops::rmsnorm_f16(&h_in, &w.hnorm, &mut h_n, n, hidden, w.rms_eps, &ops)
        .context("mtp hnorm")?;
    let pa = slot_f32(state.pool.mtp_proj_a_f32, n * hidden);
    ops.dense_gemv_f16_f16_batched(w.eh_proj_e.ptr, e_n.ptr, pa.ptr, hidden, hidden, n)
        .context("mtp eh_proj_e")?;
    let pb = slot_f32(state.pool.mtp_proj_b_f32, n * hidden);
    ops.dense_gemv_f16_f16_batched(w.eh_proj_h.ptr, h_n.ptr, pb.ptr, hidden, hidden, n)
        .context("mtp eh_proj_h")?;
    let mut psum = slot_f32(pa.ptr, n * hidden);
    flambeau_model_ops::add_f32(&pa, &pb, &mut psum, n * hidden, &ops).context("mtp proj add")?;
    let mut inp = slot_f16(state.pool.mtp_in_f16, n * hidden);
    flambeau_model_ops::cast_f32_to_f16(&psum, &mut inp, n * hidden, &ops)
        .context("mtp proj cast")?;

    let slot_ids = vec![slot_id; n];
    let delta = standard_attn_local(
        state,
        hooks,
        &inp,
        &w.attn,
        mtp_layer_idx,
        TokenBatch {
            positions,
            slot_ids: &slot_ids,
        },
        None,
    )?;
    let mut resid = inp;
    if let Some(d) = delta {
        resid = residual_add_local(state, hooks, resid, d, n)?;
    }
    Ok(resid)
}

/// Rewrite the draft layer's KV rows at `positions` from `(token, target
/// hidden)` pairs, the hiddens being the rows staged by
/// [`mtp_stage_hidden_local`]. Draft steps write those rows from the draft
/// layer's own hidden; the prompt never reaches the draft layer otherwise.
pub fn mtp_catchup_local<B: flambeau_backend::Backend, H: TopologyHooks<B>>(
    state: &mut CoreState<'_, B>,
    hooks: &mut H,
    w: &MtpWeights,
    embedding: &EmbeddingWeights,
    tokens: &[u32],
    positions: &[usize],
    slot_id: usize,
    mtp_layer_idx: usize,
) -> Result<()> {
    if tokens.is_empty() {
        return Ok(());
    }
    let h_in = state.pool.mtp_rows_f16;
    mtp_attn_rows(state, hooks, w, embedding, tokens, h_in, positions, slot_id, mtp_layer_idx)?;
    Ok(())
}

/// Run up to `n_draft` draft steps. Step 0 pairs `anchor_token` (the newest
/// sampled token, sitting at `start_pos`) with `anchor_h`, the target hidden
/// of `start_pos - 1`, and drafts the token at `start_pos + 1`; later steps
/// pair the previous draft with the draft layer's own hidden. Drafted tokens
/// are appended to `drafts` (host).
pub fn mtp_draft_local<B: flambeau_backend::Backend, H: TopologyHooks<B>>(
    state: &mut CoreState<'_, B>,
    hooks: &mut H,
    w: &MtpWeights,
    lm_head: &LmHeadWeights,
    embedding: &EmbeddingWeights,
    anchor_token: u32,
    anchor_h: &Tensor<F16>,
    start_pos: usize,
    slot_id: usize,
    mtp_layer_idx: usize,
    n_draft: usize,
    sample: &mut dyn FnMut(&[f32]) -> (u32, Vec<(u32, f32)>),
    drafts: &mut Vec<MtpDraft>,
) -> Result<()> {
    let hidden = state.hidden();
    let vocab = lm_head.vocab_size;
    let ops = state.ops();
    if hidden != lm_head.hidden {
        bail!("mtp_draft: ctx hidden {hidden} != lm_head.hidden {}", lm_head.hidden);
    }
    if state.pool.mtp_anchor_f16 == DevicePtr::NULL {
        bail!("mtp_draft: pool has no MTP scratch (mtp disabled)");
    }
    let mut h_cur = anchor_h.ptr;
    let mut tok = anchor_token;
    for step in 0..n_draft {
        let pos = start_pos + step;
        let mut resid =
            mtp_attn_rows(state, hooks, w, embedding, &[tok], h_cur, &[pos], slot_id, mtp_layer_idx)?;
        let delta = dense_ffn_local(state, hooks, &resid, &w.ffn, 1, None)?;
        if let Some(d) = delta {
            resid = residual_add_local(state, hooks, resid, d, 1)?;
        }
        // shared_head_norm → chain hidden (F16) + quantized for the head
        let mut hout = slot_f16(state.pool.mtp_hidden_f16, hidden);
        flambeau_model_ops::rmsnorm_f16(&resid, &w.shared_head_norm, &mut hout, 1, hidden, lm_head.rms_eps, &ops)
            .context("mtp shared_head_norm")?;
        let mut q = unsafe { Tensor::<Q8_1>::from_raw(state.pool.norm_q8_1, hidden) };
        flambeau_model_ops::quantize_f16_to_q8_1(&hout, &mut q, hidden, &ops)
            .context("mtp head quant")?;
        let mut logits_dev = unsafe { Tensor::<F32>::from_raw(state.pool.logits_f32_dev, vocab) };
        let act_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };
        lm_head.lm_head.qmatmul(
            &q,
            &act_null,
            &mut logits_dev,
            flambeau_ops::MatmulShape {
                m: 1,
                k: hidden,
                n: vocab,
            },
            &ops,
        )
        .context("mtp lm_head")?;
        // The persistent logits buffer is the copy target: a per-step Vec
        // would hand the runtime a fresh pageable range to pin every draft.
        if state.logits_host.len() < vocab {
            state.logits_host = vec![0.0_f32; vocab];
        }
        unsafe {
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::DeviceToHost,
                    DevicePtr(state.logits_host.as_mut_ptr() as usize),
                    state.pool.logits_f32_dev,
                    vocab * 4,
                )
                .context("mtp logits DtoH")?;
        }
        flambeau_core::Stream::synchronize(state.stream)?;
        let (d, dist) = sample(&state.logits_host[..vocab]);
        let q = dist.iter().find(|(t, _)| *t == d).map(|(_, p)| *p).unwrap_or(0.0);
        drafts.push(MtpDraft { token: d, q, dist });
        tok = d;
        h_cur = state.pool.mtp_hidden_f16;
    }
    Ok(())
}
