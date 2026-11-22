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
use crate::ctx::{EmbeddingWeights, LmHeadWeights, MtpDraft, MtpWeights, QuantWeight};

use super::{dense_ffn_local, embed_ids, residual_add_local, standard_attn_local, TokenIds};

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
    // The lead row is chosen from host-side pending state at record time; a
    // replay would keep that choice for every later token.
    if let Some(cap) = state.capture.as_deref_mut() {
        cap.mark_uncapturable("mtp: staged hiddens depend on host pending state");
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

/// One `[hidden, hidden]` half of `eh_proj` over `n` F16 rows into F32,
/// quantising the rows into the activation layout the dispatched kernel reads.
fn eh_proj_half<B: flambeau_backend::Backend>(
    state: &mut CoreState<'_, B>,
    w: &QuantWeight,
    x: &Tensor<F16>,
    out: &mut Tensor<F32>,
    n: usize,
) -> Result<()> {
    let hidden = state.hidden();
    let ops = state.ops();
    let shape = flambeau_ops::MatmulShape {
        m: n,
        k: hidden,
        n: hidden,
    };
    let layout = w.act_layout(shape, &ops);
    let (mmq_ptr, few_ptr) = match layout {
        flambeau_ops::ActLayout::Q8_1 => (DevicePtr::NULL, DevicePtr::NULL),
        flambeau_ops::ActLayout::Mmq => (state.pool.norm_q8_1_mmq, DevicePtr::NULL),
        flambeau_ops::ActLayout::Few(_) => (DevicePtr::NULL, state.pool.norm_q8_1_few),
    };
    // SAFETY: pool buffers sized for `n * hidden` activations, or NULL.
    let pool = |ptr: DevicePtr| unsafe {
        Tensor::<Q8_1>::from_raw(ptr, if ptr.as_usize() == 0 { 0 } else { n * hidden })
    };
    let mut act = pool(state.pool.norm_q8_1);
    let mut act_mmq = pool(mmq_ptr);
    let mut act_few = pool(few_ptr);
    flambeau_model_ops::quantize_f16_for_layouts(
        x,
        layout.into(),
        &mut act,
        &mut act_mmq,
        &mut act_few,
        hidden,
        n,
        &ops,
    )?;
    w.qmatmul(&act, &act_mmq, &act_few, out, shape, &ops)
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
    tokens: TokenIds<'_>,
    h_in: DevicePtr,
    positions: &[usize],
    slot_id: usize,
    mtp_layer_idx: usize,
) -> Result<Tensor<F16>> {
    let hidden = state.hidden();
    let n = positions.len();
    let ops = state.ops();
    if tokens.len() != n {
        bail!("mtp_attn_rows: {} token ids for {n} positions", tokens.len());
    }
    let e = embed_ids(state, embedding, tokens)?;
    let mut e_n = slot_f16(state.pool.mtp_e_f16, n * hidden);
    flambeau_model_ops::rmsnorm_f16(&e, &w.enorm, &mut e_n, n, hidden, w.rms_eps, &ops)
        .context("mtp enorm")?;
    let h_in = slot_f16(h_in, n * hidden);
    let mut h_n = slot_f16(state.pool.mtp_h_f16, n * hidden);
    flambeau_model_ops::rmsnorm_f16(&h_in, &w.hnorm, &mut h_n, n, hidden, w.rms_eps, &ops)
        .context("mtp hnorm")?;
    if n > state.pool.config.max_prefill_tokens {
        bail!(
            "mtp_attn_rows: {n} rows > max_prefill_tokens {}",
            state.pool.config.max_prefill_tokens
        );
    }
    let mut pa = slot_f32(state.pool.mtp_proj_a_f32, n * hidden);
    eh_proj_half(state, &w.eh_proj_e, &e_n, &mut pa, n).context("mtp eh_proj_e")?;
    let mut pb = slot_f32(state.pool.mtp_proj_b_f32, n * hidden);
    eh_proj_half(state, &w.eh_proj_h, &h_n, &mut pb, n).context("mtp eh_proj_h")?;
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
    mtp_attn_rows(state, hooks, w, embedding, TokenIds::Host(tokens), h_in, positions, slot_id, mtp_layer_idx)?;
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
    sample: &mut flambeau_runtime::MtpSampler,
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
    // Greedy draws run on device: the adjustments go up once. When the
    // embedding can gather device ids, each step's argmax lands in its own
    // result slot and the next step embeds that slot directly, so the chain
    // runs without a host round trip and the drafted tokens come back with
    // one copy and one sync at the end; otherwise each step reads back its
    // 4-byte token.
    let plan = sample.argmax.clone().filter(|p| argmax_plan_fits(state, p, 1));
    let chained = plan.is_some()
        && embedding.gathers_device_ids()
        && n_draft <= crate::core::scratch::ARGMAX_MAX_ROWS;
    let argmax_bufs = state.pool.argmax_buffers();
    let device = match plan.as_deref() {
        Some(p) => Some(upload_argmax_plan(state, p)?),
        None => None,
    };
    let mut h_cur = anchor_h.ptr;
    let mut tok = anchor_token;
    let mut prev_draw: Option<DevicePtr> = None;
    for step in 0..n_draft {
        let pos = start_pos + step;
        let ids = match prev_draw {
            Some(ptr) => TokenIds::Device { ptr, n: 1 },
            None => TokenIds::Host(std::slice::from_ref(&tok)),
        };
        let mut resid = mtp_attn_rows(state, hooks, w, embedding, ids, h_cur, &[pos], slot_id, mtp_layer_idx)?;
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
            &Tensor::null(),
            &mut logits_dev,
            flambeau_ops::MatmulShape {
                m: 1,
                k: hidden,
                n: vocab,
            },
            &ops,
        )
        .context("mtp lm_head")?;
        h_cur = state.pool.mtp_hidden_f16;
        if let Some(up) = device.as_ref() {
            let out = argmax_bufs.out.offset_bytes(if chained { step * 4 } else { 0 });
            ops.argmax_f32(flambeau_ops::ArgmaxBuffers { out, ..argmax_bufs }, vocab, up.n_adds, up.penalties)
                .context("mtp draft argmax")?;
            if chained {
                prev_draw = Some(out);
                continue;
            }
            let mut d: i32 = 0;
            // SAFETY: `out` is one i32 of argmax_scratch; synced below.
            unsafe {
                state
                    .device
                    .memcpy_async(
                        state.stream,
                        CopyDirection::DeviceToHost,
                        DevicePtr(&mut d as *mut i32 as usize),
                        out,
                        4,
                    )
                    .context("mtp draft token DtoH")?;
            }
            flambeau_core::Stream::synchronize(state.stream)?;
            drafts.push(MtpDraft { token: d as u32, q: 1.0, dist: vec![(d as u32, 1.0)] });
            tok = d as u32;
            continue;
        }
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
        let (d, dist) = (sample.draw)(&state.logits_host[..vocab]);
        let q = dist.iter().find(|(t, _)| *t == d).map(|(_, p)| *p).unwrap_or(0.0);
        drafts.push(MtpDraft { token: d, q, dist });
        tok = d;
    }
    if chained {
        let mut drawn = vec![0i32; n_draft];
        // SAFETY: `drawn` holds n_draft i32, the result slots written above
        // (n_draft <= ARGMAX_MAX_ROWS); synced below, which also releases the
        // plan's host sources.
        unsafe {
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::DeviceToHost,
                    DevicePtr(drawn.as_mut_ptr() as usize),
                    argmax_bufs.out,
                    n_draft * 4,
                )
                .context("mtp draft tokens DtoH")?;
        }
        flambeau_core::Stream::synchronize(state.stream)?;
        drafts.extend(drawn.into_iter().map(|d| MtpDraft { token: d as u32, q: 1.0, dist: vec![(d as u32, 1.0)] }));
    }
    drop(device);
    Ok(())
}

/// Whether `plan` fits the on-device draw scratch for `n_rows` logits rows.
pub fn argmax_plan_fits<B: flambeau_backend::Backend>(
    state: &CoreState<'_, B>,
    plan: &flambeau_runtime::MtpArgmax,
    n_rows: usize,
) -> bool {
    plan.adds.len() <= crate::core::scratch::ARGMAX_MAX_ADDS
        && plan.penalty_counts.len() <= state.pool.argmax_max_pairs
        && n_rows <= crate::core::scratch::ARGMAX_MAX_ROWS
        && n_rows <= state.pool.logits_rows
}

/// `plan` uploaded to the argmax scratch: the `Ops::argmax_f32` arguments,
/// and the host sources, kept alive until the stream's next sync.
struct ArgmaxUpload {
    n_adds: usize,
    penalties: Option<(usize, flambeau_ops::PenaltyKnobs)>,
    _sources: (Vec<i32>, Vec<f32>, Vec<u32>),
}

/// Upload `plan`'s additions and penalty pairs to the argmax scratch. The
/// caller has checked [`argmax_plan_fits`] and keeps the result until a sync.
fn upload_argmax_plan<B: flambeau_backend::Backend>(
    state: &mut CoreState<'_, B>,
    plan: &flambeau_runtime::MtpArgmax,
) -> Result<ArgmaxUpload> {
    let bufs = state.pool.argmax_buffers();
    let add_ids: Vec<i32> = plan.adds.iter().map(|&(id, _)| id as i32).collect();
    let add_vals: Vec<f32> = plan.adds.iter().map(|&(_, v)| v).collect();
    let counts: Vec<u32> = plan.penalty_counts.iter().flat_map(|&(t, c)| [t, c]).collect();
    for (dst, src, bytes) in [
        (bufs.add_ids, add_ids.as_ptr() as usize, add_ids.len() * 4),
        (bufs.add_vals, add_vals.as_ptr() as usize, add_vals.len() * 4),
        (bufs.penalty_counts, counts.as_ptr() as usize, counts.len() * 4),
    ] {
        if bytes > 0 {
            // SAFETY: each source fits its argmax_scratch region
            // (argmax_plan_fits) and lives in the returned ArgmaxUpload until
            // the caller's next sync.
            unsafe {
                state.device.memcpy_async(state.stream, CopyDirection::HostToDevice, dst, DevicePtr(src), bytes)?;
            }
        }
    }
    let penalties = Some((
        plan.penalty_counts.len(),
        flambeau_ops::PenaltyKnobs { repetition: plan.repetition, presence: plan.presence, frequency: plan.frequency },
    ));
    Ok(ArgmaxUpload { n_adds: add_ids.len(), penalties, _sources: (add_ids, add_vals, counts) })
}

/// The greedy draw of `plan` on each of the first `n_rows` rows of the LM
/// head's device logits; reads back the `n_rows` tokens. The caller has
/// checked [`argmax_plan_fits`].
pub fn argmax_rows_local<B: flambeau_backend::Backend>(
    state: &mut CoreState<'_, B>,
    plan: &flambeau_runtime::MtpArgmax,
    n_rows: usize,
    vocab: usize,
) -> Result<Vec<u32>> {
    let upload = upload_argmax_plan(state, plan)?;
    let base = state.pool.argmax_buffers();
    let ops = state.ops();
    for row in 0..n_rows {
        let bufs = flambeau_ops::ArgmaxBuffers {
            logits: base.logits.offset_bytes(row * vocab * 4),
            out: base.out.offset_bytes(row * 4),
            ..base
        };
        ops.argmax_f32(bufs, vocab, upload.n_adds, upload.penalties).context("mtp verify argmax")?;
    }
    let mut tokens = vec![0i32; n_rows];
    // SAFETY: `tokens` holds n_rows i32, the result slots written above
    // (n_rows <= ARGMAX_MAX_ROWS); synced below.
    unsafe {
        state.device.memcpy_async(
            state.stream,
            CopyDirection::DeviceToHost,
            DevicePtr(tokens.as_mut_ptr() as usize),
            base.out,
            n_rows * 4,
        )?;
    }
    flambeau_core::Stream::synchronize(state.stream)?;
    drop(upload);
    Ok(tokens.into_iter().map(|t| t as u32).collect())
}
