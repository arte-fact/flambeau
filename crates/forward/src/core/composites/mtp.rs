//! MTP (multi-token prediction) draft chain for qwen35 nextn heads.
//!
//! Mirrors llama.cpp's `graph_mtp`: each draft step feeds the nextn layer
//! `eh_proj(concat(enorm(e_tok), hnorm(h)))` where `h` is the target's
//! post-`output_norm` hidden for the first step and the draft layer's own
//! post-`shared_head_norm` hidden for chained steps. The draft layer is a
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

/// Run up to `n_draft` greedy draft steps starting from the anchor
/// (`anchor_token` at `start_pos` with target hidden `anchor_h`, both in
/// slot `slot_id`). Drafted tokens are appended to `drafts` (host).
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
        // e = embed(tok); e_n = rmsnorm(e, enorm)
        let e = embed_local(state, hooks, embedding, &[tok])?;
        let mut e_n = slot_f16(state.pool.mtp_e_f16, hidden);
        flambeau_model_ops::rmsnorm_f16(&e, &w.enorm, &mut e_n, 1, hidden, w.rms_eps, &ops)
            .context("mtp enorm")?;
        // h_n = rmsnorm(h_cur, hnorm)
        let h_in = unsafe { Tensor::<F16>::from_raw(h_cur, hidden) };
        let mut h_n = slot_f16(state.pool.mtp_h_f16, hidden);
        flambeau_model_ops::rmsnorm_f16(&h_in, &w.hnorm, &mut h_n, 1, hidden, w.rms_eps, &ops)
            .context("mtp hnorm")?;
        // proj = eh_proj_e @ e_n + eh_proj_h @ h_n  (F32 partials, then add + cast)
        let pa = slot_f32(state.pool.mtp_proj_a_f32, hidden);
        ops.dense_gemv_f16_f16_batched(
            w.eh_proj_e.ptr,
            e_n.ptr,
            pa.ptr,
            hidden,
            hidden,
            1,
        )
        .context("mtp eh_proj_e")?;
        let pb = slot_f32(state.pool.mtp_proj_b_f32, hidden);
        ops.dense_gemv_f16_f16_batched(
            w.eh_proj_h.ptr,
            h_n.ptr,
            pb.ptr,
            hidden,
            hidden,
            1,
        )
        .context("mtp eh_proj_h")?;
        let mut psum = unsafe { Tensor::<F32>::from_raw(pa.ptr, hidden) };
        flambeau_model_ops::add_f32(&pa, &pb, &mut psum, hidden, &ops).context("mtp proj add")?;
        let mut inp = slot_f16(state.pool.mtp_in_f16, hidden);
        flambeau_model_ops::cast_f32_to_f16(&psum, &mut inp, hidden, &ops).context("mtp proj cast")?;

        // attn (own KV slab) + ffn, mirroring the main layer loop.
        let delta = standard_attn_local(
            state,
            hooks,
            &inp,
            &w.attn,
            mtp_layer_idx,
            TokenBatch {
                positions: &[pos],
                slot_ids: &[slot_id],
            },
            None,
        )?;
        let mut resid = inp;
        if let Some(d) = delta {
            resid = residual_add_local(state, hooks, resid, d, 1)?;
        }
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
        // DtoH + argmax on host
        let mut row = vec![0.0f32; vocab];
        unsafe {
            state
                .device
                .memcpy_async(
                    state.stream,
                    CopyDirection::DeviceToHost,
                    DevicePtr(row.as_mut_ptr() as usize),
                    state.pool.logits_f32_dev,
                    vocab * 4,
                )
                .context("mtp logits DtoH")?;
        }
        flambeau_core::Stream::synchronize(state.stream)?;
        let (d, dist) = sample(&row);
        let q = dist.iter().find(|(t, _)| *t == d).map(|(_, p)| *p).unwrap_or(0.0);
        drafts.push(MtpDraft { token: d, q, dist });
        tok = d;
        h_cur = state.pool.mtp_hidden_f16;
    }
    Ok(())
}
