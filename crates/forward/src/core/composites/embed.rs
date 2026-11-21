use anyhow::{bail, Context, Result};
use flambeau_core::{op::QDtype, CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16};

use crate::core::{CoreState, TopologyHooks};
use crate::ctx::{EmbeddingTable, EmbeddingWeights, QuantWeight};

use super::slot_f16;

pub fn embed_local<B: flambeau_backend::Backend, H: TopologyHooks<B>>(
    state: &mut CoreState<'_, B>,
    _hooks: &mut H,
    weights: &EmbeddingWeights,
    tokens: &[u32],
) -> Result<Tensor<F16>> {
    let hidden = state.hidden();
    if hidden != weights.hidden {
        bail!(
            "embed: ctx hidden {hidden} != weights.hidden {}",
            weights.hidden
        );
    }
    if tokens.is_empty() {
        bail!("embed: tokens empty");
    }
    let max = weights.vocab_size;
    for &t in tokens {
        if (t as usize) >= max {
            bail!("embed: token_id {t} >= vocab_size {max}");
        }
    }
    let dst = state.pool.next_residual_slot();
    let n_elems = tokens.len() * hidden;
    // Decode-graph capture (n == 1): the row is gathered by a kernel that reads
    // `token_id` from `pool.decode_ctx_i32[1]`, so the captured graph re-sources
    // it per token with no memcpy-node rebind. Multi-token passes can't be
    // captured this way — mark them uncapturable so the driver falls back.
    let capture_single = tokens.len() == 1 && state.capture.is_some();
    if tokens.len() != 1 {
        if let Some(cap) = state.capture.as_deref_mut() {
            cap.mark_uncapturable("embed: multi-token capture unsupported");
        }
    }
    match &weights.token_embd {
        EmbeddingTable::F16(table) => gather_f16(state, table, tokens, hidden, max, dst, capture_single)?,
        EmbeddingTable::Quant(table) => gather_quant(state, table, tokens, hidden, max, dst, capture_single)?,
    }
    if let Some(scale) = weights.post_scale {
        let mut t = slot_f16(dst, n_elems);
        let ops = state.ops();
        let t_in = unsafe { Tensor::<F16>::from_raw(dst, n_elems) };
        flambeau_model_ops::scale_f16(&t_in, &mut t, n_elems, scale, &ops)?;
    }
    Ok(slot_f16(dst, n_elems))
}

fn gather_f16<B: flambeau_backend::Backend>(
    state: &mut CoreState<'_, B>,
    table: &Tensor<F16>,
    tokens: &[u32],
    hidden: usize,
    vocab: usize,
    dst: DevicePtr,
    capture_single: bool,
) -> Result<()> {
    if table.n_elems < vocab * hidden {
        bail!("embed: token_embd has {} F16 elems, need >= {}", table.n_elems, vocab * hidden);
    }
    if capture_single {
        use flambeau_ops::Ops;
        let ctx_ptr = state.pool.decode_ctx_i32;
        state
            .ops()
            .embed_gather_f16_dpos(table.ptr, dst, hidden, ctx_ptr)
            .context("embed: device-position row gather")?;
        return Ok(());
    }
    let row_bytes = hidden * 2;
    for (i, &t) in tokens.iter().enumerate() {
        let src = table.ptr.offset_bytes((t as usize) * row_bytes);
        let row_dst = dst.offset_bytes(i * row_bytes);
        // SAFETY: src points at row_bytes of token_embd; dst residual
        // slot is sized for max_prefill_tokens * hidden F16.
        unsafe {
            state
                .device
                .memcpy_async(state.stream, CopyDirection::DeviceToDevice, row_dst, src, row_bytes)
                .context("embed: DtoD row memcpy")?;
        }
    }
    Ok(())
}

fn gather_quant<B: flambeau_backend::Backend>(
    state: &mut CoreState<'_, B>,
    table: &QuantWeight,
    tokens: &[u32],
    hidden: usize,
    vocab: usize,
    dst: DevicePtr,
    capture_single: bool,
) -> Result<()> {
    use flambeau_ops::Ops;
    if table.dtype != QDtype::Q4_K {
        bail!("embed: no device gather for a {:?} token_embd", table.dtype);
    }
    if table.n_elems < vocab * hidden {
        bail!("embed: token_embd has {} elems, need >= {}", table.n_elems, vocab * hidden);
    }
    let ids = if capture_single {
        state.pool.decode_ctx_i32.offset_bytes(4)
    } else {
        let n = tokens.len();
        if n > state.pool.embed_ids_host.len() {
            bail!("embed: {n} tokens > embed id staging {}", state.pool.embed_ids_host.len());
        }
        for (slot, &t) in state.pool.embed_ids_host.iter_mut().zip(tokens) {
            *slot = t as i32;
        }
        let src = DevicePtr(state.pool.embed_ids_host.as_ptr() as usize);
        // SAFETY: embed_ids_i32 holds max_prefill_tokens i32 (>= n, checked
        // above); src is the pool's stable host staging, live for the stream.
        unsafe {
            state
                .device
                .memcpy_async(state.stream, CopyDirection::HostToDevice, state.pool.embed_ids_i32, src, n * 4)
                .context("embed: token ids HtoD")?;
        }
        state.pool.embed_ids_i32
    };
    state
        .ops()
        .embed_gather_q4_k_f16(table.ptr, dst, ids, tokens.len(), hidden)
        .context("embed: Q4_K row gather")
}
