//! Embedding lookup — DtoD memcpy of the F16 token_embd row into the
//! next residual slot.

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device};
use flambeau_model_ops::{Tensor, F16};

use crate::core::{CoreState, TopologyHooks};
use crate::ctx::EmbeddingWeights;

use super::slot_f16;

pub fn embed_local<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    _hooks: &mut H,
    weights: &EmbeddingWeights,
    token_id: u32,
) -> Result<Tensor<F16>> {
    let hidden = state.hidden();
    if hidden != weights.hidden {
        bail!(
            "embed: ctx hidden {hidden} != weights.hidden {}",
            weights.hidden
        );
    }
    if (token_id as usize) >= weights.vocab_size {
        bail!(
            "embed: token_id {token_id} >= vocab_size {}",
            weights.vocab_size
        );
    }
    if weights.token_embd.n_elems < weights.vocab_size * hidden {
        bail!(
            "embed: token_embd has {} F16 elems, need >= {}",
            weights.token_embd.n_elems,
            weights.vocab_size * hidden
        );
    }
    let row_bytes = hidden * 2;
    let src = weights
        .token_embd
        .ptr
        .offset_bytes((token_id as usize) * row_bytes);
    let dst = state.pool.next_residual_slot();
    // SAFETY: src points at >= row_bytes valid F16 weight bytes; dst is
    // a pool slot sized for hidden F16 elems; stream is live.
    unsafe {
        state
            .device
            .memcpy_async(state.stream, CopyDirection::DeviceToDevice, dst, src, row_bytes)
            .context("embed: DtoD row memcpy")?;
    }
    Ok(slot_f16(dst, hidden))
}
