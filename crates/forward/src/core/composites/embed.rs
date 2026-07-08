use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device};
use flambeau_model_ops::{Tensor, F16};

use crate::core::{CoreState, TopologyHooks};
use crate::ctx::EmbeddingWeights;

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
    if weights.token_embd.n_elems < max * hidden {
        bail!(
            "embed: token_embd has {} F16 elems, need >= {}",
            weights.token_embd.n_elems,
            max * hidden
        );
    }
    let row_bytes = hidden * 2;
    let dst = state.pool.next_residual_slot();
    let n_elems = tokens.len() * hidden;
    // Decode-graph capture (n==1): gather the embedding row with a
    // device-position kernel that reads `token_id` from `pool.decode_ctx_i32[1]`,
    // so the captured graph re-sources the row per token with no memcpy-node
    // rebind. Multi-token prefill can't be captured this way — mark the pass
    // uncapturable so the driver falls back. The non-capture path keeps the
    // plain DtoD memcpy (byte-identical).
    let capture_single = tokens.len() == 1 && state.capture.is_some();
    if tokens.len() != 1 {
        if let Some(cap) = state.capture.as_deref_mut() {
            cap.mark_uncapturable("embed: multi-token capture unsupported");
        }
    }
    if capture_single {
        use flambeau_ops::Ops;
        let ctx_ptr = state.pool.decode_ctx_i32;
        let embd = weights.token_embd.ptr;
        let ops = state.ops();
        ops.embed_gather_f16_dpos(embd, dst, hidden, ctx_ptr)
            .context("embed: device-position row gather")?;
    } else {
        for (i, &t) in tokens.iter().enumerate() {
            let src = weights
                .token_embd
                .ptr
                .offset_bytes((t as usize) * row_bytes);
            let row_dst = dst.offset_bytes(i * row_bytes);
            // SAFETY: src points at row_bytes of token_embd; dst residual
            // slot is sized for max_prefill_tokens * hidden F16.
            unsafe {
                state
                    .device
                    .memcpy_async(
                        state.stream,
                        CopyDirection::DeviceToDevice,
                        row_dst,
                        src,
                        row_bytes,
                    )
                    .context("embed: DtoD row memcpy")?;
            }
        }
    }
    if let Some(scale) = weights.post_scale {
        let mut t = slot_f16(dst, n_elems);
        let ops = state.ops();
        let t_in = unsafe { Tensor::<F16>::from_raw(dst, n_elems) };
        flambeau_model_ops::scale_f16(&t_in, &mut t, n_elems, scale, &ops)?;
    }
    Ok(slot_f16(dst, n_elems))
}
