use anyhow::Result;
use flambeau_model_ops::{Tensor, F16};

use crate::core::{CoreState, TopologyHooks};

use super::slot_f16;

pub fn residual_add_local<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    _hooks: &mut H,
    a: Tensor<F16>,
    b: Tensor<F16>,
    n_tokens: usize,
) -> Result<Tensor<F16>> {
    // Fused-residual fast path: the previous composite (gemma4
    // post_attn_norm / post_ffn_norm) already advanced the residual
    // slot and wrote `resid + rmsnorm(delta)` into it via the fused
    // `rmsnorm_f32_to_f16_add_residual` kernel. `b` IS the new
    // residual; hand it back without re-advancing or re-adding.
    if state.pool.fused_residual_already_done {
        state.pool.fused_residual_already_done = false;
        let _ = a;
        return Ok(b);
    }
    let n_elems = n_tokens * state.hidden();
    let out_ptr = state.pool.next_residual_slot();
    let mut out = slot_f16(out_ptr, n_elems);
    let ops = state.ops();
    flambeau_model_ops::add_f16(&a, &b, &mut out, n_elems, &ops)?;
    Ok(out)
}
