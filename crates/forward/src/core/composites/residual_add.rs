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
    let n_elems = n_tokens * state.hidden();
    let out_ptr = state.pool.next_residual_slot();
    let mut out = slot_f16(out_ptr, n_elems);
    let ops = state.ops();
    flambeau_model_ops::add_f16(&a, &b, &mut out, n_elems, &ops)?;
    Ok(out)
}
