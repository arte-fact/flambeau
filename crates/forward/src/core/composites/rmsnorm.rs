use anyhow::Result;
use flambeau_model_ops::{Tensor, F16};

use crate::core::{CoreState, TopologyHooks};

use super::slot_f16;

pub fn rmsnorm_local<B: flambeau_backend::Backend, H: TopologyHooks<B>>(
    state: &mut CoreState<'_, B>,
    _hooks: &mut H,
    input: &Tensor<F16>,
    weight: &Tensor<F16>,
    eps: f32,
    n_tokens: usize,
) -> Result<Tensor<F16>> {
    let hidden = state.hidden();
    let n_elems = n_tokens * hidden;
    let mut out = slot_f16(state.pool.norm, n_elems);
    let ops = state.ops();
    flambeau_model_ops::rmsnorm_f16(input, weight, &mut out, n_tokens, hidden, eps, &ops)?;
    Ok(out)
}
