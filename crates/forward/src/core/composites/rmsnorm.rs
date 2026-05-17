use anyhow::Result;
use flambeau_model_ops::{Tensor, F16};

use crate::core::{CoreState, TopologyHooks};

use super::slot_f16;

pub fn rmsnorm_local<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    _hooks: &mut H,
    input: &Tensor<F16>,
    weight: &Tensor<F16>,
    eps: f32,
) -> Result<Tensor<F16>> {
    let hidden = state.hidden();
    let mut out = slot_f16(state.pool.norm, hidden);
    let ops = state.ops();
    flambeau_model_ops::rmsnorm_f16(input, weight, &mut out, 1, hidden, eps, &ops)?;
    Ok(out)
}
