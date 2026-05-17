//! Elementwise F16 add into the next residual slot (slot pings/pongs).

use anyhow::Result;
use flambeau_model_ops::{Tensor, F16};

use crate::core::{CoreState, TopologyHooks};

use super::slot_f16;

pub fn residual_add_local<H: TopologyHooks>(
    state: &mut CoreState<'_>,
    _hooks: &mut H,
    a: Tensor<F16>,
    b: Tensor<F16>,
) -> Result<Tensor<F16>> {
    let hidden = state.hidden();
    let out_ptr = state.pool.next_residual_slot();
    let mut out = slot_f16(out_ptr, hidden);
    let ops = state.ops();
    flambeau_model_ops::add_f16(&a, &b, &mut out, hidden, &ops)?;
    Ok(out)
}
