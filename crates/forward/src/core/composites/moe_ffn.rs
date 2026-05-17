//! MoE FFN composite. Bails with a TODO until P7 (qwen3.6-v2) lands
//! the router + indexed expert matmul + combine sequence.

use anyhow::{bail, Result};
use flambeau_model_ops::{Tensor, F16};

use crate::core::{CoreState, TopologyHooks};
use crate::ctx::MoeWeights;

pub fn moe_ffn_local<H: TopologyHooks>(
    _state: &mut CoreState<'_>,
    _hooks: &mut H,
    _input: &Tensor<F16>,
    _weights: &MoeWeights,
) -> Result<Tensor<F16>> {
    bail!("moe_ffn — not implemented; lands with qwen3.6-v2 in P7")
}
