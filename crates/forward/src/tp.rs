//! Tensor-parallel ctx. Per-rank sharded AttnWeights/FfnWeights
//! (caller's loader handles the sharding); composites delegate to
//! `core/` unchanged. The AR happens through `ar_sum_f32` after the
//! row-parallel matmuls.

use anyhow::Result;
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_model_ops::{Tensor, F16};
use flambeau_ops::OpsRegistry;

use crate::core::{composites, CoreState, ScratchPool, TopologyHooks};
use crate::ctx::{
    AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, ModelLayout,
    MoeWeights,
};

/// `ar_callback` is `(rank, n_ranks, buf, n_elems, device, stream)`.
/// Pluggable so production uses BAR1 P2P AllReduce while tests use a
/// host-roundtrip Barrier.
pub struct TpHooks {
    pub rank: usize,
    pub n_ranks: usize,
    pub ar_callback: Box<
        dyn FnMut(
                usize,
                usize,
                flambeau_core::DevicePtr,
                usize,
                &HipDevice,
                &HipStream,
            ) -> Result<()>
            + Send,
    >,
}

impl TopologyHooks for TpHooks {
    fn ar_sum_f32(
        &mut self,
        buf: flambeau_core::DevicePtr,
        n_elems: usize,
        device: &HipDevice,
        stream: &HipStream,
    ) -> Result<()> {
        if self.n_ranks <= 1 {
            return Ok(());
        }
        (self.ar_callback)(self.rank, self.n_ranks, buf, n_elems, device, stream)
    }
}

pub struct TpForwardCtx<'a> {
    core: CoreState<'a>,
    hooks: TpHooks,
}

impl<'a> TpForwardCtx<'a> {
    pub fn new(
        device: &'a HipDevice,
        stream: &'a HipStream,
        reg: &'a OpsRegistry,
        pool: &'a mut ScratchPool,
        hooks: TpHooks,
    ) -> Self {
        Self {
            core: CoreState::new(device, stream, reg, pool),
            hooks,
        }
    }
}

impl ForwardCtx for TpForwardCtx<'_> {
    fn embed(&mut self, weights: &EmbeddingWeights, token_id: u32) -> Result<Tensor<F16>> {
        composites::embed_local(&mut self.core, &mut self.hooks, weights, token_id)
    }

    fn rmsnorm(
        &mut self,
        input: &Tensor<F16>,
        weight: &Tensor<F16>,
        eps: f32,
    ) -> Result<Tensor<F16>> {
        composites::rmsnorm_local(&mut self.core, &mut self.hooks, input, weight, eps)
    }

    fn residual_add(&mut self, a: Tensor<F16>, b: Tensor<F16>) -> Result<Tensor<F16>> {
        composites::residual_add_local(&mut self.core, &mut self.hooks, a, b)
    }

    fn standard_attn(
        &mut self,
        input: &Tensor<F16>,
        weights: &AttnWeights,
        layer_idx: usize,
        position: usize,
    ) -> Result<Tensor<F16>> {
        composites::standard_attn_local(
            &mut self.core,
            &mut self.hooks,
            input,
            weights,
            layer_idx,
            position,
        )
    }

    fn dense_ffn(&mut self, input: &Tensor<F16>, weights: &FfnWeights) -> Result<Tensor<F16>> {
        composites::dense_ffn_local(&mut self.core, &mut self.hooks, input, weights)
    }

    fn moe_ffn(&mut self, input: &Tensor<F16>, weights: &MoeWeights) -> Result<Tensor<F16>> {
        composites::moe_ffn_local(&mut self.core, &mut self.hooks, input, weights)
    }

    fn output_head(&mut self, input: &Tensor<F16>, lm_head: &LmHeadWeights) -> Result<()> {
        composites::output_head_local(&mut self.core, &mut self.hooks, input, lm_head)
    }

    fn layer_range<'b>(
        &'b mut self,
        layout: &'b ModelLayout,
    ) -> Box<dyn Iterator<Item = usize> + 'b> {
        Box::new(0..layout.num_layers)
    }

    fn logits(&self) -> &[f32] {
        &self.core.logits_host
    }
}
