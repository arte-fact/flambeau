//! Single-GPU forward context: one device, one stream, `NoopHooks`.

use anyhow::Result;
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_model_ops::{Tensor, F16};
use flambeau_ops::OpsRegistry;

use crate::core::{composites, CoreState, NoopHooks};
use crate::ctx::{
    AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, ModelLayout,
    MoeWeights,
};

pub use crate::core::{ScratchConfig, ScratchPool};

pub struct SingleDeviceForwardCtx<'a> {
    core: CoreState<'a>,
    hooks: NoopHooks,
}

impl<'a> SingleDeviceForwardCtx<'a> {
    pub fn new(
        device: &'a HipDevice,
        stream: &'a HipStream,
        reg: &'a OpsRegistry,
        pool: &'a mut ScratchPool,
    ) -> Self {
        Self {
            core: CoreState::new(device, stream, reg, pool),
            hooks: NoopHooks,
        }
    }

    /// Reset slot selection between forward passes. KV caches stay
    /// populated; caller's invariant is that `position` matches.
    pub fn reset(&mut self) {
        self.core.pool.current_residual_is_a = true;
    }
}

impl ForwardCtx for SingleDeviceForwardCtx<'_> {
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

    fn gdn_layer(
        &mut self,
        input: &Tensor<F16>,
        weights: &crate::ctx::GdnWeights,
        layer_idx: usize,
    ) -> Result<Tensor<F16>> {
        composites::gdn_layer_local(&mut self.core, &mut self.hooks, input, weights, layer_idx)
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

