//! PP-of-TP forward context. Stage handoff must drain BOTH the
//! sub-cluster and global-cluster default streams before peer_copy
//! reads from `hidden` — see project memory `hipcluster_stream_handles`.

use anyhow::Result;
use flambeau_model_ops::{Tensor, F16};

use crate::ctx::{
    AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, ModelLayout,
    MoeWeights,
};

pub struct HybridForwardCtx<'a> {
    pub stage_idx: usize,
    pub rank_in_stage: usize,
    pub tp_size: usize,
    pub n_stages: usize,
    pub _todo_state: std::marker::PhantomData<&'a ()>,
}

impl ForwardCtx for HybridForwardCtx<'_> {
    fn embed(&mut self, _token_embd: &EmbeddingWeights, _token_id: u32) -> Result<Tensor<F16>> {
        todo!()
    }

    fn rmsnorm(
        &mut self,
        _input: &Tensor<F16>,
        _weight: &Tensor<F16>,
        _eps: f32,
    ) -> Result<Tensor<F16>> {
        todo!()
    }

    fn residual_add(&mut self, _a: Tensor<F16>, _b: Tensor<F16>) -> Result<Tensor<F16>> {
        todo!()
    }

    fn standard_attn(
        &mut self,
        _input: &Tensor<F16>,
        _weights: &AttnWeights,
        _layer_idx: usize,
        _position: usize,
    ) -> Result<Tensor<F16>> {
        todo!()
    }

    fn dense_ffn(&mut self, _input: &Tensor<F16>, _weights: &FfnWeights) -> Result<Tensor<F16>> {
        todo!()
    }

    fn moe_ffn(&mut self, _input: &Tensor<F16>, _weights: &MoeWeights) -> Result<Tensor<F16>> {
        todo!()
    }

    fn output_head(
        &mut self,
        _input: &Tensor<F16>,
        _lm_head: &LmHeadWeights,
    ) -> Result<()> {
        todo!()
    }

    fn layer_range<'b>(
        &'b mut self,
        _layout: &'b ModelLayout,
    ) -> Box<dyn Iterator<Item = usize> + 'b> {
        todo!()
    }

    fn logits(&self) -> &[f32] {
        todo!()
    }
}
