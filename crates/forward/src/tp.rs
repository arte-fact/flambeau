//! `TpForwardCtx` — tensor-parallel forward context.
//!
//! Every rank runs every layer on its column / row shard. Composites
//! that involve row-parallel projections (`standard_attn`'s output
//! proj, `dense_ffn`'s down proj, `moe_ffn`'s combine) AR-sum
//! partials across ranks before returning.
//!
//! Per-request state owned here:
//! - rank id, n_ranks, BAR1 AllReduce handle
//! - per-rank TP sync identity (producer-done event)
//! - this rank's KV cache shard refs
//! - scratch pool (rank-local widths)
//! - position counter
//! - host logits slot (head rank only)
//!
//! Concrete fields land as composites are written.

use anyhow::Result;
use flambeau_model_ops::{Tensor, F16};

use crate::ctx::{
    AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, ModelLayout,
    MoeWeights,
};

/// Tensor-parallel forward context. Borrows per-request state from
/// the model crate that constructed it.
pub struct TpForwardCtx<'a> {
    pub rank: usize,
    pub n_ranks: usize,
    // BAR1 AR handle, per-rank TP sync core, KV refs, scratch, ...
    pub _todo_state: std::marker::PhantomData<&'a ()>,
}

impl ForwardCtx for TpForwardCtx<'_> {
    fn embed(&mut self, _token_embd: &EmbeddingWeights, _token_id: u32) -> Result<Tensor<F16>> {
        todo!("TP: embed lookup on every rank (replicated token_embd) + sqrt(hidden) scale")
    }

    fn rmsnorm(&mut self, _input: &Tensor<F16>, _weight: &Tensor<F16>) -> Result<Tensor<F16>> {
        todo!("TP: rmsnorm replicated across ranks (each rank computes the same)")
    }

    fn residual_add(&mut self, _a: Tensor<F16>, _b: Tensor<F16>) -> Result<Tensor<F16>> {
        todo!("TP: elementwise F16 add, replicated")
    }

    fn standard_attn(
        &mut self,
        _input: &Tensor<F16>,
        _weights: &AttnWeights,
        _layer_idx: usize,
        _position: usize,
    ) -> Result<Tensor<F16>> {
        todo!(
            "TP: column-parallel Q/K/V proj → RoPE → per-rank KV append → \
             flash-attn on local head shard → row-parallel output_proj → \
             tp_sum AR partials across ranks; \
             full-attn + head_dim≥256 + Q8 weights → F32 output path \
             (see project-root memory `gemma4_attn_output_proj_f16_saturate`)"
        )
    }

    fn dense_ffn(&mut self, _input: &Tensor<F16>, _weights: &FfnWeights) -> Result<Tensor<F16>> {
        todo!("TP: col-parallel gate/up → activate → row-parallel down → tp_sum AR")
    }

    fn moe_ffn(&mut self, _input: &Tensor<F16>, _weights: &MoeWeights) -> Result<Tensor<F16>> {
        todo!(
            "TP: parallel-branch composer (shared-MLP partial + routed-MoE partial) \
             with F32 AR on partials (`gemma4_moe_f16_overflow`)"
        )
    }

    fn output_head(
        &mut self,
        _input: &Tensor<F16>,
        _lm_head: &LmHeadWeights,
    ) -> Result<()> {
        todo!("TP: head rank only — rmsnorm + lm_head + softcap + DtoH")
    }

    fn layer_range<'b>(&'b mut self, _layout: &'b ModelLayout) -> Box<dyn Iterator<Item = usize> + 'b> {
        todo!("TP: yield 0..num_layers (every rank runs every layer)")
    }

    fn logits(&self) -> &[f32] {
        todo!("TP: return head rank's host logits slot")
    }
}
