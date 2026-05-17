//! `PpForwardCtx` — pipeline-parallel forward context.
//!
//! Each rank owns a contiguous range of layers. The composites run
//! local kernels on this rank's tensors; `layer_range` yields only
//! this rank's layers and triggers `peer_copy_via_host` at stage
//! boundaries.
//!
//! Per-request state owned here:
//! - rank id, n_ranks, layer-to-rank map
//! - this rank's KV cache refs (borrowed slice into the model's pool)
//! - scratch pool for intermediate F16 / F32 / Q8_1 tensors
//! - position counter
//! - host logits slot (last rank only)
//!
//! Concrete fields + methods are scaffolded as `todo!()` and land
//! incrementally per `CLAUDE.md`'s "add a composite" recipe.

use anyhow::Result;
use flambeau_model_ops::{Tensor, F16};

use crate::ctx::{
    AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, ModelLayout,
    MoeWeights,
};

/// Pipeline-parallel forward context. Borrows per-request state from
/// the model crate that constructed it.
pub struct PpForwardCtx<'a> {
    pub rank: usize,
    pub n_ranks: usize,
    // KV cache refs, scratch pool, position, host logits slot, etc.
    // land as composites are written.
    pub _todo_state: std::marker::PhantomData<&'a ()>,
}

impl ForwardCtx for PpForwardCtx<'_> {
    fn embed(&mut self, _token_embd: &EmbeddingWeights, _token_id: u32) -> Result<Tensor<F16>> {
        todo!("PP: embed lookup on rank 0; other ranks no-op + return placeholder")
    }

    fn rmsnorm(&mut self, _input: &Tensor<F16>, _weight: &Tensor<F16>) -> Result<Tensor<F16>> {
        todo!("PP: local rmsnorm call into flambeau-model-ops::rmsnorm_f16")
    }

    fn residual_add(&mut self, _a: Tensor<F16>, _b: Tensor<F16>) -> Result<Tensor<F16>> {
        todo!("PP: local elementwise F16 add")
    }

    fn standard_attn(
        &mut self,
        _input: &Tensor<F16>,
        _weights: &AttnWeights,
        _layer_idx: usize,
        _position: usize,
    ) -> Result<Tensor<F16>> {
        todo!("PP: full attention block local to this rank (no AR)")
    }

    fn dense_ffn(&mut self, _input: &Tensor<F16>, _weights: &FfnWeights) -> Result<Tensor<F16>> {
        todo!("PP: gated MLP local to this rank (no AR)")
    }

    fn moe_ffn(&mut self, _input: &Tensor<F16>, _weights: &MoeWeights) -> Result<Tensor<F16>> {
        todo!("PP: MoE composer local to this rank (no AR; router + top-k + experts + combine)")
    }

    fn output_head(
        &mut self,
        _input: &Tensor<F16>,
        _lm_head: &LmHeadWeights,
    ) -> Result<()> {
        todo!("PP: last rank only — rmsnorm + lm_head + softcap + DtoH download")
    }

    fn layer_range<'b>(&'b mut self, _layout: &'b ModelLayout) -> Box<dyn Iterator<Item = usize> + 'b> {
        todo!("PP: yield this rank's layers; trigger peer_copy_via_host on stage handoff")
    }

    fn logits(&self) -> &[f32] {
        todo!("PP: return last rank's host logits slot")
    }
}
