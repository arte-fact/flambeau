//! `HybridForwardCtx` — PP-of-TP forward context.
//!
//! N PP stages, each running TP across `tp_size` ranks. Inside a
//! stage, composites act like `TpForwardCtx` (column/row sharding,
//! intra-stage AR). Between stages, `layer_range`'s drop / final-
//! yield triggers `peer_copy_via_host` from this stage's `hidden`
//! buffer to the next stage's `hidden` buffer on all `tp_size`
//! ranks (replicated handoff).
//!
//! Critical stream-sync invariant (see project-root memory
//! `hipcluster_stream_handles`): sub-cluster default streams and
//! global cluster default streams are distinct `HipStream` handles
//! for the same physical device; the handoff path must drain BOTH
//! sets before peer_copy reads from `hidden`. Otherwise pending
//! sub-cluster work races the global peer_copy.
//!
//! Per-request state owned here:
//! - this rank's (stage_idx, rank_in_stage) coords
//! - HybridCluster handle (sub-clusters + global cluster + per-stage AR)
//! - per-rank TP sync identity (producer-done event)
//! - this stage's KV cache refs (one slice per layer in stage)
//! - scratch pool sized for stage-local widths
//! - position counter
//! - host logits slot (head stage's head rank only)

use anyhow::Result;
use flambeau_model_ops::{Tensor, F16};

use crate::ctx::{
    AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, ModelLayout,
    MoeWeights,
};

/// Hybrid PP-of-TP forward context.
pub struct HybridForwardCtx<'a> {
    pub stage_idx: usize,
    pub rank_in_stage: usize,
    pub tp_size: usize,
    pub n_stages: usize,
    // HybridCluster, AR per stage, sync core, KV refs, scratch, ...
    pub _todo_state: std::marker::PhantomData<&'a ()>,
}

impl ForwardCtx for HybridForwardCtx<'_> {
    fn embed(&mut self, _token_embd: &EmbeddingWeights, _token_id: u32) -> Result<Tensor<F16>> {
        todo!("Hybrid: stage 0 only — embed + sqrt(hidden) scale on every TP rank")
    }

    fn rmsnorm(&mut self, _input: &Tensor<F16>, _weight: &Tensor<F16>) -> Result<Tensor<F16>> {
        todo!("Hybrid: rmsnorm replicated within stage")
    }

    fn residual_add(&mut self, _a: Tensor<F16>, _b: Tensor<F16>) -> Result<Tensor<F16>> {
        todo!("Hybrid: elementwise F16 add within stage")
    }

    fn standard_attn(
        &mut self,
        _input: &Tensor<F16>,
        _weights: &AttnWeights,
        _layer_idx: usize,
        _position: usize,
    ) -> Result<Tensor<F16>> {
        todo!(
            "Hybrid: same shape as TP composite but per-stage AR (sub-cluster's \
             BarP2pAllReduce, not the global one). Full-attn + Q8 + head_dim≥256 \
             still goes through F32 output path"
        )
    }

    fn dense_ffn(&mut self, _input: &Tensor<F16>, _weights: &FfnWeights) -> Result<Tensor<F16>> {
        todo!("Hybrid: TP-style gated MLP with per-stage AR")
    }

    fn moe_ffn(&mut self, _input: &Tensor<F16>, _weights: &MoeWeights) -> Result<Tensor<F16>> {
        todo!("Hybrid: TP-style MoE composer with per-stage AR (F32 partials)")
    }

    fn output_head(
        &mut self,
        _input: &Tensor<F16>,
        _lm_head: &LmHeadWeights,
    ) -> Result<()> {
        todo!("Hybrid: head stage's head rank only — rmsnorm + lm_head + softcap + DtoH")
    }

    fn layer_range<'b>(&'b mut self, _layout: &'b ModelLayout) -> Box<dyn Iterator<Item = usize> + 'b> {
        todo!(
            "Hybrid: yield this stage's layers; at end, peer_copy_via_host \
             from this stage's `hidden` to next stage's `hidden` after \
             draining both sub-cluster AND global-cluster streams \
             (hipcluster_stream_handles invariant)"
        )
    }

    fn logits(&self) -> &[f32] {
        todo!("Hybrid: return head stage / head rank's host logits slot")
    }
}
