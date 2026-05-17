//! `ForwardCtx` — the trait every topology implements.
//!
//! Method set on this trait IS the composite-op vocabulary that model
//! forward functions write against. The same model function calls
//! these methods regardless of topology; each impl knows how to do
//! its job under PP / TP / Hybrid (AllReduce after row-parallel
//! ops, peer-copy between PP stages, intra-stage TP composition
//! inside hybrid).
//!
//! Constraints (see `CLAUDE.md`):
//! - Every method on this trait is implemented by all three impls.
//!   No method exists on only one topology.
//! - Each method bottoms out in `flambeau-model-ops` leaf primitives.
//!   This trait does not call kernels directly.
//! - Trait carries no state; per-request state lives on the impl
//!   structs (`PpForwardCtx`, `TpForwardCtx`, `HybridForwardCtx`).
//!
//! The trait is intentionally minimal in this scaffold: only the
//! method signatures + doc comments. Method bodies on each impl land
//! as composites are written, in the order the first model needs
//! them. Adding a composite means:
//!
//! 1. Add the signature here.
//! 2. Implement it on `PpForwardCtx`, `TpForwardCtx`, `HybridForwardCtx`.
//! 3. Add the matching `OpCall` variant to `RecordingCtx`.
//! 4. Add a topology-parity test.

use anyhow::Result;
use flambeau_model_ops::{Tensor, F16};

/// Forward-pass context. One impl per topology; the model writes
/// `<C: ForwardCtx>` generic.
///
/// All composites take `&mut self` because each call may advance the
/// ctx's internal scratch pool / position counter. Return types use
/// `Tensor<F16>` (the residual-stream dtype); ops that produce other
/// dtypes (the LM-head logits, sampling outputs) write through the
/// ctx's owned slot rather than returning a tensor.
pub trait ForwardCtx {
    /// Embed `token_id` using `token_embd`. Writes a fresh F16 row
    /// (hidden) and returns it as the next-step residual.
    fn embed(&mut self, token_embd: &EmbeddingWeights, token_id: u32) -> Result<Tensor<F16>>;

    /// `output = rmsnorm(input) * weight`. Returns a fresh F16 row.
    fn rmsnorm(&mut self, input: &Tensor<F16>, weight: &Tensor<F16>) -> Result<Tensor<F16>>;

    /// Elementwise F16 add. Used for residual paths.
    fn residual_add(&mut self, a: Tensor<F16>, b: Tensor<F16>) -> Result<Tensor<F16>>;

    /// Standard transformer attention block: rmsnorm-quant → Q/K/V
    /// proj → optional Q/K norm → RoPE → KV append → flash-attn /
    /// splitk-attn → output proj. Topology-aware: TP shards
    /// Q/K/V/output cols and AR-sums after output_proj; PP/Hybrid
    /// likewise within each stage.
    ///
    /// `layer_idx` selects the KV slot. `position` is the token's
    /// position in the sequence (also the KV write tail).
    fn standard_attn(
        &mut self,
        input: &Tensor<F16>,
        weights: &AttnWeights,
        layer_idx: usize,
        position: usize,
    ) -> Result<Tensor<F16>>;

    /// Dense gated FFN (gate / up / activate / down). Topology-aware:
    /// TP shards gate+up cols, down rows, AR after down.
    fn dense_ffn(&mut self, input: &Tensor<F16>, weights: &FfnWeights) -> Result<Tensor<F16>>;

    /// Mixture-of-experts FFN: router → top-k → indexed expert
    /// matmuls → combine + (optional) shared expert. Topology-aware:
    /// TP shards expert intermediate dim, ARs the combined result.
    fn moe_ffn(&mut self, input: &Tensor<F16>, weights: &MoeWeights) -> Result<Tensor<F16>>;

    /// Output head: rmsnorm → lm_head matmul → (optional) logit
    /// softcap → host download into ctx's logits slot. Returns ()
    /// because the model only ever reads the logits via `ctx.logits()`
    /// after this call.
    fn output_head(
        &mut self,
        input: &Tensor<F16>,
        lm_head: &LmHeadWeights,
    ) -> Result<()>;

    /// Iterator over the layer indices THIS rank/stage processes
    /// in this forward call. PP: only this rank's assigned layers.
    /// TP: all layers (every rank runs every layer). Hybrid: only
    /// this stage's layers.
    ///
    /// The iterator's drop / final-yield site is where stage
    /// handoff (peer_copy + sync) lives on PP / Hybrid — the model
    /// sees only `for layer_idx in ctx.layer_range(layout) { ... }`.
    fn layer_range<'a>(&'a mut self, layout: &'a ModelLayout) -> Box<dyn Iterator<Item = usize> + 'a>;

    /// Read the host-side F32 logits after `output_head`. Caller
    /// owns the lifetime; the slice is valid until the next forward
    /// step.
    fn logits(&self) -> &[f32];
}

// ----------------------------------------------------------------
// Weight handles — placeholder shapes.
//
// These are the public arg types passed to composites. Each holds
// references / handles to weight `Tensor`s owned by the model crate.
// Concrete shapes land alongside the first model-v2 (qwen35).
// ----------------------------------------------------------------

/// Token-embedding weight handle.
pub struct EmbeddingWeights {
    pub token_embd: Tensor<F16>,
    pub vocab_size: usize,
    pub hidden: usize,
}

/// Per-layer attention weight handle. Shape captured at first model.
pub struct AttnWeights {
    // attn_norm, attn_q, attn_k, attn_v, attn_output, optional
    // q_norm / k_norm, RoPE params, head_dim, n_heads, n_kv_heads,
    // window, ...
    //
    // Concrete fields land with the first composite that needs them.
    // Kept minimal here so the trait surface can stabilise first.
    pub _todo: (),
}

/// Per-layer FFN weight handle.
pub struct FfnWeights {
    // ffn_norm, ffn_gate, ffn_up, ffn_down, activation kind, ...
    pub _todo: (),
}

/// Per-layer MoE weight handle.
pub struct MoeWeights {
    // router_gate, expert_gate / up / down (stacked), router_norm
    // policy, top_k, optional shared_expert weights, ...
    pub _todo: (),
}

/// LM-head weight handle.
pub struct LmHeadWeights {
    pub output_norm: Tensor<F16>,
    pub lm_head: Tensor<F16>, // may alias token_embd for tied heads
    pub final_logit_softcap: Option<f32>,
    pub vocab_size: usize,
    pub hidden: usize,
}

/// Per-arch layout. Concrete shape lands with the first model.
pub struct ModelLayout {
    pub num_layers: usize,
    pub hidden: usize,
    // per-layer specs (n_heads, n_kv_heads, head_dim, is_swa, ffn_kind,
    // window, ...) land with first model.
    pub _todo: (),
}
