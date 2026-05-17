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
//! - Every method on this trait is implemented by all four impls
//!   (SingleDevice / Pp / Tp / Hybrid). No method exists on only one.
//! - Each method bottoms out in `flambeau-model-ops` leaf primitives.
//!   This trait does not call kernels directly.
//! - Trait carries no state; per-request state lives on the impl
//!   structs.

use anyhow::Result;
use flambeau_model_ops::{Tensor, F16, Q8_0};

/// Forward-pass context. One impl per topology; the model writes
/// `<C: ForwardCtx>` generic.
///
/// All composites take `&mut self` because each call may advance the
/// ctx's internal scratch cursor. Return types use `Tensor<F16>` (the
/// residual-stream dtype); ops that produce other dtypes (the LM-head
/// logits) write through the ctx's owned slot rather than returning a
/// tensor.
pub trait ForwardCtx {
    /// Embed `token_id` using `token_embd`. Writes a fresh F16 row
    /// (hidden) and returns it as the next-step residual.
    fn embed(&mut self, token_embd: &EmbeddingWeights, token_id: u32) -> Result<Tensor<F16>>;

    /// `output = rmsnorm(input) * weight`. Returns a fresh F16 row.
    fn rmsnorm(
        &mut self,
        input: &Tensor<F16>,
        weight: &Tensor<F16>,
        eps: f32,
    ) -> Result<Tensor<F16>>;

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
    fn output_head(&mut self, input: &Tensor<F16>, lm_head: &LmHeadWeights) -> Result<()>;

    /// Iterator over the layer indices THIS rank/stage processes
    /// in this forward call. SingleDevice/TP: all layers. PP: only
    /// this rank's assigned layers. Hybrid: only this stage's layers.
    ///
    /// The iterator's drop / final-yield site is where stage
    /// handoff (peer_copy + sync) lives on PP / Hybrid — the model
    /// sees only `for layer_idx in ctx.layer_range(layout) { ... }`.
    fn layer_range<'a>(
        &'a mut self,
        layout: &'a ModelLayout,
    ) -> Box<dyn Iterator<Item = usize> + 'a>;

    /// Read the host-side F32 logits after `output_head`. Caller
    /// owns the lifetime; the slice is valid until the next forward
    /// step.
    fn logits(&self) -> &[f32];
}

// ----------------------------------------------------------------
// Weight handles.
//
// V1 hardcodes Q8_0 for the matmul-quant slots — the simplest GGUF
// dtype that flambeau-quant ships a host quantizer for, so the P2
// synthetic test can quantise mock weights without a real GGUF.
// Generalising to a runtime-dispatched `QuantWeight` enum (covering
// Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) lands in P3 when the qwen35-v2 loader
// hits a real GGUF.
// ----------------------------------------------------------------

/// Token-embedding weight handle. P2 keeps the embedding F16 to avoid
/// host-roundtrip dequant in the synthetic test; P3 lifts this to a
/// runtime-tagged variant.
pub struct EmbeddingWeights {
    pub token_embd: Tensor<F16>,
    pub vocab_size: usize,
    pub hidden: usize,
}

/// Per-layer attention weight handle. Shape mirrors qwen3.5 dense:
/// rmsnorm + Q/K/V/output projection + optional q/k norm + RoPE +
/// optional SWA. The `partial_rotated_dims` slot selects between
/// full-RoPE (`rotated_dims == head_dim`) and NeoX-partial RoPE.
pub struct AttnWeights {
    pub attn_norm: Tensor<F16>,
    pub attn_q: Tensor<Q8_0>,
    pub attn_k: Tensor<Q8_0>,
    pub attn_v: Tensor<Q8_0>,
    pub attn_output: Tensor<Q8_0>,
    pub attn_q_norm: Option<Tensor<F16>>,
    pub attn_k_norm: Option<Tensor<F16>>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Width of the rotated dimension subset. Equals `head_dim` for
    /// full-RoPE; less for NeoX-partial (qwen3.x full-attn layers).
    pub rotated_dims: usize,
    pub rope_theta: f32,
    /// 0 = unbounded causal. Positive = SWA radius.
    pub window_size: i32,
    pub rms_eps: f32,
    /// Optional explicit softmax scale. `None` ⇒ default `1/sqrt(head_dim)`.
    pub softmax_scale: Option<f32>,
}

/// Per-layer dense FFN weight handle (qwen-style gated MLP).
pub struct FfnWeights {
    pub ffn_norm: Tensor<F16>,
    pub ffn_gate: Tensor<Q8_0>,
    pub ffn_up: Tensor<Q8_0>,
    pub ffn_down: Tensor<Q8_0>,
    pub activation: Activation,
    pub rms_eps: f32,
}

/// FFN activation kind. SwiGLU for qwen / mistral; GELU-tanh for gemma4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    SwiGLU,
    GeluTanh,
}

/// Per-layer MoE weight handle. P2 keeps this minimal — qwen3.5
/// dense + the synthetic test never call `moe_ffn`. Real fields land
/// with qwen3.6-v2 in P7.
pub struct MoeWeights {
    pub _placeholder: (),
}

/// LM-head weight handle. `lm_head` may alias `token_embd` for tied
/// heads (gemma4); the loader sets up the alias.
pub struct LmHeadWeights {
    pub output_norm: Tensor<F16>,
    pub lm_head: Tensor<Q8_0>,
    pub final_logit_softcap: Option<f32>,
    pub vocab_size: usize,
    pub hidden: usize,
    pub rms_eps: f32,
}

/// Per-arch layout. Carries everything the topology executor needs
/// to size its scratch + KV slots without re-reading the GGUF.
pub struct ModelLayout {
    pub num_layers: usize,
    pub hidden: usize,
    pub kv_max_seq_len: usize,
}
