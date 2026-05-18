//! `ForwardCtx` — the trait every topology implements. The method set
//! is the composite-op vocabulary a model's forward function writes
//! against; each impl handles the topology placement (AR, peer-copy,
//! handoff). See `core/` for the shared composite engine.

use anyhow::Result;
use flambeau_model_ops::{Tensor, F16, F32, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0};

/// One impl per topology; the model is `<C: ForwardCtx>` generic.
pub trait ForwardCtx {
    fn embed(&mut self, token_embd: &EmbeddingWeights, token_id: u32) -> Result<Tensor<F16>>;

    fn rmsnorm(
        &mut self,
        input: &Tensor<F16>,
        weight: &Tensor<F16>,
        eps: f32,
    ) -> Result<Tensor<F16>>;

    fn residual_add(&mut self, a: Tensor<F16>, b: Tensor<F16>) -> Result<Tensor<F16>>;

    /// `layer_idx` selects the KV slot; `position` is the KV write tail.
    fn standard_attn(
        &mut self,
        input: &Tensor<F16>,
        weights: &AttnWeights,
        layer_idx: usize,
        position: usize,
    ) -> Result<Tensor<F16>>;

    /// Gated-Delta-Net recurrent layer (Qwen3.5 / 3.6 / 3-Next).
    /// `layer_idx` selects the per-layer state + conv history slot.
    fn gdn_layer(
        &mut self,
        input: &Tensor<F16>,
        weights: &GdnWeights,
        layer_idx: usize,
    ) -> Result<Tensor<F16>>;

    fn dense_ffn(&mut self, input: &Tensor<F16>, weights: &FfnWeights) -> Result<Tensor<F16>>;

    fn moe_ffn(&mut self, input: &Tensor<F16>, weights: &MoeWeights) -> Result<Tensor<F16>>;

    /// Logits land in `ctx.logits()` after this returns.
    fn output_head(&mut self, input: &Tensor<F16>, lm_head: &LmHeadWeights) -> Result<()>;

    /// Layer indices THIS rank/stage processes. SingleDevice/TP: all.
    /// PP: this rank's slice. Hybrid: this stage's slice.
    fn layer_range<'a>(
        &'a mut self,
        layout: &'a ModelLayout,
    ) -> Box<dyn Iterator<Item = usize> + 'a>;

    /// Host-side F32 logits. Valid until the next forward step.
    fn logits(&self) -> &[f32];
}

/// Runtime-tagged quant weight. `QuantWeight::qmatmul` dispatches to
/// the matching `flambeau_model_ops::qmatmul_q*`.
pub enum QuantWeight {
    Q4_0(Tensor<Q4_0>),
    Q4_1(Tensor<Q4_1>),
    Q5_0(Tensor<Q5_0>),
    Q5_1(Tensor<Q5_1>),
    Q8_0(Tensor<Q8_0>),
}

impl QuantWeight {
    #[allow(clippy::too_many_arguments)]
    pub fn qmatmul(
        &self,
        act_q8_1: &Tensor<flambeau_model_ops::Q8_1>,
        act_q8_1_mmq: &Tensor<flambeau_model_ops::Q8_1>,
        output: &mut Tensor<flambeau_model_ops::F32>,
        m: usize,
        k: usize,
        n: usize,
        ops: &flambeau_ops::HipOps<'_>,
    ) -> Result<()> {
        match self {
            Self::Q4_0(w) => {
                flambeau_model_ops::qmatmul_q4_0(w, act_q8_1, act_q8_1_mmq, output, m, k, n, ops)
            }
            Self::Q4_1(w) => {
                flambeau_model_ops::qmatmul_q4_1(w, act_q8_1, act_q8_1_mmq, output, m, k, n, ops)
            }
            Self::Q5_0(w) => {
                flambeau_model_ops::qmatmul_q5_0(w, act_q8_1, act_q8_1_mmq, output, m, k, n, ops)
            }
            Self::Q5_1(w) => {
                flambeau_model_ops::qmatmul_q5_1(w, act_q8_1, act_q8_1_mmq, output, m, k, n, ops)
            }
            Self::Q8_0(w) => {
                flambeau_model_ops::qmatmul_q8_0(w, act_q8_1, act_q8_1_mmq, output, m, k, n, ops)
            }
        }
    }
}

pub struct EmbeddingWeights {
    pub token_embd: Tensor<F16>,
    pub vocab_size: usize,
    pub hidden: usize,
    /// Optional post-embed scalar multiply applied in-place. Gemma4
    /// uses `sqrt(n_embd)` (memory: `parity_vs_argmax_in_vocab`).
    pub post_scale: Option<f32>,
}

pub struct AttnWeights {
    pub attn_norm: Tensor<F16>,
    /// When `attn_q_gated`, the underlying tensor has
    /// `[2 * n_heads * head_dim, hidden]` rows in head-interleaved
    /// `[head_i_Q | head_i_gate]` layout; the composite splits the
    /// matmul output per head and applies a sigmoid gate after
    /// attention. Plain Q-only otherwise.
    pub attn_q: QuantWeight,
    pub attn_k: QuantWeight,
    /// `None` for gemma4-style "V = K via memcpy" layers (no attn_v
    /// weight on disk). `Some` means run an independent V projection.
    pub attn_v: Option<QuantWeight>,
    pub attn_output: QuantWeight,
    pub attn_q_norm: Option<Tensor<F16>>,
    pub attn_k_norm: Option<Tensor<F16>>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Equals `head_dim` for full-RoPE; less for NeoX-partial.
    pub rotated_dims: usize,
    pub rope_theta: f32,
    /// 0 = unbounded causal. Positive = SWA radius.
    pub window_size: i32,
    pub rms_eps: f32,
    /// `None` ⇒ default `1/sqrt(head_dim)`.
    pub softmax_scale: Option<f32>,
    /// `true` for qwen3.5 / qwen3.6 / qwen3-Next full-attention
    /// layers (fused Q+gate in `attn_q`). `false` for plain Q.
    pub attn_q_gated: bool,
}

pub struct FfnWeights {
    pub ffn_norm: Tensor<F16>,
    pub ffn_gate: QuantWeight,
    pub ffn_up: QuantWeight,
    pub ffn_down: QuantWeight,
    pub activation: Activation,
    pub rms_eps: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    SwiGLU,
    GeluTanh,
}

/// Routing: TopkRenorm (top-k by raw logit, then softmax over k).
/// Matches qwen3.x / gemma4 / DeepSeek conventions; algebraically
/// identical to `softmax(all)→topk→renorm` (memory:
/// `moe_topk_softmax_equivalence`).
pub struct MoeWeights {
    pub ffn_norm: Tensor<F16>,
    pub router: QuantWeight,
    pub experts_gate: Vec<QuantWeight>,
    pub experts_up: Vec<QuantWeight>,
    pub experts_down: Vec<QuantWeight>,
    pub n_experts: usize,
    pub experts_per_tok: usize,
    pub activation: Activation,
    pub rms_eps: f32,
}

/// `lm_head` may alias `token_embd` for tied heads.
pub struct LmHeadWeights {
    pub output_norm: Tensor<F16>,
    pub lm_head: QuantWeight,
    pub final_logit_softcap: Option<f32>,
    pub vocab_size: usize,
    pub hidden: usize,
    pub rms_eps: f32,
}

pub struct ModelLayout {
    pub num_layers: usize,
    pub hidden: usize,
    pub kv_max_seq_len: usize,
}

/// Per-layer attention kind. Hybrid archs (qwen3.5 / 3.6 / 3-Next)
/// alternate `FullAttn` and `Gdn`; pure-dense archs are all `FullAttn`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    FullAttn,
    Gdn,
}

/// Per-layer GDN weight handle. Mirrors `flambeau_blocks::DeltaNetLayer`
/// fields (the block this composite delegates to). All matmul weights
/// go through `QuantWeight`; norm + SSM scalars are typed tensors.
pub struct GdnWeights {
    pub attn_norm: Tensor<F16>,
    pub attn_qkv: QuantWeight,
    pub attn_gate: QuantWeight,
    pub ssm_alpha: QuantWeight,
    pub ssm_beta: QuantWeight,
    pub ssm_out: QuantWeight,
    pub ssm_dt_bias: Tensor<F32>,
    pub ssm_a: Tensor<F32>,
    pub ssm_conv1d: Tensor<F32>,
    pub ssm_norm_w: Tensor<F16>,
    pub dims: GdnDims,
    pub rms_eps: f32,
    /// `false` for cyclic `ggml_repeat_4d` (qwen3.5/3.6); `true` for
    /// reshape-interleave (qwen3-Next). Wrong choice → degenerate logits.
    pub rep_inner_layout: bool,
}

/// Shape parameters shared across every GDN layer in a model.
#[derive(Clone, Copy, Debug, Default)]
pub struct GdnDims {
    pub d_inner: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
}
