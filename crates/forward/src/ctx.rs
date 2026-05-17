//! `ForwardCtx` — the trait every topology implements. The method set
//! is the composite-op vocabulary a model's forward function writes
//! against; each impl handles the topology placement (AR, peer-copy,
//! handoff). See `core/` for the shared composite engine.

use anyhow::Result;
use flambeau_model_ops::{Tensor, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0};

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
}

pub struct AttnWeights {
    pub attn_norm: Tensor<F16>,
    pub attn_q: QuantWeight,
    pub attn_k: QuantWeight,
    pub attn_v: QuantWeight,
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

pub struct MoeWeights {
    pub _placeholder: (),
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
