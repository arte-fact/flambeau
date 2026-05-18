//! `ForwardCtx` — the trait every topology implements. The method set
//! is the composite-op vocabulary a model's forward function writes
//! against; each impl handles the topology placement (AR, peer-copy,
//! handoff). See `core/` for the shared composite engine.

use anyhow::Result;
use flambeau_model_ops::{Tensor, F16, F32};

/// One impl per topology; the model is `<C: ForwardCtx>` generic.
/// `n_tokens` is the runtime token-batch dim: `1` for decode,
/// `prompt.len()` (or chunk size) for prefill. Composites use the same
/// kernels at N=1 and N>1; the leaf-kernel branch (e.g. attn_decode vs
/// attn_prefill) lives inside the composite, not in this trait.
pub trait ForwardCtx {
    fn embed(
        &mut self,
        token_embd: &EmbeddingWeights,
        tokens: &[u32],
    ) -> Result<Tensor<F16>>;

    fn rmsnorm(
        &mut self,
        input: &Tensor<F16>,
        weight: &Tensor<F16>,
        eps: f32,
        n_tokens: usize,
    ) -> Result<Tensor<F16>>;

    fn residual_add(
        &mut self,
        a: Tensor<F16>,
        b: Tensor<F16>,
        n_tokens: usize,
    ) -> Result<Tensor<F16>>;

    /// `positions[i]` is the KV write row for `tokens[i]`; `slot_ids[i]`
    /// is the inflight-slot index whose KV slab receives that write.
    /// Single decode: positions=[pos], slot_ids=[0]. Prefill chunk:
    /// positions=[start..start+n], slot_ids=[slot;n]. Batched decode:
    /// positions=[pos_0..pos_{N-1}], slot_ids=[0..N-1].
    fn standard_attn(
        &mut self,
        input: &Tensor<F16>,
        weights: &AttnWeights,
        layer_idx: usize,
        positions: &[usize],
        slot_ids: &[usize],
    ) -> Result<Tensor<F16>>;

    /// Gated-Delta-Net recurrent layer. `slot_ids[i]` selects the
    /// per-slot recurrent state slab updated by token `i`.
    fn gdn_layer(
        &mut self,
        input: &Tensor<F16>,
        weights: &GdnWeights,
        layer_idx: usize,
        slot_ids: &[usize],
    ) -> Result<Tensor<F16>>;

    fn dense_ffn(
        &mut self,
        input: &Tensor<F16>,
        weights: &FfnWeights,
        n_tokens: usize,
    ) -> Result<Tensor<F16>>;

    fn moe_ffn(
        &mut self,
        input: &Tensor<F16>,
        weights: &MoeWeights,
        n_tokens: usize,
    ) -> Result<Tensor<F16>>;

    /// Logits for the LAST token land in `ctx.logits()` — at prefill
    /// only the next-token sampler needs the final row.
    fn output_head(
        &mut self,
        input: &Tensor<F16>,
        lm_head: &LmHeadWeights,
        n_tokens: usize,
    ) -> Result<()>;

    fn layer_range<'a>(
        &'a mut self,
        layout: &'a ModelLayout,
    ) -> Box<dyn Iterator<Item = usize> + 'a>;

    fn logits(&self) -> &[f32];
}

/// Runtime-tagged quant weight handle. Holds the device pointer plus
/// the dtype tag the HIP kernel dispatcher needs; `qmatmul` is one
/// call into `ops.qmatmul` regardless of dtype. Adding a new quant
/// family is a single arm in `loader::ggml_to_qdtype`, not a fresh
/// enum variant + match arm + typed wrapper.
#[derive(Clone, Copy, Debug)]
pub struct QuantWeight {
    pub ptr: flambeau_core::DevicePtr,
    pub dtype: flambeau_core::op::QDtype,
    pub n_elems: usize,
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
        if output.n_elems < m * n {
            anyhow::bail!(
                "qmatmul ({:?}): output has {} F32 elems, need >= {m}*{n}={}",
                self.dtype,
                output.n_elems,
                m * n
            );
        }
        <flambeau_ops::HipOps<'_> as flambeau_ops::Ops>::qmatmul(
            ops,
            self.ptr,
            act_q8_1.ptr,
            act_q8_1_mmq.ptr,
            output.ptr,
            m,
            k,
            n,
            self.dtype,
        )
    }
}

pub struct EmbeddingWeights {
    pub token_embd: Tensor<F16>,
    pub vocab_size: usize,
    pub hidden: usize,
    /// Optional post-embed scalar multiply applied in-place
    /// (gemma4 uses `sqrt(n_embd)`).
    pub post_scale: Option<f32>,
}

impl EmbeddingWeights {
    /// NULL-ptr placeholder for PP ranks that don't own layer 0.
    /// `PpForwardCtx::embed` only dereferences the weights on the
    /// first rank; non-first ranks peer-receive into the residual
    /// slot and never touch the token_embd tensor.
    pub fn placeholder(vocab_size: usize, hidden: usize) -> Self {
        Self {
            // SAFETY: NULL ptr + 0 elems makes the tensor opaque;
            // safe construction since no read ever fires on it.
            token_embd: unsafe { Tensor::<F16>::from_raw(flambeau_core::DevicePtr::NULL, 0) },
            vocab_size,
            hidden,
            post_scale: None,
        }
    }
}

/// RoPE layout. `Interleaved` rotates `(x[2i], x[2i+1])` pairs
/// (gemma4). `NeoxSplit` rotates `(x[i], x[i + rotated_dims/2])`
/// over the first `rotated_dims` (qwen3, qwen3-next).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeVariant {
    Interleaved,
    NeoxSplit,
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
    pub rope_variant: RopeVariant,
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

/// MoE routing: top-k by raw logit, then softmax over k
/// (algebraically equivalent to `softmax(all) → topk → renorm`).
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
    /// Always-on shared expert (qwen3-moe family). When `Some`, its
    /// per-token-sigmoid-gated dense FFN output is added to the routed
    /// experts' accumulator before the residual.
    pub shared: Option<SharedExpertWeights>,
}

pub struct SharedExpertWeights {
    pub gate: QuantWeight,
    pub up: QuantWeight,
    pub down: QuantWeight,
    /// F32 `[hidden]` per-token scalar gate for the dense output;
    /// `Some` for Qwen3.6-35B-A3B and qwen3next, `None` for variants
    /// that emit the dense output unscaled.
    pub gate_inp: Option<Tensor<F32>>,
    pub intermediate: usize,
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

impl LmHeadWeights {
    /// NULL-ptr placeholder for PP ranks that don't own the last layer.
    /// `PpForwardCtx::output_head` only dereferences the weights on the
    /// last rank; non-last ranks peer-send the post-final-layer hidden
    /// state and never invoke `output_head_local`.
    pub fn placeholder(vocab_size: usize, hidden: usize, rms_eps: f32) -> Self {
        use flambeau_core::op::QDtype;
        Self {
            // SAFETY: NULL ptr + 0 elems is opaque; never read.
            output_norm: unsafe {
                Tensor::<F16>::from_raw(flambeau_core::DevicePtr::NULL, 0)
            },
            lm_head: QuantWeight {
                ptr: flambeau_core::DevicePtr::NULL,
                dtype: QDtype::F16,
                n_elems: 0,
            },
            final_logit_softcap: None,
            vocab_size,
            hidden,
            rms_eps,
        }
    }
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
    pub ssm_norm_w: Tensor<F32>,
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
