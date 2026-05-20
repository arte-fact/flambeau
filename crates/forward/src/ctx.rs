//! `ForwardCtx` — the trait every topology implements. The method set
//! is the composite-op vocabulary a model's forward function writes
//! against; each impl handles the topology placement (AR, peer-copy,
//! handoff). See `core/` for the shared composite engine.

use anyhow::Result;
use flambeau_core::DevicePtr;
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

    /// Multiply a `[n_tokens, hidden]` F16 buffer by `scale` in place.
    /// Used for gemma4's per-layer `layer_output_scale` applied to the
    /// residual after both attention + FFN residuals.
    fn scale_inplace_f16(
        &mut self,
        buf: Tensor<F16>,
        scale: f32,
        n_tokens: usize,
    ) -> Result<Tensor<F16>>;

    /// `positions[i]` is the KV write row for `tokens[i]`; `slot_ids[i]`
    /// is the inflight-slot index whose KV slab receives that write.
    /// Single decode: positions=[pos], slot_ids=[0]. Prefill chunk:
    /// positions=[start..start+n], slot_ids=[slot;n]. Batched decode:
    /// positions=[pos_0..pos_{N-1}], slot_ids=[0..N-1].
    /// `Result<Option<Tensor>>`: `Some(delta)` means the caller must
    /// follow with `residual_add(input, delta)`. `None` means the
    /// composite has already folded AR + residual-add into `input`
    /// in-place via the BAR1 fused kernel (no separate `residual_add`
    /// needed and harmful to attempt). Fusion only kicks in when
    /// (a) TP+BAR1 is engaged and (b) no per-arch op (e.g., gemma4
    /// `post_attn_norm`) sits between AR and residual-add.
    /// `next_norm` is the FOLLOWING composite's input rmsnorm weight
    /// (e.g., `&ffn[li].ffn_norm` after attn). When the composite
    /// fuses AR + residual + rmsnorm-of-next, it writes the rmsnormed
    /// buffer to `pool.norm` and sets `pool.input_pre_normed = true`;
    /// the next composite skips its own rmsnorm. `None` skips the
    /// fold even on TP.
    fn standard_attn(
        &mut self,
        input: &Tensor<F16>,
        weights: &AttnWeights,
        layer_idx: usize,
        positions: &[usize],
        slot_ids: &[usize],
        next_norm: Option<&Tensor<F16>>,
    ) -> Result<Option<Tensor<F16>>>;

    fn gdn_layer(
        &mut self,
        input: &Tensor<F16>,
        weights: &GdnWeights,
        layer_idx: usize,
        slot_ids: &[usize],
        next_norm: Option<&Tensor<F16>>,
    ) -> Result<Option<Tensor<F16>>>;

    fn dense_ffn(
        &mut self,
        input: &Tensor<F16>,
        weights: &FfnWeights,
        n_tokens: usize,
        next_norm: Option<&Tensor<F16>>,
    ) -> Result<Option<Tensor<F16>>>;

    fn moe_ffn(
        &mut self,
        input: &Tensor<F16>,
        weights: &MoeWeights,
        n_tokens: usize,
        next_norm: Option<&Tensor<F16>>,
    ) -> Result<Option<Tensor<F16>>>;

    /// Per-layer side-channel embedding apply (gemma 4n / E2B / E4B).
    /// Reads `resid` (F16, length hidden), reads the slice
    /// `table_dev[layer_idx * pe .. (layer_idx + 1) * pe]` (F32),
    /// rewrites `resid` in place with the side-channel residual.
    /// `pe` is the per-layer side-channel width (256 on E4B).
    /// Default impl panics — only impls that own a Pool can run the
    /// apply (engine, testing). RecordingCtx records the call.
    fn per_layer_embd_apply(
        &mut self,
        resid: &mut Tensor<F16>,
        weights: &flambeau_blocks::per_layer_embd::PerLayerEmbedLayerWeights,
        table_dev: DevicePtr,
        layer_idx: usize,
        pe: usize,
        rms_eps: f32,
    ) -> Result<()> {
        let _ = (resid, weights, table_dev, layer_idx, pe, rms_eps);
        anyhow::bail!("per_layer_embd_apply not implemented for this ctx")
    }

    /// Per-token build + upload of the side-channel embedding table.
    /// Reads `main_embd` (F16 [hidden] for the current token) via DtoH,
    /// runs
    /// [`flambeau_blocks::per_layer_embd::build_inp_per_layer_table`]
    /// host-side, and uploads the resulting `[n_layer * pe]` F32 table
    /// to `table_dev`. Caller passes the GGUF raw byte slices for the
    /// three per-layer-embd globals plus the row-sliced token embedding
    /// bytes for the current token.
    #[allow(clippy::too_many_arguments)]
    fn per_layer_embd_build_table(
        &mut self,
        main_embd: &Tensor<F16>,
        tok_embd_row_raw: &[u8],
        tok_embd_dtype: flambeau_quant::GgmlDType,
        model_proj_raw: &[u8],
        model_proj_dtype: flambeau_quant::GgmlDType,
        proj_norm_raw: &[u8],
        table_dev: DevicePtr,
        pe: usize,
        n_layer: usize,
        hidden: usize,
        rms_eps: f32,
    ) -> Result<()> {
        let _ = (
            main_embd, tok_embd_row_raw, tok_embd_dtype, model_proj_raw, model_proj_dtype,
            proj_norm_raw, table_dev, pe, n_layer, hidden, rms_eps,
        );
        anyhow::bail!("per_layer_embd_build_table not implemented for this ctx")
    }

    /// When `slot_ids` are all equal (prefill / single decode), only
    /// the LAST token's logits land in `ctx.logits()` (vocab elems).
    /// When all distinct (batched-decode), N rows of logits land in
    /// `ctx.logits()` in row-major `[N, vocab]` order — caller slices
    /// `slot_ids[i]`'s logits from row `i`.
    fn output_head(
        &mut self,
        input: &Tensor<F16>,
        lm_head: &LmHeadWeights,
        slot_ids: &[usize],
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

    /// True iff `dtype` has an F16-direct MMVQ kernel registered in
    /// `ops.mmvq_f16_direct`. Callers gate the fused-cast fast path on
    /// this; unsupported dtypes (F16, BF16) keep the F32+cast pair.
    pub fn supports_decode_to_f16(&self) -> bool {
        use flambeau_core::op::QDtype;
        matches!(
            self.dtype,
            QDtype::Q4_0
                | QDtype::Q4_1
                | QDtype::Q5_0
                | QDtype::Q5_1
                | QDtype::Q8_0
                | QDtype::Q2_K
                | QDtype::Q3_K
                | QDtype::Q4_K
                | QDtype::Q5_K
                | QDtype::Q6_K
                | QDtype::Q8_K
                | QDtype::IQ1_S
                | QDtype::IQ1_M
                | QDtype::IQ2_XXS
                | QDtype::IQ2_XS
                | QDtype::IQ2_S
                | QDtype::IQ3_XXS
                | QDtype::IQ3_S
                | QDtype::IQ4_NL
                | QDtype::IQ4_XS
        )
    }

    /// Decode-only (`m=1`) MMVQ writing directly into an F16
    /// destination. Saturating cast happens inside the kernel — saves
    /// one `cast_f32_to_f16` launch per call. Caller must check
    /// [`Self::supports_decode_to_f16`] first; F16/BF16 weights bail.
    pub fn qmatmul_decode_to_f16(
        &self,
        act_q8_1: &Tensor<flambeau_model_ops::Q8_1>,
        output: &mut Tensor<flambeau_model_ops::F16>,
        k: usize,
        n: usize,
        ops: &flambeau_ops::HipOps<'_>,
    ) -> Result<()> {
        if output.n_elems < n {
            anyhow::bail!(
                "qmatmul_decode_to_f16 ({:?}): output has {} F16 elems, need >= {n}",
                self.dtype,
                output.n_elems,
            );
        }
        <flambeau_ops::HipOps<'_> as flambeau_ops::Ops>::mmvq_f16_direct(
            ops,
            self.ptr,
            act_q8_1.ptr,
            output.ptr,
            n,
            k,
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
    /// Optional norm applied to the F16 attention delta BEFORE the
    /// outer residual_add. Gemma4 sets this to `post_attention_norm.weight`;
    /// every other arch leaves it `None`.
    pub post_attn_norm: Option<Tensor<F16>>,
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
    /// Optional unit-weights F16 tensor sized `[head_dim]`. When
    /// `Some`, the V tensor receives per-head `rmsnorm_f16(V, ones)`
    /// before the attention compute. Gemma4 trained this in; legacy
    /// allocates an equivalent `v_ones_f16` scratch.
    pub attn_v_unit_norm_w: Option<Tensor<F16>>,
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
    /// Per-layer KV-cache share source (gemma 4n / E2B / E4B's
    /// `shared_kv_layers`). When `Some(src_layer)`, this layer's
    /// attention reads K/V from layer `src_layer`'s KV cache slot
    /// instead of its own, and skips the `kv_append` write. The Q
    /// projection still uses this layer's `attn_q`. The K/V
    /// projection bytes are computed but discarded (cheaper than
    /// branching the composite). `None` ⇒ normal own-KV path.
    pub kv_share_src: Option<usize>,
}

pub struct FfnWeights {
    pub ffn_norm: Tensor<F16>,
    /// Optional norm applied to the F16 FFN delta BEFORE the outer
    /// residual_add. Gemma4 sets this to `post_ffw_norm.weight`;
    /// every other arch leaves it `None`.
    pub post_ffn_norm: Option<Tensor<F16>>,
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
///
/// Two shapes share this struct:
///
/// * qwen3-moe (default): single pre-norm (`ffn_norm`), optional shared
///   expert with per-token sigmoid gate, optional F16 `post_ffn_norm`.
///   The cascade fields below stay `None`.
/// * gemma4 MoE (26B-A4B): a 5-norm F32 cascade. All cascade fields are
///   populated; the composite detects this by
///   `pre_router_weight_f16.is_some()` and runs the F32 partial path
///   (`flambeau-model-ops::moe_cascade_*`).
pub struct MoeWeights {
    pub ffn_norm: Tensor<F16>,
    /// Optional norm applied to the F16 MoE delta BEFORE the outer
    /// residual_add. qwen3-moe leaves it `None`; gemma4 MoE uses the
    /// F32 cascade fields below instead.
    pub post_ffn_norm: Option<Tensor<F16>>,
    pub router: QuantWeight,
    pub experts_gate: Vec<QuantWeight>,
    pub experts_up: Vec<QuantWeight>,
    pub experts_down: Vec<QuantWeight>,
    pub n_experts: usize,
    pub experts_per_tok: usize,
    pub activation: Activation,
    pub rms_eps: f32,
    /// Always-on shared expert / shared MLP run in parallel with the
    /// routed experts. qwen3-moe family attaches it with a per-token
    /// sigmoid gate (`SharedExpertWeights::gate_inp = Some(...)`);
    /// gemma4 MoE attaches it as a plain dense FFN
    /// (`gate_inp = None`).
    pub shared: Option<SharedExpertWeights>,

    // ---- gemma4 cascade fields (all Some together, all None for qwen) ----
    /// F16 `[hidden]` rmsnorm weight applied to the attn-residual
    /// BEFORE the router. Gemma4-only — derived at load from
    /// `ffn_gate_inp.scale` (F32 [hidden]) × 1/sqrt(hidden), then cast
    /// to F16.
    pub pre_router_weight_f16: Option<Tensor<F16>>,
    /// F16 `[hidden]` rmsnorm weight applied to the attn-residual
    /// BEFORE the routed-MoE branch (separate from `ffn_norm` which is
    /// the shared-MLP pre-norm).
    pub pre_ffw_norm_2_f16: Option<Tensor<F16>>,
    /// F32 `[hidden]` rmsnorm weight applied to the shared-MLP F32
    /// partial (before combine).
    pub post_ffw_norm_1_f32: Option<Tensor<F32>>,
    /// F32 `[hidden]` rmsnorm weight applied to the routed-MoE F32
    /// partial (before combine).
    pub post_ffw_norm_2_f32: Option<Tensor<F32>>,
    /// F32 `[hidden]` rmsnorm weight applied to the combined F32
    /// partial after `cur_mlp + cur_moe`. The result is cast F32→F16
    /// and added to the residual.
    pub post_ffn_norm_f32: Option<Tensor<F32>>,
    /// F32 `[n_experts]` per-expert weight scale folded into the
    /// router top-k weights before the indexed-MoE forward.
    pub expert_down_scale_f32: Option<Tensor<F32>>,
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
