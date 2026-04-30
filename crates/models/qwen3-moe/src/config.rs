//! Qwen3.x MoE config parsed from GGUF metadata.
//!
//! Covers two closely-related Qwen3.x architectures:
//!
//! - `qwen3moe` — pure transformer MoE (Qwen3 / Qwen3-Coder 30B-A3B class).
//!   Every layer is full self-attention with GQA + RoPE, followed by a MoE
//!   FFN. No SSM, no shared expert, no `full_attention_interval`.
//! - `qwen35moe` / `qwen36moe` — hybrid Gated-Delta-Net + full-attention +
//!   MoE-with-shared-expert. Layers alternate: `(il + 1) % full_attention_interval != 0`
//!   → recurrent/GDN, else full-attention.
//!
//! Key GGUF metadata keys (namespaced by `{arch}`):
//! - `{arch}.attention.head_count`, `.head_count_kv`, `.key_length`, `.value_length`,
//!   `.layer_norm_rms_epsilon`
//! - `{arch}.embedding_length`, `.block_count`, `.context_length`
//! - `{arch}.rope.freq_base`, `.rope.dimension_count`, `.rope.dimension_sections`
//! - `{arch}.expert_count`, `.expert_used_count`, `.expert_feed_forward_length`,
//!   `.expert_shared_feed_forward_length`
//! - `{arch}.full_attention_interval`
//! - `{arch}.ssm.inner_size`, `.ssm.state_size`, `.ssm.group_count`,
//!   `.ssm.time_step_rank`, `.ssm.conv_kernel`

use flambeau_quant::GgufFile;

#[derive(Debug, thiserror::Error)]
pub enum Qwen3MoEConfigError {
    #[error("expected a Qwen3-family architecture (qwen35 / qwen35moe / qwen36moe), got {got:?}")]
    WrongArchitecture { got: Option<String> },
    #[error("missing GGUF metadata key `{0}`")]
    MissingKey(String),
    #[error("metadata key `{0}` had unexpected type")]
    BadType(String),
}

/// Architecture tags this crate handles. Each value shares the `{arch}.foo`
/// metadata convention + `blk.{i}.*` tensor naming, but differs in which
/// ops the forward pass composes.
/// Architectures this crate parses config for. `qwen35` is the dense-hybrid
/// variant (Qwen3.5 / 3.6 without MoE — pure dense FFN per layer). V2.2
/// scaffold: config parser accepts it, but the weight loader + forward path
/// still assume MoE — a qwen35 GGUF will fail to LOAD until the dense-FFN
/// paths are wired (see `doc/V2-BACKLOG.md#V2.2`).
pub const SUPPORTED_ARCHS: &[&str] = &["qwen35moe", "qwen36moe", "qwen35", "qwen3next"];

/// Which attention family the model uses.
///
/// `Hybrid` implies the presence of `ssm.*` metadata and `full_attention_interval`.
/// `Dense` is pure-transformer MoE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionFamily {
    /// Every layer is self-attention with GQA + RoPE (qwen3moe).
    Dense,
    /// Mix of Gated-Delta-Net (most layers) + Gated-Full-Attention (every Nth).
    Hybrid,
}

/// Gated-Delta-Net dimensions, derived from `{arch}.ssm.*`. Present iff
/// `family == Hybrid`.
#[derive(Debug, Clone)]
pub struct GdnDims {
    /// `ssm.inner_size` — `d_inner`, the SSM's internal channel count.
    pub d_inner: usize,
    /// `ssm.state_size` — `head_k_dim`, the per-head state dimension.
    pub head_k_dim: usize,
    /// `ssm.group_count` — number of K/Q heads inside GDN.
    pub num_k_heads: usize,
    /// `ssm.time_step_rank` — number of V heads (= dt rank).
    pub num_v_heads: usize,
    /// `ssm.conv_kernel` — causal conv1d kernel width. Usually 4.
    pub conv_kernel: usize,
}

impl GdnDims {
    /// `head_v_dim = d_inner / num_v_heads`.
    pub fn head_v_dim(&self) -> usize {
        self.d_inner / self.num_v_heads
    }

    /// `conv_channels = d_inner + 2 * num_k_heads * head_k_dim`.
    pub fn conv_channels(&self) -> usize {
        self.d_inner + 2 * self.num_k_heads * self.head_k_dim
    }
}

/// Multi-frequency partial-RoPE spec, used by `qwen35moe` full-attention
/// layers. `rope.dimension_count` gives how many of `head_dim` are rotated
/// (usually < head_dim → partial RoPE); `rope.dimension_sections` is a
/// 4-element array of per-section widths summing to `rope.dimension_count / 2`.
#[derive(Debug, Clone)]
pub struct RopeSpec {
    pub freq_base: f32,
    /// Dimensions rotated (others pass through unchanged). `<= head_dim`.
    pub rotated_dims: usize,
    /// Per-section widths `[s0, s1, s2, s3]` summing to `rotated_dims / 2`.
    /// When `None`, uniform-frequency RoPE applies across `rotated_dims`.
    pub sections: Option<[u32; 4]>,
}

/// Every number the forward pass needs, pulled once at load time.
#[derive(Debug, Clone)]
pub struct Qwen3MoEConfig {
    /// `qwen3moe` / `qwen35moe` / `qwen36moe` — `general.architecture`.
    pub arch: String,
    /// Which attention family this arch uses.
    pub family: AttentionFamily,

    /// Hidden size (model dim). GGUF: `{arch}.embedding_length`.
    pub hidden_size: usize,
    /// Vocabulary size, taken from `token_embd.weight`'s outer dim.
    pub vocab_size: usize,
    /// Transformer blocks. `{arch}.block_count`.
    pub num_layers: usize,
    /// Q heads per layer. `{arch}.attention.head_count`.
    pub num_heads: usize,
    /// KV heads per layer (GQA). `{arch}.attention.head_count_kv`.
    pub num_kv_heads: usize,
    /// Per-head dim. `{arch}.attention.key_length`, falling back to
    /// `hidden_size / num_heads`.
    pub head_dim: usize,
    /// Max sequence length. `{arch}.context_length`.
    pub context_length: usize,
    /// RMSNorm epsilon. `{arch}.attention.layer_norm_rms_epsilon`.
    pub rms_norm_eps: f32,

    /// RoPE spec (frequency base, rotated dims, optional multi-section widths).
    pub rope: RopeSpec,

    /// MoE experts per layer. `{arch}.expert_count`.
    pub num_experts: usize,
    /// Top-k experts activated per token. `{arch}.expert_used_count`.
    pub num_experts_per_tok: usize,
    /// Routed-expert gate/up/down intermediate dim. `{arch}.expert_feed_forward_length`.
    pub moe_intermediate_size: usize,
    /// Shared-expert intermediate dim, `Some(..)` iff the arch has an
    /// always-on shared expert. `{arch}.expert_shared_feed_forward_length`.
    pub shared_expert_intermediate_size: Option<usize>,

    /// Every `(il + 1) % full_attention_interval == 0` layer is full-attention;
    /// others are GDN. `None` for pure-dense arches.
    pub full_attention_interval: Option<usize>,

    /// GDN dimensions, `Some` iff `family == Hybrid`.
    pub gdn: Option<GdnDims>,

    /// `output.weight` absent → LM head tied to `token_embd.weight`.
    pub tied_lm_head: bool,
}

impl Qwen3MoEConfig {
    /// `true` iff this is the dense-hybrid arch (`qwen35`) — no routed MoE
    /// experts, only a single dense FFN per layer. Used by the loader +
    /// forward path to pick the dense-FFN code path (V2.2).
    pub fn is_dense_ffn(&self) -> bool {
        self.num_experts == 0
    }

    /// Parse config from an opened GGUF file. Fails with a typed error if
    /// required keys are missing or the architecture tag isn't supported.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, Qwen3MoEConfigError> {
        let arch = file
            .architecture()
            .ok_or_else(|| Qwen3MoEConfigError::MissingKey("general.architecture".into()))?
            .to_string();
        if !SUPPORTED_ARCHS.contains(&arch.as_str()) {
            return Err(Qwen3MoEConfigError::WrongArchitecture { got: Some(arch) });
        }

        // V1: `qwen35moe` / `qwen36moe` / `qwen35` — all are Hybrid (GDN + full-attn)
        // or Hybrid-degenerate (full-attn-only when full_attention_interval=1, which is
        // how `qwen35` dense looks at the family level). Pure-MoE `qwen3moe` was
        // dropped — see `project_v1_bench_matrix.md` (Coder-30B was 0.27× combined
        // and the qwen3moe-specific forward_dense_attn_* path lacked the kernel
        // optimizations qwen35moe got).
        let family = AttentionFamily::Hybrid;
        let is_dense_ffn = arch == "qwen35";

        let key = |suffix: &str| format!("{arch}.{suffix}");
        let mk_missing = |suffix: &str| Qwen3MoEConfigError::MissingKey(key(suffix));
        let req_u32 = |suffix: &str| {
            file.metadata_u32(&key(suffix))
                .ok_or_else(|| mk_missing(suffix))
                .map(|v| v as usize)
        };
        let req_f32 = |suffix: &str| {
            file.metadata_f32(&key(suffix))
                .ok_or_else(|| mk_missing(suffix))
        };
        let opt_u32 = |suffix: &str| file.metadata_u32(&key(suffix)).map(|v| v as usize);
        let opt_f32 = |suffix: &str| file.metadata_f32(&key(suffix));

        let hidden_size = req_u32("embedding_length")?;
        let num_heads = req_u32("attention.head_count")?;
        let num_kv_heads = req_u32("attention.head_count_kv")?;
        let head_dim = opt_u32("attention.key_length").unwrap_or(hidden_size / num_heads);
        // **OOM-cap** — `FLAMBEAU_MAX_CTX` clamps the model's declared
        // context window for the purpose of KV preallocation. Modern
        // models ship 128k–256k native context; the per-layer KV scales
        // linearly with this and 4× MI50 cards run out of VRAM even on
        // 9B-class models when the full window is preallocated. Clients
        // that send shorter prompts (the typical case) get the same
        // behaviour at a fraction of the VRAM. Set to 0 or unset to use
        // the model's native value.
        let model_ctx = req_u32("context_length")?;
        let context_length = match std::env::var("FLAMBEAU_MAX_CTX")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|n| *n > 0)
        {
            Some(cap) if cap < model_ctx => {
                tracing::info!(
                    target: "flambeau_qwen3_moe::config",
                    model_ctx,
                    cap,
                    "FLAMBEAU_MAX_CTX clamping context_length below model native"
                );
                cap
            }
            _ => model_ctx,
        };
        let num_layers = req_u32("block_count")?;
        let rms_norm_eps = req_f32("attention.layer_norm_rms_epsilon")?;

        let rope_freq_base = opt_f32("rope.freq_base").unwrap_or(10_000.0);
        let rope_rotated = opt_u32("rope.dimension_count").unwrap_or(head_dim);
        let rope_sections = read_rope_sections(file, &key("rope.dimension_sections"))?;
        let rope = RopeSpec {
            freq_base: rope_freq_base,
            rotated_dims: rope_rotated,
            sections: rope_sections,
        };

        // Dense-hybrid (`qwen35`) has no MoE metadata. Default to 0/1 so the
        // config loads; downstream loader will check `num_experts==0` and
        // branch to the dense-FFN path (V2.2 follow-up).
        let (num_experts, num_experts_per_tok, moe_intermediate_size) = if is_dense_ffn {
            (
                0,
                1,
                opt_u32("feed_forward_length").unwrap_or(0),
            )
        } else {
            (
                req_u32("expert_count")?,
                req_u32("expert_used_count")?,
                req_u32("expert_feed_forward_length")?,
            )
        };
        let shared_expert_intermediate_size = opt_u32("expert_shared_feed_forward_length")
            .filter(|&v| v > 0);

        let full_attention_interval = opt_u32("full_attention_interval").filter(|&v| v > 0);

        let gdn = if family == AttentionFamily::Hybrid {
            Some(GdnDims {
                d_inner: req_u32("ssm.inner_size")?,
                head_k_dim: req_u32("ssm.state_size")?,
                num_k_heads: req_u32("ssm.group_count")?,
                num_v_heads: req_u32("ssm.time_step_rank")?,
                conv_kernel: req_u32("ssm.conv_kernel")?,
            })
        } else {
            None
        };

        let vocab_size = file
            .info("token_embd.weight")
            .ok()
            .and_then(|ti| ti.dims.first().copied())
            .map(|v| v as usize)
            .ok_or_else(|| {
                Qwen3MoEConfigError::MissingKey("tensor token_embd.weight".into())
            })?;
        let tied_lm_head = file.info("output.weight").is_err();

        Ok(Self {
            arch,
            family,
            hidden_size,
            vocab_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            context_length,
            rms_norm_eps,
            rope,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            full_attention_interval,
            gdn,
            tied_lm_head,
        })
    }

    /// Group size for GQA — how many Q heads share one KV head.
    pub fn num_kv_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Q projection output dim. For `qwen35moe` full-attention layers this
    /// is the un-gated Q half only — the gate half is separate.
    pub fn q_proj_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// KV projection output dim (same for K and V).
    pub fn kv_proj_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Is layer `il` a recurrent (GDN) layer? Dense arches always return `false`.
    ///
    /// Matches candle's convention: `(il + 1) % full_attention_interval != 0`
    /// → recurrent. Full-attention layers fall on the interval boundary.
    pub fn is_recurrent(&self, il: usize) -> bool {
        match self.full_attention_interval {
            Some(n) if n > 0 => (il + 1) % n != 0,
            _ => false,
        }
    }

    /// Count of GDN layers in the model.
    pub fn num_recurrent_layers(&self) -> usize {
        (0..self.num_layers).filter(|&il| self.is_recurrent(il)).count()
    }

    /// Count of full-attention layers. Equals `num_layers` on dense arches.
    pub fn num_full_attn_layers(&self) -> usize {
        self.num_layers - self.num_recurrent_layers()
    }
}

fn read_rope_sections(
    file: &GgufFile,
    key: &str,
) -> Result<Option<[u32; 4]>, Qwen3MoEConfigError> {
    let Some(v) = file.metadata.get(key) else {
        return Ok(None);
    };
    let Some(arr) = v.as_array() else {
        return Err(Qwen3MoEConfigError::BadType(key.to_string()));
    };
    if arr.len() != 4 {
        return Err(Qwen3MoEConfigError::BadType(key.to_string()));
    }
    let mut out = [0u32; 4];
    for (i, e) in arr.iter().enumerate() {
        out[i] = e
            .as_u32()
            .ok_or_else(|| Qwen3MoEConfigError::BadType(key.to_string()))?;
    }
    Ok(Some(out))
}
