//! Gemma 4 config parsed from GGUF metadata.
//!
//! Source of truth for keys: `/artefact/llama.cpp/src/llama-model.cpp:1629`
//! (`case LLM_ARCH_GEMMA4`). See `certs/research/gemma4_recon.md` for the
//! per-variant audit of the 5 local GGUFs.

use flambeau_quant::{GgufFile, Value};

/// Architecture tag this crate parses.
pub const SUPPORTED_ARCHS: &[&str] = &["gemma4"];

#[derive(Debug, thiserror::Error)]
pub enum Gemma4ConfigError {
    #[error("expected Gemma 4 architecture (gemma4), got {got:?}")]
    WrongArchitecture { got: Option<String> },
    #[error("missing GGUF metadata key `{0}`")]
    MissingKey(String),
    #[error("metadata key `{0}` had unexpected type")]
    BadType(String),
    #[error("unsupported gemma4 variant: n_layer={n_layer}")]
    UnknownVariant { n_layer: usize },
}

/// Gemma 4 size variants, identified by `block_count` per llama.cpp's
/// `switch (hparams.n_layer)` at model.cpp:1649.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4Variant {
    /// 26B-A4B MoE — 30 layers, 128 experts / 8 active.
    Moe26BA4B,
    /// E2B edge dense — 35 layers, per-layer side-channel embedding.
    E2B,
    /// E4B edge dense — 42 layers, per-layer side-channel embedding,
    /// shared-KV tail of `shared_kv_layers` layers.
    E4B,
    /// 31B dense — 60 layers.
    Dense31B,
}

impl Gemma4Variant {
    pub fn from_n_layer(n_layer: usize) -> Option<Self> {
        match n_layer {
            30 => Some(Self::Moe26BA4B),
            35 => Some(Self::E2B),
            42 => Some(Self::E4B),
            60 => Some(Self::Dense31B),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Moe26BA4B => "26B-A4B",
            Self::E2B => "E2B",
            Self::E4B => "E4B",
            Self::Dense31B => "31B",
        }
    }

    pub fn is_moe(self) -> bool {
        matches!(self, Self::Moe26BA4B)
    }
}

/// Routed-expert dims, present iff [`Gemma4Variant::is_moe`].
#[derive(Debug, Clone, Copy)]
pub struct MoeDims {
    /// `gemma4.expert_count`.
    pub num_experts: usize,
    /// `gemma4.expert_used_count`.
    pub num_experts_per_tok: usize,
    /// `gemma4.expert_feed_forward_length` (= `n_ff_exp`).
    pub moe_intermediate_size: usize,
}

/// Per-layer side-channel embedding, present iff
/// `gemma4.embedding_length_per_layer_input > 0` (E2B + E4B only).
#[derive(Debug, Clone, Copy)]
pub struct PerLayerEmbed {
    /// `gemma4.embedding_length_per_layer_input` (per-layer projection width).
    pub n_embd_per_layer: usize,
}

/// Every number the forward pass needs, pulled once at load time.
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub arch: String,
    pub variant: Gemma4Variant,

    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    /// Per-layer `n_kv_heads`. Non-uniform on 26B-A4B and 31B (SWA
    /// layers carry more KV heads with a smaller per-head dim than
    /// full-attention layers).
    pub num_kv_heads: Vec<usize>,
    /// Per-head k length for full-attention layers
    /// (`gemma4.attention.key_length`, fed directly into
    /// `n_embd_head_k` per llama.cpp `model.cpp:832`).
    pub head_dim: usize,
    /// Per-head k length for SWA layers
    /// (`gemma4.attention.key_length_swa`).
    pub head_dim_swa: usize,
    pub context_length: usize,
    pub rms_norm_eps: f32,

    /// Dense FFN intermediate dim (`gemma4.feed_forward_length`). For
    /// MoE variants this is the **shared MLP** width (every MoE layer
    /// has a shared MLP in parallel with the routed experts).
    pub feed_forward_length: usize,

    /// RoPE frequency base for full-attention layers
    /// (`gemma4.rope.freq_base`). Distinct from `rope_freq_base_swa`.
    pub rope_freq_base: f32,
    /// RoPE frequency base for SWA layers (`gemma4.rope.freq_base_swa`).
    pub rope_freq_base_swa: f32,
    /// Rotated dim count for full-attn layers
    /// (`gemma4.rope.dimension_count`).
    pub rope_dim: usize,
    /// Rotated dim count for SWA layers
    /// (`gemma4.rope.dimension_count_swa`).
    pub rope_dim_swa: usize,

    /// Per-layer iSWA flag — `swa_layers[il] = true` iff layer `il` is
    /// a sliding-window-attention layer. Read from
    /// `gemma4.attention.sliding_window_pattern` (per-layer bool array).
    pub swa_layers: Vec<bool>,
    /// SWA radius (`gemma4.attention.sliding_window`). Only meaningful
    /// when at least one entry of `swa_layers` is `true`.
    pub sliding_window: usize,

    /// Count of "shared-KV tail" layers. The last
    /// `shared_kv_layers` layers do NOT own a KV cache; they read from
    /// an earlier layer's KV. Mirrors llama.cpp's
    /// `n_layer_kv_from_start = n_layer - shared_kv_layers`. `0` on
    /// 26B-A4B / 31B; `>0` on E4B (and presumably E2B).
    pub shared_kv_layers: usize,

    pub moe: Option<MoeDims>,
    pub per_layer_embed: Option<PerLayerEmbed>,

    /// LM-head softcap (`gemma4.final_logit_softcapping`). `0.0` ⇒
    /// disabled; gemma4 uses `30.0` across all 5 audited files.
    pub final_logit_softcap: f32,

    /// `output.weight` absent → LM head tied to `token_embd.weight`.
    /// All 5 audited gemma4 GGUFs are tied.
    pub tied_lm_head: bool,
}

impl Gemma4Config {
    pub fn from_gguf(file: &GgufFile) -> Result<Self, Gemma4ConfigError> {
        let arch = file
            .architecture()
            .ok_or_else(|| Gemma4ConfigError::MissingKey("general.architecture".into()))?
            .to_string();
        if !SUPPORTED_ARCHS.contains(&arch.as_str()) {
            return Err(Gemma4ConfigError::WrongArchitecture { got: Some(arch) });
        }

        let key = |suffix: &str| format!("{arch}.{suffix}");
        let mk_missing = |suffix: &str| Gemma4ConfigError::MissingKey(key(suffix));
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
        let num_layers = req_u32("block_count")?;
        let context_length = req_u32("context_length")?;
        let rms_norm_eps = req_f32("attention.layer_norm_rms_epsilon")?;

        // `attention.head_count_kv` is stored as a per-layer array
        // (length n_layer). Uniform in our 5 files but the loader
        // honours the array.
        let num_kv_heads = read_usize_array_or_scalar(
            file,
            &key("attention.head_count_kv"),
            num_layers,
        )?;

        // `key_length` and `key_length_swa` are PER-HEAD (= n_embd_head_k_full /
        // n_embd_head_k_swa per llama.cpp `model.cpp:832`), not totals. SWA
        // layers commonly carry more KV heads at a smaller per-head dim;
        // total per-layer K width = n_kv_heads[il] * head_dim_for_layer(il).
        let head_dim = req_u32("attention.key_length")?;
        let head_dim_swa = req_u32("attention.key_length_swa")?;

        let feed_forward_length = req_u32("feed_forward_length")?;

        let rope_freq_base = opt_f32("rope.freq_base").unwrap_or(10_000.0);
        let rope_freq_base_swa = opt_f32("rope.freq_base_swa").unwrap_or(rope_freq_base);
        let rope_dim = opt_u32("rope.dimension_count").unwrap_or(head_dim);
        let rope_dim_swa = opt_u32("rope.dimension_count_swa").unwrap_or(head_dim_swa);

        let swa_layers = read_bool_array_or_scalar(
            file,
            &key("attention.sliding_window_pattern"),
            num_layers,
        )?;
        let sliding_window = req_u32("attention.sliding_window")?;
        let shared_kv_layers = opt_u32("attention.shared_kv_layers").unwrap_or(0);

        let moe = if let Some(num_experts) = opt_u32("expert_count") {
            Some(MoeDims {
                num_experts,
                num_experts_per_tok: req_u32("expert_used_count")?,
                moe_intermediate_size: req_u32("expert_feed_forward_length")?,
            })
        } else {
            None
        };

        let n_embd_per_layer = opt_u32("embedding_length_per_layer_input").unwrap_or(0);
        let per_layer_embed = if n_embd_per_layer > 0 {
            Some(PerLayerEmbed { n_embd_per_layer })
        } else {
            None
        };

        let final_logit_softcap = opt_f32("final_logit_softcapping").unwrap_or(0.0);

        let variant = Gemma4Variant::from_n_layer(num_layers)
            .ok_or(Gemma4ConfigError::UnknownVariant { n_layer: num_layers })?;
        // MoE/dense flag from variant must match metadata presence.
        if variant.is_moe() != moe.is_some() {
            return Err(Gemma4ConfigError::BadType(key("expert_count")));
        }

        let vocab_size = file
            .info("token_embd.weight")
            .ok()
            .and_then(|ti| ti.dims.first().copied())
            .map(|v| v as usize)
            .ok_or_else(|| {
                Gemma4ConfigError::MissingKey("tensor token_embd.weight".into())
            })?;
        let tied_lm_head = file.info("output.weight").is_err();

        Ok(Self {
            arch,
            variant,
            hidden_size,
            vocab_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim_swa,
            context_length,
            rms_norm_eps,
            feed_forward_length,
            rope_freq_base,
            rope_freq_base_swa,
            rope_dim,
            rope_dim_swa,
            swa_layers,
            sliding_window,
            shared_kv_layers,
            moe,
            per_layer_embed,
            final_logit_softcap,
            tied_lm_head,
        })
    }

    /// `true` iff layer `il` is a sliding-window-attention layer.
    pub fn is_swa(&self, il: usize) -> bool {
        self.swa_layers.get(il).copied().unwrap_or(false)
    }

    /// `true` iff layer `il` owns its own KV cache. Tail layers in the
    /// shared-KV range read from an earlier layer instead.
    pub fn has_kv(&self, il: usize) -> bool {
        let kv_from_start = self.num_layers.saturating_sub(self.shared_kv_layers);
        il < kv_from_start
    }

    /// Per-layer KV head count.
    pub fn n_kv_heads(&self, il: usize) -> usize {
        self.num_kv_heads[il]
    }

    /// Per-layer head_dim — `head_dim_swa` when the layer is SWA,
    /// `head_dim` otherwise.
    pub fn head_dim_for_layer(&self, il: usize) -> usize {
        if self.is_swa(il) { self.head_dim_swa } else { self.head_dim }
    }

    /// Per-layer rotated-dim count for RoPE.
    pub fn rope_dim_for_layer(&self, il: usize) -> usize {
        if self.is_swa(il) { self.rope_dim_swa } else { self.rope_dim }
    }

    /// Per-layer RoPE frequency base.
    pub fn rope_freq_base_for_layer(&self, il: usize) -> f32 {
        if self.is_swa(il) { self.rope_freq_base_swa } else { self.rope_freq_base }
    }

    /// Count of full-attention layers (used for `rope_freqs` tensor
    /// allocation in llama.cpp; flambeau loader can ignore beyond
    /// validating the per-layer flag).
    pub fn num_full_attn_layers(&self) -> usize {
        self.swa_layers.iter().filter(|b| !**b).count()
    }
}

#[cfg(feature = "hip")]
impl flambeau_blocks::ModelConfig for Gemma4Config {
    fn hidden(&self) -> usize { self.hidden_size }
    fn ff_len(&self) -> usize { self.feed_forward_length }
    fn n_heads(&self, _layer: usize) -> usize { self.num_heads }
    fn n_kv_heads(&self, layer: usize) -> usize { self.num_kv_heads[layer] }
    fn head_dim(&self, layer: usize) -> usize { self.head_dim_for_layer(layer) }
    fn rms_norm_eps(&self) -> f32 { self.rms_norm_eps }
    fn vocab_size(&self) -> usize { self.vocab_size }
}

/// Read a metadata key as a length-`n` array of `usize`. If the key
/// stores a scalar, broadcast it.
fn read_usize_array_or_scalar(
    file: &GgufFile,
    key: &str,
    n: usize,
) -> Result<Vec<usize>, Gemma4ConfigError> {
    let Some(v) = file.metadata.get(key) else {
        return Err(Gemma4ConfigError::MissingKey(key.to_string()));
    };
    if let Some(arr) = v.as_array() {
        if arr.len() != n {
            return Err(Gemma4ConfigError::BadType(key.to_string()));
        }
        let mut out = Vec::with_capacity(n);
        for e in arr {
            let u = e
                .as_u32()
                .ok_or_else(|| Gemma4ConfigError::BadType(key.to_string()))?;
            out.push(u as usize);
        }
        Ok(out)
    } else if let Some(u) = v.as_u32() {
        Ok(vec![u as usize; n])
    } else {
        Err(Gemma4ConfigError::BadType(key.to_string()))
    }
}

/// Read a metadata key as a length-`n` array of `bool`. If the key
/// stores a scalar bool, broadcast it.
fn read_bool_array_or_scalar(
    file: &GgufFile,
    key: &str,
    n: usize,
) -> Result<Vec<bool>, Gemma4ConfigError> {
    let Some(v) = file.metadata.get(key) else {
        return Err(Gemma4ConfigError::MissingKey(key.to_string()));
    };
    if let Some(arr) = v.as_array() {
        if arr.len() != n {
            return Err(Gemma4ConfigError::BadType(key.to_string()));
        }
        let mut out = Vec::with_capacity(n);
        for e in arr {
            let b = match e {
                Value::Bool(b) => *b,
                _ => return Err(Gemma4ConfigError::BadType(key.to_string())),
            };
            out.push(b);
        }
        Ok(out)
    } else if let Value::Bool(b) = v {
        Ok(vec![*b; n])
    } else {
        Err(Gemma4ConfigError::BadType(key.to_string()))
    }
}

