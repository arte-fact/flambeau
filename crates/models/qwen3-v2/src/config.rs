//! GGUF metadata → `Qwen3V2Config`. Mirrors the relevant subset of
//! `flambeau-qwen3-moe::config::Qwen3MoEConfig`'s parsing logic for
//! the `qwen3` architecture tag (pure dense full-attention).

use flambeau_quant::GgufFile;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Qwen3V2ConfigError {
    #[error("missing required GGUF key: {0}")]
    MissingKey(String),
    #[error("expected `qwen3` architecture, got {got:?}")]
    WrongArchitecture { got: Option<String> },
    #[error("token_embd.weight tensor missing — cannot infer vocab size")]
    MissingTokenEmbd,
}

/// Static config for a loaded qwen3 dense model.
#[derive(Debug, Clone)]
pub struct Qwen3V2Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub num_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rotated_dims: usize,
    pub rope_theta: f32,
    pub vocab_size: usize,
    pub rms_eps: f32,
    pub context_length: usize,
    /// `true` when the GGUF omits `output.weight` and the LM head must
    /// be tied to `token_embd.weight`.
    pub tied_lm_head: bool,
}

impl Qwen3V2Config {
    pub fn from_gguf(file: &GgufFile) -> Result<Self, Qwen3V2ConfigError> {
        let arch = file.architecture().map(|s| s.to_string());
        if arch.as_deref() != Some("qwen3") {
            return Err(Qwen3V2ConfigError::WrongArchitecture { got: arch });
        }
        let key = |suffix: &str| format!("qwen3.{suffix}");
        let mk_missing = |suffix: &str| Qwen3V2ConfigError::MissingKey(key(suffix));
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

        let hidden = req_u32("embedding_length")?;
        let n_heads = req_u32("attention.head_count")?;
        let n_kv_heads = req_u32("attention.head_count_kv")?;
        let head_dim = opt_u32("attention.key_length").unwrap_or(hidden / n_heads);
        let num_layers = req_u32("block_count")?;
        let intermediate = req_u32("feed_forward_length")?;
        let rms_eps = req_f32("attention.layer_norm_rms_epsilon")?;
        let context_length = req_u32("context_length")?;
        let rope_theta = opt_f32("rope.freq_base").unwrap_or(10_000.0);
        let rotated_dims = opt_u32("rope.dimension_count").unwrap_or(head_dim);

        let vocab_size = file
            .info("token_embd.weight")
            .ok()
            .and_then(|ti| ti.dims.first().copied())
            .map(|v| v as usize)
            .ok_or(Qwen3V2ConfigError::MissingTokenEmbd)?;

        let tied_lm_head = file.info("output.weight").is_err();

        Ok(Self {
            hidden,
            intermediate,
            num_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            rotated_dims,
            rope_theta,
            vocab_size,
            rms_eps,
            context_length,
            tied_lm_head,
        })
    }
}
