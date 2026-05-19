use flambeau_quant::{GgufFile, Value};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Gemma4V2ConfigError {
    #[error("missing required GGUF key: {0}")]
    MissingKey(String),
    #[error("expected `gemma4` architecture, got {got:?}")]
    WrongArchitecture { got: Option<String> },
    #[error("token_embd.weight tensor missing — cannot infer vocab size")]
    MissingTokenEmbd,
    #[error("gemma4-v2 does not yet support variants with per-layer embd (gemma 4n / E2B / E4B): embedding_length_per_layer_input > 0")]
    PerLayerEmbdUnsupported,
}

/// Routed-expert dims for the MoE variant (26B-A4B). Each MoE layer
/// has a shared MLP at [`Gemma4V2Config::intermediate`] running in
/// parallel with the routed experts at `moe_intermediate`.
#[derive(Debug, Clone, Copy)]
pub struct MoeDims {
    pub num_experts: usize,
    pub experts_per_tok: usize,
    pub moe_intermediate: usize,
}

/// Per-layer attention dims. SWA layers and full-attn layers can differ
/// in `head_dim` / `rotated_dims` / `rope_theta` / `window_size`.
#[derive(Debug, Clone, Copy)]
pub struct LayerAttnDims {
    pub head_dim: usize,
    pub rotated_dims: usize,
    pub rope_theta: f32,
    pub window_size: i32,
}

#[derive(Debug, Clone)]
pub struct Gemma4V2Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    /// Per-layer (gemma4 stores `head_count_kv` as an array; uniform in
    /// most files but the spec allows per-layer).
    pub num_kv_heads: Vec<usize>,
    /// Per-layer attention shape. `attn[li].window_size > 0` for SWA layers.
    pub attn: Vec<LayerAttnDims>,
    pub rms_eps: f32,
    pub context_length: usize,
    pub vocab_size: usize,
    pub final_logit_softcap: f32,
    pub tied_lm_head: bool,
    /// Present iff `gemma4.expert_count > 0` (26B-A4B). When set, every
    /// layer's FFN is the routed MoE + the shared dense MLP running in
    /// parallel; the shared MLP uses `intermediate`, the routed
    /// experts use `moe.moe_intermediate`.
    pub moe: Option<MoeDims>,
}

impl Gemma4V2Config {
    pub fn from_gguf(file: &GgufFile) -> Result<Self, Gemma4V2ConfigError> {
        let arch = file.architecture().map(|s| s.to_string());
        if arch.as_deref() != Some("gemma4") {
            return Err(Gemma4V2ConfigError::WrongArchitecture { got: arch });
        }
        let key = |s: &str| format!("gemma4.{s}");
        let mk_missing = |s: &str| Gemma4V2ConfigError::MissingKey(key(s));
        let req_u32 = |s: &str| {
            file.metadata_u32(&key(s))
                .ok_or_else(|| mk_missing(s))
                .map(|v| v as usize)
        };
        let req_f32 = |s: &str| file.metadata_f32(&key(s)).ok_or_else(|| mk_missing(s));
        let opt_u32 = |s: &str| file.metadata_u32(&key(s)).map(|v| v as usize);
        let opt_f32 = |s: &str| file.metadata_f32(&key(s));

        if opt_u32("embedding_length_per_layer_input").unwrap_or(0) > 0 {
            return Err(Gemma4V2ConfigError::PerLayerEmbdUnsupported);
        }
        let moe = match opt_u32("expert_count") {
            Some(n) if n > 0 => Some(MoeDims {
                num_experts: n,
                experts_per_tok: req_u32("expert_used_count")?,
                moe_intermediate: req_u32("expert_feed_forward_length")?,
            }),
            _ => None,
        };

        let hidden = req_u32("embedding_length")?;
        let num_heads = req_u32("attention.head_count")?;
        let num_layers = req_u32("block_count")?;
        let context_length = req_u32("context_length")?;
        let rms_eps = req_f32("attention.layer_norm_rms_epsilon")?;
        let intermediate = req_u32("feed_forward_length")?;

        let head_dim_global = req_u32("attention.key_length")?;
        let head_dim_swa = req_u32("attention.key_length_swa")?;
        let rope_theta_global = opt_f32("rope.freq_base").unwrap_or(10_000.0);
        let rope_theta_swa = opt_f32("rope.freq_base_swa").unwrap_or(rope_theta_global);
        let rotated_global = opt_u32("rope.dimension_count").unwrap_or(head_dim_global);
        let rotated_swa = opt_u32("rope.dimension_count_swa").unwrap_or(head_dim_swa);
        let sliding_window = req_u32("attention.sliding_window")? as i32;

        let swa_layers = read_bool_array_or_scalar(
            file,
            &key("attention.sliding_window_pattern"),
            num_layers,
        )?;
        let num_kv_heads = read_usize_array_or_scalar(
            file,
            &key("attention.head_count_kv"),
            num_layers,
        )?;

        let attn: Vec<LayerAttnDims> = (0..num_layers)
            .map(|li| {
                if swa_layers[li] {
                    LayerAttnDims {
                        head_dim: head_dim_swa,
                        rotated_dims: rotated_swa,
                        rope_theta: rope_theta_swa,
                        window_size: sliding_window,
                    }
                } else {
                    LayerAttnDims {
                        head_dim: head_dim_global,
                        rotated_dims: rotated_global,
                        rope_theta: rope_theta_global,
                        window_size: 0,
                    }
                }
            })
            .collect();

        let final_logit_softcap = opt_f32("final_logit_softcapping").unwrap_or(0.0);

        let vocab_size = file
            .info("token_embd.weight")
            .ok()
            .and_then(|ti| ti.dims.first().copied())
            .map(|v| v as usize)
            .ok_or(Gemma4V2ConfigError::MissingTokenEmbd)?;
        let tied_lm_head = file.info("output.weight").is_err();

        Ok(Self {
            hidden,
            intermediate,
            num_layers,
            num_heads,
            num_kv_heads,
            attn,
            rms_eps,
            context_length,
            vocab_size,
            final_logit_softcap,
            tied_lm_head,
            moe,
        })
    }
}

fn read_bool_array_or_scalar(
    file: &GgufFile,
    key: &str,
    n: usize,
) -> Result<Vec<bool>, Gemma4V2ConfigError> {
    let v = file
        .metadata
        .get(key)
        .ok_or_else(|| Gemma4V2ConfigError::MissingKey(key.to_string()))?;
    if let Some(arr) = v.as_array() {
        if arr.len() != n {
            return Err(Gemma4V2ConfigError::MissingKey(format!(
                "{key} (array len {} != {n})",
                arr.len()
            )));
        }
        arr.iter()
            .map(|e| match e {
                Value::Bool(b) => Ok(*b),
                _ => Err(Gemma4V2ConfigError::MissingKey(format!("{key} (bad element type)"))),
            })
            .collect()
    } else if let Value::Bool(b) = v {
        Ok(vec![*b; n])
    } else {
        Err(Gemma4V2ConfigError::MissingKey(format!("{key} (not bool/array)")))
    }
}

fn read_usize_array_or_scalar(
    file: &GgufFile,
    key: &str,
    n: usize,
) -> Result<Vec<usize>, Gemma4V2ConfigError> {
    let v = file
        .metadata
        .get(key)
        .ok_or_else(|| Gemma4V2ConfigError::MissingKey(key.to_string()))?;
    if let Some(arr) = v.as_array() {
        if arr.len() != n {
            return Err(Gemma4V2ConfigError::MissingKey(format!(
                "{key} (array len {} != {n})",
                arr.len()
            )));
        }
        arr.iter()
            .map(|e| match e {
                Value::U32(x) => Ok(*x as usize),
                Value::I32(x) => Ok(*x as usize),
                Value::U64(x) => Ok(*x as usize),
                Value::I64(x) => Ok(*x as usize),
                _ => Err(Gemma4V2ConfigError::MissingKey(format!(
                    "{key} (bad element type)"
                ))),
            })
            .collect()
    } else {
        match v {
            Value::U32(x) => Ok(vec![*x as usize; n]),
            Value::I32(x) => Ok(vec![*x as usize; n]),
            Value::U64(x) => Ok(vec![*x as usize; n]),
            Value::I64(x) => Ok(vec![*x as usize; n]),
            _ => Err(Gemma4V2ConfigError::MissingKey(format!(
                "{key} (not int/array)"
            ))),
        }
    }
}
