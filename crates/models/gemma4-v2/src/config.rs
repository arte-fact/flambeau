use flambeau_forward::KvLayerShape;
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
}

/// Per-layer side-channel embedding dims (gemma 4n / E2B / E4B). When
/// present, each layer mixes a `[pe]`-wide F32 vector into the
/// residual stream after the FFN residual add. The vector is
/// precomputed once per token from `per_layer_token_embd` +
/// `per_layer_model_proj` (see
/// `flambeau_forward::per_layer_embd::build_inp_per_layer_table`).
#[derive(Debug, Clone, Copy)]
pub struct PerLayerEmbdDims {
    /// Per-layer side-channel width (256 on E4B).
    pub pe: usize,
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
    /// Per-layer KV-share source — `Some(src)` means this layer
    /// reuses layer `src`'s KV cache slot (gemma 4n's
    /// `shared_kv_layers`). `None` for own-KV layers. Resolver
    /// matches the most recent has_kv layer of the same attention
    /// type (SWA vs full), mirroring llama.cpp's
    /// `build_attn_inp_kv_iswa` slot match.
    pub kv_share_src: Vec<Option<usize>>,
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
    /// Present iff `gemma4.embedding_length_per_layer_input > 0` (E2B,
    /// E4B). When set, every layer applies a per-layer side-channel
    /// embedding block after the FFN residual add.
    pub per_layer_embd: Option<PerLayerEmbdDims>,
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

        let per_layer_embd = match opt_u32("embedding_length_per_layer_input") {
            Some(n) if n > 0 => Some(PerLayerEmbdDims { pe: n }),
            _ => None,
        };
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

        let swa_layers =
            read_bool_array_or_scalar(file, &key("attention.sliding_window_pattern"), num_layers)?;
        let num_kv_heads =
            read_usize_array_or_scalar(file, &key("attention.head_count_kv"), num_layers)?;

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

        // gemma 4n shared-KV: the trailing `shared_kv_layers` layers
        // reuse an earlier layer's K/V cache slot. Resolver matches
        // each shared layer to the most recent has_kv layer of the
        // same attention type (SWA vs full), mirroring llama.cpp's
        // `build_attn_inp_kv_iswa`. Default 0 ⇒ every layer owns KV.
        let shared_kv_layers = opt_u32("attention.shared_kv_layers").unwrap_or(0);
        let n_kv_from_start = num_layers.saturating_sub(shared_kv_layers);
        let mut kv_share_src: Vec<Option<usize>> = vec![None; num_layers];
        let mut last_swa_with_kv: Option<usize> = None;
        let mut last_full_with_kv: Option<usize> = None;
        for li in 0..num_layers {
            let is_swa = attn[li].window_size > 0;
            if li < n_kv_from_start {
                if is_swa {
                    last_swa_with_kv = Some(li);
                } else {
                    last_full_with_kv = Some(li);
                }
            } else {
                kv_share_src[li] = if is_swa {
                    last_swa_with_kv
                } else {
                    last_full_with_kv
                };
            }
        }

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
            kv_share_src,
            rms_eps,
            context_length,
            vocab_size,
            final_logit_softcap,
            tied_lm_head,
            moe,
            per_layer_embd,
        })
    }
}

impl Gemma4V2Config {
    /// Minimum absolute layer index referenced by any `kv_share_src`
    /// entry, or `None` when no layer shares KV. The server uses this
    /// as the PP-split boundary: the last PP rank must own
    /// `[boundary..num_layers]` so every shared layer and its source
    /// live on the same rank (cross-rank KV sharing is not wired —
    /// `standard_attn` would `bail!` on a source not in its local
    /// pool).
    pub fn pp_kv_share_boundary(&self) -> Option<usize> {
        self.kv_share_src.iter().filter_map(|o| *o).min()
    }
}

impl KvLayerShape for Gemma4V2Config {
    fn num_layers(&self) -> usize {
        self.num_layers
    }
    fn kv_width_at(&self, li: usize, n_ranks: usize) -> usize {
        (self.num_kv_heads[li] / n_ranks) * self.attn[li].head_dim
    }
    fn head_dim_at(&self, li: usize) -> usize {
        self.attn[li].head_dim
    }
    fn window_size_at(&self, li: usize) -> i32 {
        self.attn[li].window_size
    }
    fn kv_depth_at(&self, li: usize, max_seq_len: usize, prefill_ubatch: usize) -> usize {
        // SWA layers ring-buffer at window + one prefill batch (the live
        // span one forward can touch); global layers stay full-context.
        // Clamped to max_seq_len so a small ctx cap never inflates the slab.
        let window = self.attn[li].window_size;
        if window > 0 {
            (window as usize + prefill_ubatch).min(max_seq_len)
        } else {
            max_seq_len
        }
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
                _ => Err(Gemma4V2ConfigError::MissingKey(format!(
                    "{key} (bad element type)"
                ))),
            })
            .collect()
    } else if let Value::Bool(b) = v {
        Ok(vec![*b; n])
    } else {
        Err(Gemma4V2ConfigError::MissingKey(format!(
            "{key} (not bool/array)"
        )))
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
