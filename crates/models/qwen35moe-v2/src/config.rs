use flambeau_forward::ctx::GdnDims;
use flambeau_quant::GgufFile;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Qwen35MoeV2ConfigError {
    #[error("missing required GGUF key: {0}")]
    MissingKey(String),
    #[error("expected `qwen35moe` architecture, got {got:?}")]
    WrongArchitecture { got: Option<String> },
    #[error("token_embd.weight tensor missing — cannot infer vocab size")]
    MissingTokenEmbd,
}

#[derive(Debug, Clone)]
pub struct Qwen35MoeV2Config {
    pub hidden: usize,
    pub num_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rotated_dims: usize,
    pub rope_theta: f32,
    pub vocab_size: usize,
    pub rms_eps: f32,
    pub context_length: usize,
    pub tied_lm_head: bool,
    pub gdn: GdnDims,
    pub full_attention_interval: usize,
    pub num_experts: usize,
    pub experts_per_tok: usize,
    pub expert_intermediate: usize,
    pub shared_expert_intermediate: usize,
}

impl Qwen35MoeV2Config {
    pub fn from_gguf(file: &GgufFile) -> Result<Self, Qwen35MoeV2ConfigError> {
        let arch = file.architecture().map(|s| s.to_string());
        if arch.as_deref() != Some("qwen35moe") {
            return Err(Qwen35MoeV2ConfigError::WrongArchitecture { got: arch });
        }
        let key = |suffix: &str| format!("qwen35moe.{suffix}");
        let mk_missing = |suffix: &str| Qwen35MoeV2ConfigError::MissingKey(key(suffix));
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
        let rms_eps = req_f32("attention.layer_norm_rms_epsilon")?;
        let context_length = req_u32("context_length")?;
        let rope_theta = opt_f32("rope.freq_base").unwrap_or(10_000.0);
        let rotated_dims = opt_u32("rope.dimension_count").unwrap_or(head_dim);
        let full_attention_interval = opt_u32("full_attention_interval")
            .filter(|&v| v > 0)
            .unwrap_or(1);

        let num_experts = req_u32("expert_count")?;
        let experts_per_tok = req_u32("expert_used_count")?;
        let expert_intermediate = req_u32("expert_feed_forward_length")?;
        let shared_expert_intermediate = opt_u32("expert_shared_feed_forward_length").unwrap_or(0);

        let d_inner = req_u32("ssm.inner_size")?;
        let num_v_heads = req_u32("ssm.time_step_rank")?;
        let num_k_heads = req_u32("ssm.group_count")?;
        let head_k_dim = req_u32("ssm.state_size")?;
        let conv_kernel = req_u32("ssm.conv_kernel")?;
        let head_v_dim = d_inner / num_v_heads;
        let conv_channels = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;

        let vocab_size = file
            .info("token_embd.weight")
            .ok()
            .and_then(|ti| ti.dims.first().copied())
            .map(|v| v as usize)
            .ok_or(Qwen35MoeV2ConfigError::MissingTokenEmbd)?;
        let tied_lm_head = file.info("output.weight").is_err();

        Ok(Self {
            hidden,
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
            gdn: GdnDims {
                d_inner,
                num_v_heads,
                num_k_heads,
                head_k_dim,
                head_v_dim,
                conv_channels,
                conv_kernel,
            },
            full_attention_interval,
            num_experts,
            experts_per_tok,
            expert_intermediate,
            shared_expert_intermediate,
        })
    }

    /// `(il + 1) % full_attention_interval != 0` ⇒ GDN, else full-attn.
    pub fn is_recurrent(&self, il: usize) -> bool {
        (il + 1) % self.full_attention_interval != 0
    }
}
