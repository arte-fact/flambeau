//! Weight-name map — single source of truth for every GGUF tensor the
//! Gemma 4 family touches. Source: `certs/research/gemma4_recon.md`
//! (5-file audit) cross-checked against llama.cpp
//! `src/llama-model.cpp:4552`.
//!
//! Three families of layers:
//! - **AttnSwa** / **AttnFull** — same attention surface, differing
//!   per-layer head dims and rope base. SWA and full-attn layers carry
//!   the same tensor names; the distinction is configured per-layer at
//!   forward time.
//! - **DenseFfn** — gate / up / down triple (E2B, E4B, 31B).
//! - **MoeFfn** — shared MLP + routed-experts (26B-A4B). Shared MLP and
//!   routed branch run in parallel and are summed.
//!
//! Plus per-layer side-channel embedding (E2B + E4B only):
//! `inp_gate`, `proj`, `post_norm` per layer; `per_layer_token_embd`,
//! `per_layer_model_proj`, `per_layer_proj_norm` globally.

/// Global tensor names (not per-layer).
#[derive(Debug, Clone)]
pub struct GlobalNames {
    pub token_embd: String,
    pub output_norm: String,
    pub output: String,
    /// Present in every gemma4 GGUF — full-attn layers' learned RoPE
    /// freq correction factors. `TENSOR_DUPLICATED` past layer 0.
    pub rope_freqs: String,
    /// Per-layer side-channel embedding table (E2B/E4B only).
    pub per_layer_token_embd: String,
    /// Per-layer side-channel input projection (E2B/E4B only).
    pub per_layer_model_proj: String,
    /// Per-layer side-channel projection norm (E2B/E4B only).
    pub per_layer_proj_norm: String,
}

impl GlobalNames {
    pub fn default_names() -> Self {
        Self {
            token_embd: "token_embd.weight".to_string(),
            output_norm: "output_norm.weight".to_string(),
            output: "output.weight".to_string(),
            rope_freqs: "rope_freqs.weight".to_string(),
            per_layer_token_embd: "per_layer_token_embd.weight".to_string(),
            per_layer_model_proj: "per_layer_model_proj.weight".to_string(),
            per_layer_proj_norm: "per_layer_proj_norm.weight".to_string(),
        }
    }
}

/// Shared attention names — same set for SWA and full-attn layers. The
/// `attn_k`, `attn_v`, `attn_k_norm` tensors are *optional* on the
/// shared-KV tail (`has_kv(il) == false`); loaders treat them as
/// `TENSOR_NOT_REQUIRED` mirroring llama.cpp PR #21739.
#[derive(Debug, Clone)]
pub struct AttnNames {
    pub attn_norm: String,
    pub attn_q: String,
    pub attn_k: String,
    pub attn_v: String,
    pub attn_output: String,
    pub attn_q_norm: String,
    pub attn_k_norm: String,
    /// Post-attention RMSNorm (Gemma's "double-norm" — distinct from
    /// the per-layer side-channel `post_norm`).
    pub post_attention_norm: String,
    /// Optional per-layer learned scalar applied to the layer's output.
    pub layer_output_scale: String,
}

/// Dense FFN names — used by E2B, E4B, 31B (every layer) AND by the
/// MoE 26B-A4B variant for the shared-MLP branch present in every MoE
/// layer.
#[derive(Debug, Clone)]
pub struct DenseFfnNames {
    pub ffn_norm: String,
    pub ffn_gate: String,
    pub ffn_up: String,
    pub ffn_down: String,
    /// Post-FFN RMSNorm.
    pub post_ffw_norm: String,
}

/// MoE-specific names. Present only on `Gemma4Variant::Moe26BA4B`. The
/// gate+up tensor is fused on disk (`[n_experts, 2*n_ff_exp, hidden]`)
/// and split at load into `gate_exps` + `up_exps` for the existing
/// indexed-MMVQ kernels.
#[derive(Debug, Clone)]
pub struct MoeFfnNames {
    /// Router matmul weight: `[n_experts, hidden]` F32.
    pub ffn_gate_inp: String,
    /// Router pre-scale tensor: `[hidden]` F32. Multiplied element-wise
    /// into the router input before the matmul (= llama.cpp's
    /// `ffn_gate_inp_s`). GGUF naming convention: the scale lives at
    /// `<base>.scale` under the same base name as the matmul weight.
    pub ffn_gate_inp_scale: String,
    /// Fused gate+up per expert: `[n_experts, 2*n_ff_exp, hidden]`.
    /// Split at load.
    pub ffn_gate_up_exps: String,
    /// Per-expert down: `[n_experts, hidden, n_ff_exp]`.
    pub ffn_down_exps: String,
    /// Per-expert down scale: `[n_experts]` F32.
    pub ffn_down_exps_scale: String,
    /// Pre-MoE branch RMSNorm.
    pub pre_ffw_norm_2: String,
    /// Post-shared-MLP RMSNorm.
    pub post_ffw_norm_1: String,
    /// Post-MoE branch RMSNorm.
    pub post_ffw_norm_2: String,
}

/// Per-layer side-channel embedding names (E2B + E4B only).
#[derive(Debug, Clone)]
pub struct PerLayerEmbedNames {
    /// Pre-projection gate (`per_layer_inp_gate` in llama.cpp):
    /// `[per_layer_embd, hidden]` F32.
    pub inp_gate: String,
    /// Back-projection (`per_layer_proj`): `[hidden, per_layer_embd]` F32.
    pub proj: String,
    /// Side-channel post-RMSNorm (`per_layer_post_norm`).
    pub post_norm: String,
}

fn prefix(layer: usize) -> String {
    format!("blk.{layer}")
}

impl AttnNames {
    pub fn for_layer(layer: usize) -> Self {
        let p = prefix(layer);
        Self {
            attn_norm: format!("{p}.attn_norm.weight"),
            attn_q: format!("{p}.attn_q.weight"),
            attn_k: format!("{p}.attn_k.weight"),
            attn_v: format!("{p}.attn_v.weight"),
            attn_output: format!("{p}.attn_output.weight"),
            attn_q_norm: format!("{p}.attn_q_norm.weight"),
            attn_k_norm: format!("{p}.attn_k_norm.weight"),
            post_attention_norm: format!("{p}.post_attention_norm.weight"),
            layer_output_scale: format!("{p}.layer_output_scale.weight"),
        }
    }
}

impl DenseFfnNames {
    pub fn for_layer(layer: usize) -> Self {
        let p = prefix(layer);
        Self {
            ffn_norm: format!("{p}.ffn_norm.weight"),
            ffn_gate: format!("{p}.ffn_gate.weight"),
            ffn_up: format!("{p}.ffn_up.weight"),
            ffn_down: format!("{p}.ffn_down.weight"),
            post_ffw_norm: format!("{p}.post_ffw_norm.weight"),
        }
    }
}

impl MoeFfnNames {
    pub fn for_layer(layer: usize) -> Self {
        let p = prefix(layer);
        Self {
            ffn_gate_inp: format!("{p}.ffn_gate_inp.weight"),
            ffn_gate_inp_scale: format!("{p}.ffn_gate_inp.scale"),
            ffn_gate_up_exps: format!("{p}.ffn_gate_up_exps.weight"),
            ffn_down_exps: format!("{p}.ffn_down_exps.weight"),
            ffn_down_exps_scale: format!("{p}.ffn_down_exps.scale"),
            pre_ffw_norm_2: format!("{p}.pre_ffw_norm_2.weight"),
            post_ffw_norm_1: format!("{p}.post_ffw_norm_1.weight"),
            post_ffw_norm_2: format!("{p}.post_ffw_norm_2.weight"),
        }
    }
}

impl PerLayerEmbedNames {
    pub fn for_layer(layer: usize) -> Self {
        let p = prefix(layer);
        Self {
            inp_gate: format!("{p}.inp_gate.weight"),
            proj: format!("{p}.proj.weight"),
            post_norm: format!("{p}.post_norm.weight"),
        }
    }
}
