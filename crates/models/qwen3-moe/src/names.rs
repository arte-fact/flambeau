//! Weight-name map — single source of truth for every GGUF tensor the
//! Qwen3.x MoE family touches. Covers three layer flavours:
//!
//! - `DenseAttn` — pure-transformer self-attention with GQA (qwen3moe).
//! - `FullAttn` — Gated Full-Attention with output gate + multi-freq RoPE
//!   (qwen35moe / qwen36moe hybrid, on every Nth layer).
//! - `Gdn` — Gated-Delta-Net recurrent layer (qwen35moe hybrid, other layers).
//!
//! Each variant shares common FFN-side names (MoE + optional shared expert,
//! `attn_norm`, `post_attention_norm`).

/// Names shared across every layer flavour.
#[derive(Debug, Clone)]
pub struct CommonNames {
    /// Pre-attention RMSNorm.
    pub attn_norm: String,
    /// Post-attention / pre-FFN RMSNorm. Present on hybrid arches
    /// (qwen35moe); absent on pure dense `qwen3moe` (where `ffn_norm` is
    /// used instead — see `DenseFfnNames`).
    pub post_attention_norm: String,
    /// `ffn_norm.weight` — pre-FFN norm on dense qwen3moe. On hybrid arches
    /// the role is played by `post_attention_norm` above; `ffn_norm` may
    /// also exist but isn't load-bearing.
    pub ffn_norm: String,
}

/// Dense transformer attention block (qwen3moe family).
#[derive(Debug, Clone)]
pub struct DenseAttnNames {
    pub attn_q: String,
    pub attn_k: String,
    pub attn_v: String,
    pub attn_output: String,
    pub attn_q_norm: String,
    pub attn_k_norm: String,
    pub attn_q_bias: String,
    pub attn_k_bias: String,
    pub attn_v_bias: String,
}

/// Gated Full-Attention block (qwen35moe hybrid, every Nth layer).
/// Same as `DenseAttnNames` but with an extra `attn_gate` weight — the
/// `attn_q` tensor on disk carries both Q and the output gate concatenated
/// along the output dim, split inside the forward pass.
#[derive(Debug, Clone)]
pub struct FullAttnNames {
    pub attn_q: String,
    pub attn_k: String,
    pub attn_v: String,
    pub attn_output: String,
    pub attn_q_norm: String,
    pub attn_k_norm: String,
}

/// Gated-Delta-Net block (qwen35moe hybrid, recurrent layers).
#[derive(Debug, Clone)]
pub struct GdnNames {
    /// Fused QKV projection: hidden → `conv_channels`.
    pub attn_qkv: String,
    /// Output gate projection: hidden → `d_inner`.
    pub attn_gate: String,
    /// Alpha parameter projection (present iff the model is Qwen3.x-MoE;
    /// Qwen3-Next uses the fused `ssm_ba` instead — we'll handle that
    /// variant later).
    pub ssm_alpha: String,
    /// Beta parameter projection.
    pub ssm_beta: String,
    /// Fused ba (alternative to alpha+beta — Qwen3-Next).
    pub ssm_ba: String,
    /// Decay-rate scalar, shape `[num_v_heads]`.
    pub ssm_a: String,
    /// Time-step bias, shape `[num_v_heads]`.
    pub ssm_dt_bias: String,
    /// Causal conv1d weights.
    pub ssm_conv1d: String,
    /// Per-head-v RMSNorm weight.
    pub ssm_norm: String,
    /// Output projection: `d_inner` → hidden.
    pub ssm_out: String,
}

/// Dense FFN names — single gate/up/down triple per layer. Used by
/// `arch=qwen35` (Qwen3.5 dense-hybrid: GDN + full-attn + dense FFN).
/// For MoE arches (`qwen35moe`/`qwen36moe`) see `MoeFfnNames`.
#[derive(Debug, Clone)]
pub struct DenseFfnNames {
    pub ffn_gate: String,
    pub ffn_up: String,
    pub ffn_down: String,
}

/// MoE FFN names — always routed experts; optional shared-expert path on
/// hybrid arches.
#[derive(Debug, Clone)]
pub struct MoeFfnNames {
    /// Router logits: `[n_experts, hidden]` F32.
    pub ffn_gate_inp: String,
    /// Routed expert gate: `[n_experts, moe_intermediate, hidden]`.
    pub ffn_gate_exps: String,
    /// Routed expert up: same shape as gate.
    pub ffn_up_exps: String,
    /// Routed expert down: `[n_experts, hidden, moe_intermediate]`.
    pub ffn_down_exps: String,

    // Shared-expert names — exist iff `cfg.shared_expert_intermediate_size
    // .is_some()`; forward pass consults `CommonNames`-adjacent metadata to
    // decide whether to load them.
    /// Shared-expert gate scalar: `[hidden]` F32.
    pub ffn_gate_inp_shexp: String,
    /// Shared-expert gate: `[shared_intermediate, hidden]`.
    pub ffn_gate_shexp: String,
    pub ffn_up_shexp: String,
    pub ffn_down_shexp: String,
}

impl CommonNames {
    pub fn for_layer(layer: usize) -> Self {
        let prefix = format!("blk.{layer}");
        Self {
            attn_norm: format!("{prefix}.attn_norm.weight"),
            post_attention_norm: format!("{prefix}.post_attention_norm.weight"),
            ffn_norm: format!("{prefix}.ffn_norm.weight"),
        }
    }
}

impl DenseAttnNames {
    pub fn for_layer(layer: usize) -> Self {
        let prefix = format!("blk.{layer}");
        Self {
            attn_q: format!("{prefix}.attn_q.weight"),
            attn_k: format!("{prefix}.attn_k.weight"),
            attn_v: format!("{prefix}.attn_v.weight"),
            attn_output: format!("{prefix}.attn_output.weight"),
            attn_q_norm: format!("{prefix}.attn_q_norm.weight"),
            attn_k_norm: format!("{prefix}.attn_k_norm.weight"),
            attn_q_bias: format!("{prefix}.attn_q.bias"),
            attn_k_bias: format!("{prefix}.attn_k.bias"),
            attn_v_bias: format!("{prefix}.attn_v.bias"),
        }
    }
}

impl FullAttnNames {
    pub fn for_layer(layer: usize) -> Self {
        let prefix = format!("blk.{layer}");
        Self {
            attn_q: format!("{prefix}.attn_q.weight"),
            attn_k: format!("{prefix}.attn_k.weight"),
            attn_v: format!("{prefix}.attn_v.weight"),
            attn_output: format!("{prefix}.attn_output.weight"),
            attn_q_norm: format!("{prefix}.attn_q_norm.weight"),
            attn_k_norm: format!("{prefix}.attn_k_norm.weight"),
        }
    }
}

impl GdnNames {
    pub fn for_layer(layer: usize) -> Self {
        let prefix = format!("blk.{layer}");
        Self {
            attn_qkv: format!("{prefix}.attn_qkv.weight"),
            attn_gate: format!("{prefix}.attn_gate.weight"),
            ssm_alpha: format!("{prefix}.ssm_alpha.weight"),
            ssm_beta: format!("{prefix}.ssm_beta.weight"),
            ssm_ba: format!("{prefix}.ssm_ba.weight"),
            ssm_a: format!("{prefix}.ssm_a"),
            ssm_dt_bias: format!("{prefix}.ssm_dt.bias"),
            ssm_conv1d: format!("{prefix}.ssm_conv1d.weight"),
            ssm_norm: format!("{prefix}.ssm_norm.weight"),
            ssm_out: format!("{prefix}.ssm_out.weight"),
        }
    }
}

impl DenseFfnNames {
    pub fn for_layer(layer: usize) -> Self {
        let prefix = format!("blk.{layer}");
        Self {
            ffn_gate: format!("{prefix}.ffn_gate.weight"),
            ffn_up: format!("{prefix}.ffn_up.weight"),
            ffn_down: format!("{prefix}.ffn_down.weight"),
        }
    }
}

impl MoeFfnNames {
    pub fn for_layer(layer: usize) -> Self {
        let prefix = format!("blk.{layer}");
        Self {
            ffn_gate_inp: format!("{prefix}.ffn_gate_inp.weight"),
            ffn_gate_exps: format!("{prefix}.ffn_gate_exps.weight"),
            ffn_up_exps: format!("{prefix}.ffn_up_exps.weight"),
            ffn_down_exps: format!("{prefix}.ffn_down_exps.weight"),
            ffn_gate_inp_shexp: format!("{prefix}.ffn_gate_inp_shexp.weight"),
            ffn_gate_shexp: format!("{prefix}.ffn_gate_shexp.weight"),
            ffn_up_shexp: format!("{prefix}.ffn_up_shexp.weight"),
            ffn_down_shexp: format!("{prefix}.ffn_down_shexp.weight"),
        }
    }
}

/// Backward-compat alias — V1.7.2 code referenced `TensorNames::for_layer`
/// for the dense case. Kept so V1.7.2 tests still type-check. The new
/// hybrid code path uses the split `*Names` structs above.
pub type TensorNames = DenseAttnNames;

/// Build a flat `Vec<CommonNames>` covering every transformer block.
pub fn layer_names(num_layers: usize) -> Vec<CommonNames> {
    (0..num_layers).map(CommonNames::for_layer).collect()
}

/// Global (non-per-layer) tensor names.
#[derive(Debug)]
pub struct GlobalNames;

impl GlobalNames {
    pub const TOKEN_EMBD: &'static str = "token_embd.weight";
    pub const OUTPUT_NORM: &'static str = "output_norm.weight";
    pub const OUTPUT: &'static str = "output.weight";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_attn_names_follow_llama_cpp_convention() {
        let t = DenseAttnNames::for_layer(0);
        assert_eq!(t.attn_q, "blk.0.attn_q.weight");
        assert_eq!(t.attn_q_norm, "blk.0.attn_q_norm.weight");
    }

    #[test]
    fn gdn_names_include_ssm_tensors() {
        let g = GdnNames::for_layer(7);
        assert_eq!(g.attn_qkv, "blk.7.attn_qkv.weight");
        assert_eq!(g.attn_gate, "blk.7.attn_gate.weight");
        assert_eq!(g.ssm_conv1d, "blk.7.ssm_conv1d.weight");
        assert_eq!(g.ssm_dt_bias, "blk.7.ssm_dt.bias");
        assert_eq!(g.ssm_a, "blk.7.ssm_a");
    }

    #[test]
    fn dense_ffn_names_follow_llama_cpp_convention() {
        let d = DenseFfnNames::for_layer(5);
        assert_eq!(d.ffn_gate, "blk.5.ffn_gate.weight");
        assert_eq!(d.ffn_up, "blk.5.ffn_up.weight");
        assert_eq!(d.ffn_down, "blk.5.ffn_down.weight");
    }

    #[test]
    fn moe_names_cover_shared_expert() {
        let m = MoeFfnNames::for_layer(0);
        assert_eq!(m.ffn_gate_exps, "blk.0.ffn_gate_exps.weight");
        assert_eq!(m.ffn_gate_shexp, "blk.0.ffn_gate_shexp.weight");
        assert_eq!(m.ffn_gate_inp_shexp, "blk.0.ffn_gate_inp_shexp.weight");
    }

    #[test]
    fn layer_names_spans_all_layers() {
        let names = layer_names(48);
        assert_eq!(names.len(), 48);
        assert_eq!(names[47].attn_norm, "blk.47.attn_norm.weight");
    }
}
