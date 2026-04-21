//! Model layout — description of every tensor the model touches, paired
//! with its dtype + byte size + source tensor name.
//!
//! V1.7.2-ext: split per-layer tensor sets into the three flavours the
//! Qwen3.x MoE family ships:
//! - [`LayerAttnBlock::Dense`] — standard self-attention (qwen3moe)
//! - [`LayerAttnBlock::FullAttn`] — gated full-attention (qwen35moe every Nth)
//! - [`LayerAttnBlock::Gdn`] — Gated-Delta-Net recurrent (qwen35moe the rest)
//!
//! The FFN and norm tensors are shared across all flavours and live on
//! [`LayerDescriptor`] directly.

use anyhow::{bail, Result};
use flambeau_quant::{GgmlDType, GgufFile, TensorInfo};

use crate::config::{AttentionFamily, Qwen3MoEConfig};
use crate::names::{
    CommonNames, DenseAttnNames, FullAttnNames, GdnNames, GlobalNames, MoeFfnNames,
};

/// Tensor name + cached `TensorInfo` + byte size. Cloned from the GGUF
/// index so the model owns its own view; pointer-into-mmap reads still go
/// through `GgufFile::tensor_raw(name)` since the mmap is the source of
/// truth for payloads.
#[derive(Debug, Clone)]
pub struct ResolvedTensor {
    pub name: String,
    pub dtype: GgmlDType,
    pub dims: Vec<u64>,
    pub size_bytes: u64,
}

impl ResolvedTensor {
    fn from_info(name: String, info: &TensorInfo) -> Self {
        Self {
            name,
            dtype: info.dtype,
            dims: info.dims.clone(),
            size_bytes: info.size_in_bytes(),
        }
    }
}

/// Per-layer attention block descriptors. One of three variants depending
/// on the arch + layer index (see [`Qwen3MoEConfig::is_recurrent`]).
#[derive(Debug, Clone)]
pub enum LayerAttnBlock {
    Dense(DenseAttnTensors),
    FullAttn(FullAttnTensors),
    Gdn(GdnTensors),
}

#[derive(Debug, Clone)]
pub struct DenseAttnTensors {
    pub attn_q: ResolvedTensor,
    pub attn_k: ResolvedTensor,
    pub attn_v: ResolvedTensor,
    pub attn_output: ResolvedTensor,
    pub attn_q_norm: ResolvedTensor,
    pub attn_k_norm: ResolvedTensor,
    pub attn_q_bias: Option<ResolvedTensor>,
    pub attn_k_bias: Option<ResolvedTensor>,
    pub attn_v_bias: Option<ResolvedTensor>,
}

#[derive(Debug, Clone)]
pub struct FullAttnTensors {
    /// Concatenated [Q, output_gate] projection — shape `[2 * q_proj_dim, hidden]`.
    /// Forward splits output dim in half.
    pub attn_q: ResolvedTensor,
    pub attn_k: ResolvedTensor,
    pub attn_v: ResolvedTensor,
    pub attn_output: ResolvedTensor,
    pub attn_q_norm: ResolvedTensor,
    pub attn_k_norm: ResolvedTensor,
}

#[derive(Debug, Clone)]
pub struct GdnTensors {
    pub attn_qkv: ResolvedTensor,
    pub attn_gate: ResolvedTensor,
    /// Either (ssm_alpha + ssm_beta) or (ssm_ba) — Qwen3.6 uses the split,
    /// Qwen3-Next uses the fused form. `None` in the opposite slot.
    pub ssm_alpha: Option<ResolvedTensor>,
    pub ssm_beta: Option<ResolvedTensor>,
    pub ssm_ba: Option<ResolvedTensor>,
    pub ssm_a: ResolvedTensor,
    pub ssm_dt_bias: ResolvedTensor,
    pub ssm_conv1d: ResolvedTensor,
    pub ssm_norm: ResolvedTensor,
    pub ssm_out: ResolvedTensor,
}

/// Per-layer FFN descriptors. Every layer uses MoE — the only variation is
/// whether a shared expert is present (hybrid arches only).
#[derive(Debug, Clone)]
pub struct MoeFfnTensors {
    pub ffn_gate_inp: ResolvedTensor,
    pub ffn_gate_exps: ResolvedTensor,
    pub ffn_up_exps: ResolvedTensor,
    pub ffn_down_exps: ResolvedTensor,
    /// Present iff the arch carries an always-on shared expert. The gate
    /// scalar `ffn_gate_inp_shexp` mixes shared-expert output with routed
    /// expert output.
    pub shared: Option<SharedExpertTensors>,
}

#[derive(Debug, Clone)]
pub struct SharedExpertTensors {
    pub ffn_gate_inp_shexp: ResolvedTensor,
    pub ffn_gate_shexp: ResolvedTensor,
    pub ffn_up_shexp: ResolvedTensor,
    pub ffn_down_shexp: ResolvedTensor,
}

/// Per-block bound tensor descriptors.
#[derive(Debug, Clone)]
pub struct LayerDescriptor {
    pub layer_idx: usize,
    pub attn_norm: ResolvedTensor,
    /// Present on hybrid arches; absent on pure dense qwen3moe where
    /// [`ffn_norm`] plays the same role.
    pub post_attention_norm: Option<ResolvedTensor>,
    /// Present on dense qwen3moe; absent on hybrid arches (the role is
    /// taken by `post_attention_norm`).
    pub ffn_norm: Option<ResolvedTensor>,
    pub attn: LayerAttnBlock,
    pub ffn: MoeFfnTensors,
}

/// All tensors that land on device for a full Qwen3.x MoE forward pass.
#[derive(Debug, Clone)]
pub struct ModelLayout {
    pub token_embd: ResolvedTensor,
    pub output_norm: ResolvedTensor,
    /// `None` iff LM head is tied to `token_embd`.
    pub output: Option<ResolvedTensor>,
    pub layers: Vec<LayerDescriptor>,
}

impl ModelLayout {
    /// Walk every named tensor the config says we need. Fails if any
    /// required tensor is missing. Optional tensors (biases, tied-head,
    /// fused-vs-split GDN parameters) silently degrade to `None`.
    pub fn from_gguf(file: &GgufFile, cfg: &Qwen3MoEConfig) -> Result<Self> {
        let required = |name: &str| -> Result<ResolvedTensor> {
            match file.info(name) {
                Ok(info) => Ok(ResolvedTensor::from_info(name.to_string(), info)),
                Err(e) => bail!("required tensor `{name}` missing from GGUF: {e}"),
            }
        };
        let optional = |name: &str| -> Option<ResolvedTensor> {
            file.info(name)
                .ok()
                .map(|info| ResolvedTensor::from_info(name.to_string(), info))
        };

        let token_embd = required(GlobalNames::TOKEN_EMBD)?;
        let output_norm = required(GlobalNames::OUTPUT_NORM)?;
        let output = if cfg.tied_lm_head {
            None
        } else {
            Some(required(GlobalNames::OUTPUT)?)
        };

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for layer_idx in 0..cfg.num_layers {
            let c = CommonNames::for_layer(layer_idx);
            let m = MoeFfnNames::for_layer(layer_idx);

            let attn_norm = required(&c.attn_norm)?;
            // Hybrid arches expose `post_attention_norm`; dense `ffn_norm`.
            let (post_attention_norm, ffn_norm) = match cfg.family {
                AttentionFamily::Dense => (None, Some(required(&c.ffn_norm)?)),
                AttentionFamily::Hybrid => {
                    (Some(required(&c.post_attention_norm)?), None)
                }
            };

            let attn = match cfg.family {
                AttentionFamily::Dense => {
                    let n = DenseAttnNames::for_layer(layer_idx);
                    LayerAttnBlock::Dense(DenseAttnTensors {
                        attn_q: required(&n.attn_q)?,
                        attn_k: required(&n.attn_k)?,
                        attn_v: required(&n.attn_v)?,
                        attn_output: required(&n.attn_output)?,
                        attn_q_norm: required(&n.attn_q_norm)?,
                        attn_k_norm: required(&n.attn_k_norm)?,
                        attn_q_bias: optional(&n.attn_q_bias),
                        attn_k_bias: optional(&n.attn_k_bias),
                        attn_v_bias: optional(&n.attn_v_bias),
                    })
                }
                AttentionFamily::Hybrid if cfg.is_recurrent(layer_idx) => {
                    let n = GdnNames::for_layer(layer_idx);
                    LayerAttnBlock::Gdn(GdnTensors {
                        attn_qkv: required(&n.attn_qkv)?,
                        attn_gate: required(&n.attn_gate)?,
                        ssm_alpha: optional(&n.ssm_alpha),
                        ssm_beta: optional(&n.ssm_beta),
                        ssm_ba: optional(&n.ssm_ba),
                        ssm_a: required(&n.ssm_a)?,
                        ssm_dt_bias: required(&n.ssm_dt_bias)?,
                        ssm_conv1d: required(&n.ssm_conv1d)?,
                        ssm_norm: required(&n.ssm_norm)?,
                        ssm_out: required(&n.ssm_out)?,
                    })
                }
                AttentionFamily::Hybrid => {
                    let n = FullAttnNames::for_layer(layer_idx);
                    LayerAttnBlock::FullAttn(FullAttnTensors {
                        attn_q: required(&n.attn_q)?,
                        attn_k: required(&n.attn_k)?,
                        attn_v: required(&n.attn_v)?,
                        attn_output: required(&n.attn_output)?,
                        attn_q_norm: required(&n.attn_q_norm)?,
                        attn_k_norm: required(&n.attn_k_norm)?,
                    })
                }
            };

            let shared = if cfg.shared_expert_intermediate_size.is_some() {
                Some(SharedExpertTensors {
                    ffn_gate_inp_shexp: required(&m.ffn_gate_inp_shexp)?,
                    ffn_gate_shexp: required(&m.ffn_gate_shexp)?,
                    ffn_up_shexp: required(&m.ffn_up_shexp)?,
                    ffn_down_shexp: required(&m.ffn_down_shexp)?,
                })
            } else {
                None
            };

            let ffn = MoeFfnTensors {
                ffn_gate_inp: required(&m.ffn_gate_inp)?,
                ffn_gate_exps: required(&m.ffn_gate_exps)?,
                ffn_up_exps: required(&m.ffn_up_exps)?,
                ffn_down_exps: required(&m.ffn_down_exps)?,
                shared,
            };

            layers.push(LayerDescriptor {
                layer_idx,
                attn_norm,
                post_attention_norm,
                ffn_norm,
                attn,
                ffn,
            });
        }

        Ok(Self { token_embd, output_norm, output, layers })
    }

    /// Sum of weight bytes across every tensor. Useful for load-time VRAM
    /// budgeting before any upload happens.
    pub fn total_bytes(&self) -> u64 {
        let mut total = self.token_embd.size_bytes + self.output_norm.size_bytes;
        if let Some(o) = &self.output {
            total += o.size_bytes;
        }
        for l in &self.layers {
            total += l.attn_norm.size_bytes;
            if let Some(t) = &l.post_attention_norm {
                total += t.size_bytes;
            }
            if let Some(t) = &l.ffn_norm {
                total += t.size_bytes;
            }
            total += attn_block_bytes(&l.attn);
            total += ffn_bytes(&l.ffn);
        }
        total
    }
}

fn attn_block_bytes(block: &LayerAttnBlock) -> u64 {
    match block {
        LayerAttnBlock::Dense(d) => {
            let mut s = d.attn_q.size_bytes
                + d.attn_k.size_bytes
                + d.attn_v.size_bytes
                + d.attn_output.size_bytes
                + d.attn_q_norm.size_bytes
                + d.attn_k_norm.size_bytes;
            for b in [&d.attn_q_bias, &d.attn_k_bias, &d.attn_v_bias]
                .into_iter()
                .flatten()
            {
                s += b.size_bytes;
            }
            s
        }
        LayerAttnBlock::FullAttn(f) => {
            f.attn_q.size_bytes
                + f.attn_k.size_bytes
                + f.attn_v.size_bytes
                + f.attn_output.size_bytes
                + f.attn_q_norm.size_bytes
                + f.attn_k_norm.size_bytes
        }
        LayerAttnBlock::Gdn(g) => {
            let mut s = g.attn_qkv.size_bytes
                + g.attn_gate.size_bytes
                + g.ssm_a.size_bytes
                + g.ssm_dt_bias.size_bytes
                + g.ssm_conv1d.size_bytes
                + g.ssm_norm.size_bytes
                + g.ssm_out.size_bytes;
            for t in [&g.ssm_alpha, &g.ssm_beta, &g.ssm_ba]
                .into_iter()
                .flatten()
            {
                s += t.size_bytes;
            }
            s
        }
    }
}

fn ffn_bytes(f: &MoeFfnTensors) -> u64 {
    let mut s = f.ffn_gate_inp.size_bytes
        + f.ffn_gate_exps.size_bytes
        + f.ffn_up_exps.size_bytes
        + f.ffn_down_exps.size_bytes;
    if let Some(sh) = &f.shared {
        s += sh.ffn_gate_inp_shexp.size_bytes
            + sh.ffn_gate_shexp.size_bytes
            + sh.ffn_up_shexp.size_bytes
            + sh.ffn_down_shexp.size_bytes;
    }
    s
}
