//! Tensor-presence resolver — given a parsed [`crate::config::Gemma4Config`]
//! and a [`flambeau_quant::GgufFile`], walk every expected tensor name and
//! confirm it resolves to a `TensorInfo`. Optional-on-tail tensors
//! (`attn_k`, `attn_v`, `attn_k_norm` when `has_kv == false`) are
//! treated as `NOT_REQUIRED`, mirroring llama.cpp PR #21739.
//!
//! S4 scope: name resolution + descriptor build only. Device upload
//! and on-the-fly split of `ffn_gate_up_exps` land in S5.

use flambeau_quant::{GgufFile, TensorInfo};

use crate::config::{Gemma4Config, Gemma4Variant};
use crate::layout::{FfnKind, ModelLayout};
use crate::names::{AttnNames, DenseFfnNames, GlobalNames, MoeFfnNames, PerLayerEmbedNames};

#[derive(Debug, thiserror::Error)]
pub enum Gemma4WeightsError {
    #[error("required tensor `{0}` is missing from the GGUF")]
    MissingTensor(String),
    #[error("tensor `{name}` has unexpected dims {got:?}; expected {expected:?}")]
    BadShape {
        name: String,
        got: Vec<u64>,
        expected: Vec<u64>,
    },
}

/// Resolved attention tensors for one layer.
/// - `attn_k` / `attn_k_norm` are optional **only** for shared-KV-tail
///   layers (`has_kv(il) == false`, mirrors llama.cpp PR #21739).
/// - `attn_v` is **always** optional (gemma4 "alternative attention" —
///   when absent the forward path uses `Vcur = Kcur`, see
///   `gemma4-iswa.cpp:83-86`).
/// - `layer_output_scale` is always optional.
#[derive(Debug, Clone)]
pub struct AttnTensors {
    pub attn_norm: TensorInfo,
    pub attn_q: TensorInfo,
    pub attn_k: Option<TensorInfo>,
    pub attn_v: Option<TensorInfo>,
    pub attn_output: TensorInfo,
    pub attn_q_norm: TensorInfo,
    pub attn_k_norm: Option<TensorInfo>,
    pub post_attention_norm: TensorInfo,
    pub layer_output_scale: Option<TensorInfo>,
}

#[derive(Debug, Clone)]
pub struct DenseFfnTensors {
    pub ffn_norm: TensorInfo,
    pub ffn_gate: TensorInfo,
    pub ffn_up: TensorInfo,
    pub ffn_down: TensorInfo,
    pub post_ffw_norm: TensorInfo,
}

#[derive(Debug, Clone)]
pub struct MoeFfnTensors {
    pub ffn_gate_inp: TensorInfo,
    pub ffn_gate_inp_scale: TensorInfo,
    pub ffn_gate_up_exps: TensorInfo,
    pub ffn_down_exps: TensorInfo,
    pub ffn_down_exps_scale: Option<TensorInfo>,
    pub pre_ffw_norm_2: TensorInfo,
    pub post_ffw_norm_1: TensorInfo,
    pub post_ffw_norm_2: TensorInfo,
}

#[derive(Debug, Clone)]
pub struct PerLayerEmbedTensors {
    pub inp_gate: TensorInfo,
    pub proj: TensorInfo,
    pub post_norm: TensorInfo,
}

#[derive(Debug, Clone)]
pub struct LayerTensors {
    pub attn: AttnTensors,
    /// Shared MLP — present on every layer of every variant (gemma4
    /// has dense FFN names even on MoE layers; the MoE branch is
    /// added in parallel through [`Self::moe`]).
    pub dense_ffn: DenseFfnTensors,
    /// Routed-expert tensors — `Some` iff the variant is MoE.
    pub moe: Option<MoeFfnTensors>,
    pub per_layer_embed: Option<PerLayerEmbedTensors>,
}

#[derive(Debug, Clone)]
pub struct GlobalTensors {
    pub token_embd: TensorInfo,
    pub output_norm: TensorInfo,
    /// `Some` iff the GGUF carries a distinct `output.weight`; `None`
    /// means the LM head is tied to `token_embd.weight`. All 5
    /// audited gemma4 GGUFs are tied (None).
    pub output: Option<TensorInfo>,
    pub rope_freqs: Option<TensorInfo>,
    pub per_layer_token_embd: Option<TensorInfo>,
    pub per_layer_model_proj: Option<TensorInfo>,
    pub per_layer_proj_norm: Option<TensorInfo>,
}

#[derive(Debug, Clone)]
pub struct ResolvedWeights {
    pub global: GlobalTensors,
    pub layers: Vec<LayerTensors>,
}

fn req<'a>(file: &'a GgufFile, name: &str) -> Result<&'a TensorInfo, Gemma4WeightsError> {
    file.tensors
        .get(name)
        .ok_or_else(|| Gemma4WeightsError::MissingTensor(name.to_string()))
}

fn opt<'a>(file: &'a GgufFile, name: &str) -> Option<&'a TensorInfo> {
    file.tensors.get(name)
}

/// Resolve every expected tensor name against the GGUF tensor index.
/// Returns a fully populated [`ResolvedWeights`]; downstream code can
/// build device buffers from the per-tensor offsets without touching
/// the metadata again.
pub fn resolve_weights(
    file: &GgufFile,
    cfg: &Gemma4Config,
    layout: &ModelLayout,
) -> Result<ResolvedWeights, Gemma4WeightsError> {
    let globals = GlobalNames::default_names();
    let global = GlobalTensors {
        token_embd: req(file, &globals.token_embd)?.clone(),
        output_norm: req(file, &globals.output_norm)?.clone(),
        // `output` is optional — tied LM head when absent.
        output: opt(file, &globals.output).cloned(),
        rope_freqs: opt(file, &globals.rope_freqs).cloned(),
        per_layer_token_embd: if cfg.per_layer_embed.is_some() {
            Some(req(file, &globals.per_layer_token_embd)?.clone())
        } else {
            None
        },
        per_layer_model_proj: if cfg.per_layer_embed.is_some() {
            Some(req(file, &globals.per_layer_model_proj)?.clone())
        } else {
            None
        },
        per_layer_proj_norm: if cfg.per_layer_embed.is_some() {
            Some(req(file, &globals.per_layer_proj_norm)?.clone())
        } else {
            None
        },
    };

    let mut layers = Vec::with_capacity(cfg.num_layers);
    for spec in &layout.layers {
        let il = spec.index;
        let an = AttnNames::for_layer(il);
        let attn = AttnTensors {
            attn_norm: req(file, &an.attn_norm)?.clone(),
            attn_q: req(file, &an.attn_q)?.clone(),
            attn_k: if spec.has_kv {
                Some(req(file, &an.attn_k)?.clone())
            } else {
                opt(file, &an.attn_k).cloned()
            },
            // `attn_v` is always optional: gemma4 full-attention layers
            // omit V proj and reuse K (alternative attention). The
            // forward path checks for None and routes Vcur = Kcur.
            attn_v: opt(file, &an.attn_v).cloned(),
            attn_output: req(file, &an.attn_output)?.clone(),
            attn_q_norm: req(file, &an.attn_q_norm)?.clone(),
            attn_k_norm: if spec.has_kv {
                Some(req(file, &an.attn_k_norm)?.clone())
            } else {
                opt(file, &an.attn_k_norm).cloned()
            },
            post_attention_norm: req(file, &an.post_attention_norm)?.clone(),
            layer_output_scale: opt(file, &an.layer_output_scale).cloned(),
        };

        let dn = DenseFfnNames::for_layer(il);
        let dense_ffn = DenseFfnTensors {
            ffn_norm: req(file, &dn.ffn_norm)?.clone(),
            ffn_gate: req(file, &dn.ffn_gate)?.clone(),
            ffn_up: req(file, &dn.ffn_up)?.clone(),
            ffn_down: req(file, &dn.ffn_down)?.clone(),
            post_ffw_norm: req(file, &dn.post_ffw_norm)?.clone(),
        };

        let moe = if spec.ffn_kind == FfnKind::Moe {
            let mn = MoeFfnNames::for_layer(il);
            Some(MoeFfnTensors {
                ffn_gate_inp: req(file, &mn.ffn_gate_inp)?.clone(),
                ffn_gate_inp_scale: req(file, &mn.ffn_gate_inp_scale)?.clone(),
                ffn_gate_up_exps: req(file, &mn.ffn_gate_up_exps)?.clone(),
                ffn_down_exps: req(file, &mn.ffn_down_exps)?.clone(),
                ffn_down_exps_scale: opt(file, &mn.ffn_down_exps_scale).cloned(),
                pre_ffw_norm_2: req(file, &mn.pre_ffw_norm_2)?.clone(),
                post_ffw_norm_1: req(file, &mn.post_ffw_norm_1)?.clone(),
                post_ffw_norm_2: req(file, &mn.post_ffw_norm_2)?.clone(),
            })
        } else {
            None
        };

        let per_layer_embed = if cfg.per_layer_embed.is_some() {
            let pn = PerLayerEmbedNames::for_layer(il);
            Some(PerLayerEmbedTensors {
                inp_gate: req(file, &pn.inp_gate)?.clone(),
                proj: req(file, &pn.proj)?.clone(),
                post_norm: req(file, &pn.post_norm)?.clone(),
            })
        } else {
            None
        };

        layers.push(LayerTensors {
            attn,
            dense_ffn,
            moe,
            per_layer_embed,
        });
    }

    Ok(ResolvedWeights { global, layers })
}

/// Sanity-check the resolved tensors against expected shapes from the
/// config. Catches loader bugs before any device upload (S5). Strict
/// per-shape — diverging shapes produce a [`Gemma4WeightsError::BadShape`].
pub fn validate_shapes(
    cfg: &Gemma4Config,
    layout: &ModelLayout,
    weights: &ResolvedWeights,
) -> Result<(), Gemma4WeightsError> {
    let hidden = cfg.hidden_size as u64;
    let vocab = cfg.vocab_size as u64;
    let n_layer = cfg.num_layers as u64;

    expect_dims(&weights.global.token_embd, &[vocab, hidden])?;
    expect_dims(&weights.global.output_norm, &[hidden])?;
    if let Some(t) = &weights.global.output {
        expect_dims(t, &[vocab, hidden])?;
    }
    if let Some(per) = cfg.per_layer_embed {
        let pe = per.n_embd_per_layer as u64;
        expect_dims(
            weights.global.per_layer_token_embd.as_ref().expect("checked"),
            &[vocab, pe * n_layer],
        )?;
        expect_dims(
            weights.global.per_layer_model_proj.as_ref().expect("checked"),
            &[pe * n_layer, hidden],
        )?;
        expect_dims(
            weights.global.per_layer_proj_norm.as_ref().expect("checked"),
            &[pe],
        )?;
    }

    for (i, spec) in layout.layers.iter().enumerate() {
        let l = &weights.layers[i];
        let n_heads = spec.n_heads as u64;
        let n_kv = spec.n_kv_heads as u64;
        let head_dim = spec.head_dim as u64;
        let q_total = n_heads * head_dim;
        let kv_total = n_kv * head_dim;

        expect_dims(&l.attn.attn_norm, &[hidden])?;
        expect_dims(&l.attn.attn_q, &[q_total, hidden])?;
        expect_dims(&l.attn.attn_output, &[hidden, q_total])?;
        expect_dims(&l.attn.attn_q_norm, &[head_dim])?;
        expect_dims(&l.attn.post_attention_norm, &[hidden])?;
        if let Some(t) = &l.attn.attn_k {
            expect_dims(t, &[kv_total, hidden])?;
        }
        if let Some(t) = &l.attn.attn_v {
            expect_dims(t, &[kv_total, hidden])?;
        }
        if let Some(t) = &l.attn.attn_k_norm {
            expect_dims(t, &[head_dim])?;
        }
        if let Some(t) = &l.attn.layer_output_scale {
            expect_dims(t, &[1])?;
        }

        let ff = cfg.feed_forward_length as u64;
        expect_dims(&l.dense_ffn.ffn_norm, &[hidden])?;
        expect_dims(&l.dense_ffn.ffn_gate, &[ff, hidden])?;
        expect_dims(&l.dense_ffn.ffn_up, &[ff, hidden])?;
        expect_dims(&l.dense_ffn.ffn_down, &[hidden, ff])?;
        expect_dims(&l.dense_ffn.post_ffw_norm, &[hidden])?;

        if let Some(moe) = &l.moe {
            let m = cfg.moe.expect("variant says MoE but cfg has no moe block");
            let n_exp = m.num_experts as u64;
            let n_ff_exp = m.moe_intermediate_size as u64;
            expect_dims(&moe.ffn_gate_inp, &[n_exp, hidden])?;
            expect_dims(&moe.ffn_gate_inp_scale, &[hidden])?;
            expect_dims(&moe.ffn_gate_up_exps, &[n_exp, 2 * n_ff_exp, hidden])?;
            expect_dims(&moe.ffn_down_exps, &[n_exp, hidden, n_ff_exp])?;
            if let Some(s) = &moe.ffn_down_exps_scale {
                expect_dims(s, &[n_exp])?;
            }
            expect_dims(&moe.pre_ffw_norm_2, &[hidden])?;
            expect_dims(&moe.post_ffw_norm_1, &[hidden])?;
            expect_dims(&moe.post_ffw_norm_2, &[hidden])?;
        }

        if let Some(per) = &l.per_layer_embed {
            let pe = cfg
                .per_layer_embed
                .expect("layer has per_layer_embed but cfg does not")
                .n_embd_per_layer as u64;
            expect_dims(&per.inp_gate, &[pe, hidden])?;
            expect_dims(&per.proj, &[hidden, pe])?;
            expect_dims(&per.post_norm, &[hidden])?;
        }
    }

    // Variant cross-check.
    let _ = match layout.variant {
        Gemma4Variant::Moe26BA4B => cfg.moe.expect("MoE variant must carry moe dims"),
        _ => return Ok(()),
    };

    Ok(())
}

fn expect_dims(t: &TensorInfo, expected: &[u64]) -> Result<(), Gemma4WeightsError> {
    if t.dims.as_slice() != expected {
        return Err(Gemma4WeightsError::BadShape {
            name: t.name.clone(),
            got: t.dims.clone(),
            expected: expected.to_vec(),
        });
    }
    Ok(())
}
