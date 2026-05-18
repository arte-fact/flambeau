//! Gemma-4 dense GGUF → device. Per-layer SWA alternation; every layer
//! is FullAttn (no GDN/MoE in this first cut). Single entry parametrised
//! by `ShardMode`.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{Device, DevicePtr};
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingWeights, FfnWeights, LmHeadWeights, ModelLayout,
};
use flambeau_forward::loader::{
    load_dense_attn_layer, load_dense_ffn_layer, load_embedding, load_lm_head,
    DenseAttnLayerSpec, DenseFfnLayerSpec, EmbeddingSpec, LmHeadSpec, ShardMode,
};
use flambeau_quant::GgufFile;

use crate::config::Gemma4V2Config;

pub struct Gemma4V2Model {
    pub config: Gemma4V2Config,
    pub layout: ModelLayout,
    pub embedding: EmbeddingWeights,
    pub attn: Vec<AttnWeights>,
    pub ffn: Vec<FfnWeights>,
    pub lm_head: LmHeadWeights,

    pub(crate) allocs: Vec<(DevicePtr, usize)>,
    pub(crate) device_id: i32,
}

impl Gemma4V2Model {
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if device.default_stream().device_id() != self.device_id {
            bail!(
                "Gemma4V2Model::dispose: device mismatch (model on {}, called on {})",
                self.device_id,
                device.default_stream().device_id()
            );
        }
        for (ptr, bytes) in self.allocs.drain(..) {
            // SAFETY: ptr returned by `device.alloc(bytes)` in the loader.
            unsafe { device.dealloc(ptr, bytes) }
                .with_context(|| format!("dealloc {bytes} bytes"))?;
        }
        Ok(())
    }
}

fn load_with_shard(
    file: &GgufFile,
    device: &HipDevice,
    shard: ShardMode,
) -> Result<Gemma4V2Model> {
    let config = Gemma4V2Config::from_gguf(file).context("parse gemma4 config")?;
    let mut allocs: Vec<(DevicePtr, usize)> = Vec::new();

    // Gemma4: inpL *= sqrt(n_embd) after embed (memory: parity_vs_argmax_in_vocab).
    let embedding = load_embedding(
        file,
        device,
        &EmbeddingSpec {
            token_embd_name: "token_embd.weight",
            vocab_size: config.vocab_size,
            hidden: config.hidden,
            post_scale: Some((config.hidden as f32).sqrt()),
        },
        &mut allocs,
    )?;

    let mut attn = Vec::with_capacity(config.num_layers);
    let mut ffn = Vec::with_capacity(config.num_layers);
    for li in 0..config.num_layers {
        let p = format!("blk.{li}");
        let dims = config.attn[li];
        let n_kv_heads = config.num_kv_heads[li];

        let (norm, q, k, output, q_norm, k_norm) = (
            format!("{p}.attn_norm.weight"),
            format!("{p}.attn_q.weight"),
            format!("{p}.attn_k.weight"),
            format!("{p}.attn_output.weight"),
            format!("{p}.attn_q_norm.weight"),
            format!("{p}.attn_k_norm.weight"),
        );
        attn.push(load_dense_attn_layer(
            file,
            device,
            &DenseAttnLayerSpec {
                attn_norm_name: &norm,
                attn_q_name: &q,
                attn_k_name: &k,
                // V from K via DtoD memcpy (no attn_v on disk; memory:
                // gemma4_attn_output_proj_f16_saturate).
                attn_v_name: None,
                attn_output_name: &output,
                attn_q_norm_name: Some(&q_norm),
                attn_k_norm_name: Some(&k_norm),
                n_heads: config.num_heads,
                n_kv_heads,
                head_dim: dims.head_dim,
                hidden: config.hidden,
                rotated_dims: dims.rotated_dims,
                rope_theta: dims.rope_theta,
                window_size: dims.window_size,
                rms_eps: config.rms_eps,
                // Gemma4: softmax_scale = 1.0, not 1/sqrt(head_dim).
                softmax_scale: Some(1.0),
            },
            shard,
            &mut allocs,
        )?);

        let (ffn_norm, ffn_gate, ffn_up, ffn_down) = (
            format!("{p}.post_attention_norm.weight"),
            format!("{p}.ffn_gate.weight"),
            format!("{p}.ffn_up.weight"),
            format!("{p}.ffn_down.weight"),
        );
        ffn.push(load_dense_ffn_layer(
            file,
            device,
            &DenseFfnLayerSpec {
                ffn_norm_name: &ffn_norm,
                ffn_gate_name: &ffn_gate,
                ffn_up_name: &ffn_up,
                ffn_down_name: &ffn_down,
                hidden: config.hidden,
                intermediate: config.intermediate,
                activation: Activation::GeluTanh,
                rms_eps: config.rms_eps,
            },
            shard,
            &mut allocs,
        )?);
    }

    let lm_head_name = if config.tied_lm_head {
        "token_embd.weight"
    } else {
        "output.weight"
    };
    let final_logit_softcap = if config.final_logit_softcap > 0.0 {
        Some(config.final_logit_softcap)
    } else {
        None
    };
    let lm_head = load_lm_head(
        file,
        device,
        &LmHeadSpec {
            output_norm_name: "output_norm.weight",
            lm_head_name,
            vocab_size: config.vocab_size,
            hidden: config.hidden,
            rms_eps: config.rms_eps,
            final_logit_softcap,
        },
        &mut allocs,
    )?;

    let layout = ModelLayout {
        num_layers: config.num_layers,
        hidden: config.hidden,
        kv_max_seq_len: config.context_length,
    };
    let device_id = device.default_stream().device_id();
    Ok(Gemma4V2Model {
        config,
        layout,
        embedding,
        attn,
        ffn,
        lm_head,
        allocs,
        device_id,
    })
}

pub fn load_from_gguf(file: &GgufFile, device: &HipDevice) -> Result<Gemma4V2Model> {
    load_with_shard(file, device, ShardMode::Replicated)
}

pub fn load_tp_shard_from_gguf(
    file: &GgufFile,
    device: &HipDevice,
    rank: usize,
    n_ranks: usize,
) -> Result<Gemma4V2Model> {
    load_with_shard(file, device, ShardMode::Tp { rank, n_ranks })
}
