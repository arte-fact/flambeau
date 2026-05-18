//! qwen35 GGUF → device. Per-layer dispatch on `is_recurrent` picks
//! either dense-attn or GDN; FFN is dense everywhere. GDN TP mode is
//! auto-picked by `arch::gdn_tp_mode_for` from per-rank geometry.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{Device, DevicePtr};
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingWeights, FfnWeights, GdnWeights, LayerKind, LmHeadWeights,
    ModelLayout,
};
use flambeau_forward::loader::{
    load_dense_attn_layer, load_dense_ffn_layer, load_embedding, load_gdn_layer, load_lm_head,
    DenseAttnLayerSpec, DenseFfnLayerSpec, EmbeddingSpec, GdnLayerSpec, LmHeadSpec, ShardMode,
};
use flambeau_quant::GgufFile;

use crate::arch::{gdn_tp_mode_for, per_rank_gdn_dims};
use crate::config::Qwen35V2Config;

pub struct Qwen35V2Model {
    pub config: Qwen35V2Config,
    pub layout: ModelLayout,
    pub embedding: EmbeddingWeights,
    pub layer_kinds: Vec<LayerKind>,
    pub full_attn: Vec<Option<AttnWeights>>,
    pub gdn: Vec<Option<GdnWeights>>,
    pub ffn: Vec<Option<FfnWeights>>,
    pub lm_head: LmHeadWeights,

    pub(crate) allocs: Vec<(DevicePtr, usize)>,
    pub(crate) device_id: i32,
}

impl Qwen35V2Model {
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if device.default_stream().device_id() != self.device_id {
            bail!(
                "Qwen35V2Model::dispose: device mismatch (model on {}, called on {})",
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
    layer_range: Option<(usize, usize)>,
    ctx_cap: Option<usize>,
) -> Result<Qwen35V2Model> {
    let mut config = Qwen35V2Config::from_gguf(file).context("parse qwen35 config")?;
    if let Some(cap) = ctx_cap {
        if cap > 0 && cap < config.context_length {
            config.context_length = cap;
        }
    }
    let mut allocs: Vec<(DevicePtr, usize)> = Vec::new();
    let n_ranks = shard.n_ranks();
    let tp_mode = gdn_tp_mode_for(config.gdn, n_ranks);
    // Per-rank GdnDims under TP. At n_ranks == 1 this is the identity.
    let g = if n_ranks > 1 {
        per_rank_gdn_dims(config.gdn, n_ranks)
    } else {
        config.gdn
    };

    let owns_embed = layer_range.map_or(true, |(s, _)| s == 0);
    let owns_lm_head = layer_range.map_or(true, |(_, e)| e == config.num_layers);
    let in_range = |li: usize| -> bool {
        layer_range.map_or(true, |(s, e)| li >= s && li < e)
    };

    let embedding = if owns_embed {
        load_embedding(
            file,
            device,
            &EmbeddingSpec {
                token_embd_name: "token_embd.weight",
                vocab_size: config.vocab_size,
                hidden: config.hidden,
                post_scale: None,
            },
            &mut allocs,
        )?
    } else {
        EmbeddingWeights::placeholder(config.vocab_size, config.hidden)
    };

    let mut layer_kinds = Vec::with_capacity(config.num_layers);
    let mut full_attn = Vec::with_capacity(config.num_layers);
    let mut gdn = Vec::with_capacity(config.num_layers);
    let mut ffn = Vec::with_capacity(config.num_layers);

    for li in 0..config.num_layers {
        if !in_range(li) {
            full_attn.push(None);
            gdn.push(None);
            ffn.push(None);
            layer_kinds.push(if config.is_recurrent(li) {
                LayerKind::Gdn
            } else {
                LayerKind::FullAttn
            });
            continue;
        }
        let p = format!("blk.{li}");
        let attn_norm_name = format!("{p}.attn_norm.weight");

        if config.is_recurrent(li) {
            let (qkv, gate, alpha, beta, out, dt_bias, a, conv1d, norm_w) = (
                format!("{p}.attn_qkv.weight"),
                format!("{p}.attn_gate.weight"),
                format!("{p}.ssm_alpha.weight"),
                format!("{p}.ssm_beta.weight"),
                format!("{p}.ssm_out.weight"),
                format!("{p}.ssm_dt.bias"),
                format!("{p}.ssm_a"),
                format!("{p}.ssm_conv1d.weight"),
                format!("{p}.ssm_norm.weight"),
            );
            let w = load_gdn_layer(
                file,
                device,
                &GdnLayerSpec {
                    attn_norm_name: &attn_norm_name,
                    attn_qkv_name: &qkv,
                    attn_gate_name: &gate,
                    ssm_alpha_name: &alpha,
                    ssm_beta_name: &beta,
                    ssm_out_name: &out,
                    ssm_dt_bias_name: &dt_bias,
                    ssm_a_name: &a,
                    ssm_conv1d_name: &conv1d,
                    ssm_norm_name: &norm_w,
                    hidden: config.hidden,
                    dims: g,
                    rms_eps: config.rms_eps,
                    rep_inner_layout: false,
                    tp_mode,
                },
                shard,
                &mut allocs,
            )?;
            full_attn.push(None);
            gdn.push(Some(w));
            layer_kinds.push(LayerKind::Gdn);
        } else {
            let (q, k, v, output, q_norm, k_norm) = (
                format!("{p}.attn_q.weight"),
                format!("{p}.attn_k.weight"),
                format!("{p}.attn_v.weight"),
                format!("{p}.attn_output.weight"),
                format!("{p}.attn_q_norm.weight"),
                format!("{p}.attn_k_norm.weight"),
            );
            let w = load_dense_attn_layer(
                file,
                device,
                &DenseAttnLayerSpec {
                    attn_norm_name: &attn_norm_name,
                    attn_q_name: &q,
                    attn_k_name: &k,
                    attn_v_name: Some(&v),
                    attn_output_name: &output,
                    attn_q_norm_name: Some(&q_norm),
                    attn_k_norm_name: Some(&k_norm),
                    n_heads: config.n_heads,
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim,
                    hidden: config.hidden,
                    rotated_dims: config.rotated_dims,
                    rope_theta: config.rope_theta,
                    rope_variant: flambeau_forward::ctx::RopeVariant::NeoxSplit,
                    window_size: 0,
                    rms_eps: config.rms_eps,
                    softmax_scale: None,
                    attn_q_gated: true,
                },
                shard,
                &mut allocs,
            )?;
            full_attn.push(Some(w));
            gdn.push(None);
            layer_kinds.push(LayerKind::FullAttn);
        }

        // qwen35 uses `post_attention_norm` as the pre-FFN norm on hybrid arches.
        let (ffn_norm, ffn_gate, ffn_up, ffn_down) = (
            format!("{p}.post_attention_norm.weight"),
            format!("{p}.ffn_gate.weight"),
            format!("{p}.ffn_up.weight"),
            format!("{p}.ffn_down.weight"),
        );
        ffn.push(Some(load_dense_ffn_layer(
            file,
            device,
            &DenseFfnLayerSpec {
                ffn_norm_name: &ffn_norm,
                ffn_gate_name: &ffn_gate,
                ffn_up_name: &ffn_up,
                ffn_down_name: &ffn_down,
                hidden: config.hidden,
                intermediate: config.intermediate,
                activation: Activation::SwiGLU,
                rms_eps: config.rms_eps,
            },
            shard,
            &mut allocs,
        )?));
    }

    let lm_head = if owns_lm_head {
        let lm_head_name = if config.tied_lm_head {
            "token_embd.weight"
        } else {
            "output.weight"
        };
        load_lm_head(
            file,
            device,
            &LmHeadSpec {
                output_norm_name: "output_norm.weight",
                lm_head_name,
                vocab_size: config.vocab_size,
                hidden: config.hidden,
                rms_eps: config.rms_eps,
                final_logit_softcap: None,
            },
            &mut allocs,
        )?
    } else {
        LmHeadWeights::placeholder(config.vocab_size, config.hidden, config.rms_eps)
    };

    let layout = ModelLayout {
        num_layers: config.num_layers,
        hidden: config.hidden,
        kv_max_seq_len: config.context_length,
    };
    let device_id = device.default_stream().device_id();
    Ok(Qwen35V2Model {
        config,
        layout,
        embedding,
        layer_kinds,
        full_attn,
        gdn,
        ffn,
        lm_head,
        allocs,
        device_id,
    })
}

pub fn load_from_gguf(
    file: &GgufFile,
    device: &HipDevice,
    layer_range: Option<(usize, usize)>,
    ctx_cap: Option<usize>,
) -> Result<Qwen35V2Model> {
    load_with_shard(file, device, ShardMode::Replicated, layer_range, ctx_cap)
}

pub fn load_tp_shard_from_gguf(
    file: &GgufFile,
    device: &HipDevice,
    rank: usize,
    n_ranks: usize,
    layer_range: Option<(usize, usize)>,
    ctx_cap: Option<usize>,
) -> Result<Qwen35V2Model> {
    load_with_shard(
        file,
        device,
        ShardMode::Tp { rank, n_ranks },
        layer_range,
        ctx_cap,
    )
}
