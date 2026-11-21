//! qwen35 GGUF → device. Per-layer dispatch on `is_recurrent` picks
//! either dense-attn or GDN; FFN is dense everywhere. GDN TP mode is
//! auto-picked by `arch::gdn_tp_mode_for` from per-rank geometry.

use anyhow::{bail, Context, Result};
use flambeau_core::{Device, DevicePtr};
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingWeights, FfnWeights, GdnWeights, LayerKind, LmHeadWeights,
    ModelLayout,
};
use flambeau_forward::loader::{
    gdn_tp_mode_for, load_dense_attn_layer, load_dense_ffn_layer, load_embedding, load_gdn_layer,
    load_lm_head, per_rank_gdn_dims, DenseAttnLayerSpec, DenseFfnLayerSpec, EmbeddingSpec,
    GdnLayerSpec, LmHeadSpec, LoadOptions, ShardMode,
};
use flambeau_quant::GgufFile;

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
    /// Embedded MTP/nextn draft head (`blk.{num_layers}`), when the GGUF
    /// carries it. Attn/FFN are sharded by `shard` (TP-aware); `eh_proj`
    /// halves stay replicated per rank (the post-AR hidden is full on every
    /// rank, so no AllReduce is needed after eh_proj).
    pub mtp: Option<flambeau_forward::ctx::MtpWeights>,

    pub(crate) allocs: Vec<(DevicePtr, usize)>,
    pub(crate) device_id: i32,
}

impl Qwen35V2Model {
    pub fn dispose(&mut self, device: &impl Device) -> Result<()> {
        if device.id() != self.device_id {
            bail!(
                "Qwen35V2Model::dispose: device mismatch (model on {}, called on {})",
                self.device_id,
                device.id()
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
    device: &impl Device,
    shard: ShardMode,
    layer_range: Option<(usize, usize)>,
    opts: LoadOptions,
) -> Result<Qwen35V2Model> {
    let mut config = Qwen35V2Config::from_gguf(file).context("parse qwen35 config")?;
    if let Some(cap) = opts.ctx_cap {
        if cap > 0 && cap < config.context_length {
            config.context_length = cap;
        }
    }
    let mut allocs: Vec<(DevicePtr, usize)> = Vec::new();
    let n_ranks = shard.n_ranks();
    let tp_mode = gdn_tp_mode_for(config.gdn, n_ranks);
    let g = per_rank_gdn_dims(config.gdn, n_ranks);

    let owns_embed = layer_range.is_none_or(|(s, _)| s == 0);
    let owns_lm_head = layer_range.is_none_or(|(_, e)| e == config.num_layers);
    let in_range = |li: usize| -> bool { layer_range.is_none_or(|(s, e)| li >= s && li < e) };

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
                    post_attn_norm_name: None,
                    attn_q_name: &q,
                    attn_k_name: &k,
                    attn_v_name: Some(&v),
                    attn_output_name: &output,
                    attn_q_norm_name: Some(&q_norm),
                    attn_k_norm_name: Some(&k_norm),
                    attn_v_unit_norm: false,
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
                    kv_share_src: None,
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
                post_ffn_norm_name: None,
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

    // MTP/nextn draft head: loaded only when requested, the GGUF embeds
    // it and this rank owns the LM head. The draft block is
    // `blk.{num_layers}`.
    let mtp = if mtp_head_wanted(opts, config.nextn_layers, owns_lm_head) {
        let p = format!("blk.{}", config.num_layers);
        let nl = |s: &str| format!("{p}.{s}");
        let attn = load_dense_attn_layer(
            file,
            device,
            &DenseAttnLayerSpec {
                attn_norm_name: &nl("attn_norm.weight"),
                post_attn_norm_name: None,
                attn_q_name: &nl("attn_q.weight"),
                attn_k_name: &nl("attn_k.weight"),
                attn_v_name: Some(&nl("attn_v.weight")),
                attn_output_name: &nl("attn_output.weight"),
                attn_q_norm_name: Some(&nl("attn_q_norm.weight")),
                attn_k_norm_name: Some(&nl("attn_k_norm.weight")),
                attn_v_unit_norm: false,
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
                kv_share_src: None,
            },
            shard,
            &mut allocs,
        )?;
        let ffn = load_dense_ffn_layer(
            file,
            device,
            &DenseFfnLayerSpec {
                ffn_norm_name: &nl("post_attention_norm.weight"),
                post_ffn_norm_name: None,
                ffn_gate_name: &nl("ffn_gate.weight"),
                ffn_up_name: &nl("ffn_up.weight"),
                ffn_down_name: &nl("ffn_down.weight"),
                hidden: config.hidden,
                intermediate: config.intermediate,
                activation: Activation::SwiGLU,
                rms_eps: config.rms_eps,
            },
            shard,
            &mut allocs,
        )?;
        // eh_proj: [2*hidden, hidden] column-major (ne0 = 2h contiguous).
        // Split the input dim per output column: e-half = [0..h),
        // h-half = [h..2h). Dequantize to F32, split, upload as F16.
        let h = config.hidden;
        let eh_raw = file
            .dequantize_tensor(&nl("nextn.eh_proj.weight"))
            .with_context(|| format!("dequantize {p}.nextn.eh_proj.weight"))?;
        anyhow::ensure!(
            eh_raw.len() == 2 * h * h,
            "eh_proj: {} elems, expected {}",
            eh_raw.len(),
            2 * h * h
        );
        let mut w_e = vec![0f32; h * h];
        let mut w_h = vec![0f32; h * h];
        for j in 0..h {
            for i in 0..h {
                w_e[i + j * h] = eh_raw[i + j * 2 * h];
                w_h[i + j * h] = eh_raw[h + i + j * 2 * h];
            }
        }
        let eh_proj_e =
            flambeau_forward::loader::upload_f16_from_f32(device, &w_e, &mut allocs)?;
        let eh_proj_h =
            flambeau_forward::loader::upload_f16_from_f32(device, &w_h, &mut allocs)?;
        let enorm = flambeau_forward::loader::upload_dequant_to_f16(
            file,
            device,
            &nl("nextn.enorm.weight"),
            h,
            &mut allocs,
        )?;
        let hnorm = flambeau_forward::loader::upload_dequant_to_f16(
            file,
            device,
            &nl("nextn.hnorm.weight"),
            h,
            &mut allocs,
        )?;
        let shared_head_norm = flambeau_forward::loader::upload_dequant_to_f16(
            file,
            device,
            &nl("nextn.shared_head_norm.weight"),
            h,
            &mut allocs,
        )?;
        Some(flambeau_forward::ctx::MtpWeights {
            eh_proj_e,
            eh_proj_h,
            enorm,
            hnorm,
            shared_head_norm,
            attn,
            ffn,
            rms_eps: config.rms_eps,
        })
    } else {
        None
    };

    let layout = ModelLayout {
        num_layers: config.num_layers,
        hidden: config.hidden,
        kv_max_seq_len: config.context_length,
    };
    let device_id = device.id();
    Ok(Qwen35V2Model {
        config,
        layout,
        embedding,
        layer_kinds,
        full_attn,
        gdn,
        ffn,
        lm_head,
        mtp,
        allocs,
        device_id,
    })
}

pub fn load_from_gguf(
    file: &GgufFile,
    device: &impl Device,
    layer_range: Option<(usize, usize)>,
    opts: LoadOptions,
) -> Result<Qwen35V2Model> {
    load_with_shard(file, device, ShardMode::Replicated, layer_range, opts)
}

pub fn load_tp_shard_from_gguf(
    file: &GgufFile,
    device: &impl Device,
    rank: usize,
    n_ranks: usize,
    layer_range: Option<(usize, usize)>,
    opts: LoadOptions,
) -> Result<Qwen35V2Model> {
    load_with_shard(
        file,
        device,
        ShardMode::Tp { rank, n_ranks },
        layer_range,
        opts,
    )
}

fn mtp_head_wanted(opts: LoadOptions, nextn_layers: usize, owns_lm_head: bool) -> bool {
    opts.draft_head && nextn_layers > 0 && owns_lm_head
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtp_head_loads_only_when_requested() {
        let on = LoadOptions {
            ctx_cap: None,
            draft_head: true,
        };
        let off = LoadOptions::default();
        assert!(mtp_head_wanted(on, 1, true));
        assert!(!mtp_head_wanted(off, 1, true));
        assert!(!mtp_head_wanted(on, 0, true));
        assert!(!mtp_head_wanted(on, 1, false));
    }
}
