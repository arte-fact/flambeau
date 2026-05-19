//! qwen35moe GGUF → device. Per-layer dispatch on `is_recurrent`
//! picks GDN or full-attn-gated; FFN is routed MoE on every layer,
//! plus an always-on shared expert when the GGUF carries the
//! `ffn_*_shexp` tensors (Qwen3.6-35B-A3B).

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{Device, DevicePtr};
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingWeights, GdnWeights, LayerKind, LmHeadWeights, ModelLayout,
    MoeWeights, SharedExpertWeights,
};
use flambeau_forward::loader::{
    load_dense_attn_layer, load_embedding, load_gdn_layer, load_lm_head, upload_col_sharded_quant,
    upload_dequant_to_f16, upload_f32_tensor, upload_moe_experts_stacked,
    upload_moe_experts_stacked_col_sharded, upload_moe_experts_stacked_row_sharded,
    upload_quant_weight, upload_row_sharded_quant, DenseAttnLayerSpec, EmbeddingSpec,
    GdnLayerSpec, GdnTpMode, LmHeadSpec, ShardMode,
};
use flambeau_quant::GgufFile;

use crate::config::Qwen35MoeV2Config;

pub struct Qwen35MoeV2Model {
    pub config: Qwen35MoeV2Config,
    pub layout: ModelLayout,
    pub embedding: EmbeddingWeights,
    pub layer_kinds: Vec<LayerKind>,
    pub full_attn: Vec<Option<AttnWeights>>,
    pub gdn: Vec<Option<GdnWeights>>,
    pub ffn: Vec<Option<MoeWeights>>,
    pub lm_head: LmHeadWeights,

    pub(crate) allocs: Vec<(DevicePtr, usize)>,
    pub(crate) device_id: i32,
}

impl Qwen35MoeV2Model {
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if device.default_stream().device_id() != self.device_id {
            bail!(
                "Qwen35MoeV2Model::dispose: device mismatch (model on {}, called on {})",
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
) -> Result<Qwen35MoeV2Model> {
    let mut config = Qwen35MoeV2Config::from_gguf(file).context("parse qwen35moe config")?;
    if let Some(cap) = ctx_cap {
        if cap > 0 && cap < config.context_length {
            config.context_length = cap;
        }
    }
    let mut allocs: Vec<(DevicePtr, usize)> = Vec::new();
    let n_ranks = shard.n_ranks();
    // Per-rank GdnDims under TP (KReplicated mode keeps num_k_heads at
    // its global value; only num_v_heads divides). At n_ranks == 1
    // this is the identity.
    let g = if n_ranks > 1 {
        crate::arch::per_rank_gdn_dims(config.gdn, n_ranks)
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
                    tp_mode: GdnTpMode::KReplicated,
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
                },
                shard,
                &mut allocs,
            )?;
            full_attn.push(Some(w));
            gdn.push(None);
            layer_kinds.push(LayerKind::FullAttn);
        }

        let ffn_norm_name = format!("{p}.post_attention_norm.weight");
        let router_name = format!("{p}.ffn_gate_inp.weight");
        let gate_exps_name = format!("{p}.ffn_gate_exps.weight");
        let up_exps_name = format!("{p}.ffn_up_exps.weight");
        let down_exps_name = format!("{p}.ffn_down_exps.weight");

        let ffn_norm = upload_dequant_to_f16(
            file,
            device,
            &ffn_norm_name,
            config.hidden,
            &mut allocs,
        )?;
        let router = upload_quant_weight(
            file,
            device,
            &router_name,
            config.num_experts * config.hidden,
            &mut allocs,
        )?;
        // gate/up: `[n_experts, intermediate, hidden]` — col-shard the
        // `intermediate` dim (= outer dim of each expert's slab) so each
        // rank computes a partial gate/up over [local_intermediate, hidden].
        let experts_gate = upload_moe_experts_stacked_col_sharded(
            file,
            device,
            &gate_exps_name,
            config.num_experts,
            config.expert_intermediate,
            config.hidden,
            shard,
            &mut allocs,
        )?;
        let experts_up = upload_moe_experts_stacked_col_sharded(
            file,
            device,
            &up_exps_name,
            config.num_experts,
            config.expert_intermediate,
            config.hidden,
            shard,
            &mut allocs,
        )?;
        // down: `[n_experts, hidden, intermediate]` — row-shard the
        // `intermediate` (= inner row dim). The local matmul yields a
        // [hidden] partial; the v2 moe_ffn composite's ar_sum_f32 folds
        // the cross-rank sum.
        let experts_down = upload_moe_experts_stacked_row_sharded(
            file,
            device,
            &down_exps_name,
            config.num_experts,
            config.hidden,
            config.expert_intermediate,
            shard,
            &mut allocs,
        )?;
        let shared = if config.shared_expert_intermediate > 0 {
            let gate_shexp_name = format!("{p}.ffn_gate_shexp.weight");
            let up_shexp_name = format!("{p}.ffn_up_shexp.weight");
            let down_shexp_name = format!("{p}.ffn_down_shexp.weight");
            let gate_inp_shexp_name = format!("{p}.ffn_gate_inp_shexp.weight");
            let inter = config.shared_expert_intermediate;
            // Shared-expert TP: gate/up col-shard along `intermediate`
            // (= outer rows of `[intermediate, hidden]`); down row-shard
            // along `intermediate` (= inner cols of `[hidden, intermediate]`).
            let (gate, up, down) = match shard {
                ShardMode::Replicated => (
                    upload_quant_weight(file, device, &gate_shexp_name, inter * config.hidden, &mut allocs)?,
                    upload_quant_weight(file, device, &up_shexp_name, inter * config.hidden, &mut allocs)?,
                    upload_quant_weight(file, device, &down_shexp_name, config.hidden * inter, &mut allocs)?,
                ),
                ShardMode::Tp { rank, n_ranks } => (
                    upload_col_sharded_quant(file, device, &gate_shexp_name, inter, config.hidden, rank, n_ranks, &mut allocs)?,
                    upload_col_sharded_quant(file, device, &up_shexp_name, inter, config.hidden, rank, n_ranks, &mut allocs)?,
                    upload_row_sharded_quant(file, device, &down_shexp_name, config.hidden, inter, rank, n_ranks, &mut allocs)?,
                ),
            };
            let gate_inp = file
                .info(&gate_inp_shexp_name)
                .ok()
                .map(|_| {
                    upload_f32_tensor(
                        file,
                        device,
                        &gate_inp_shexp_name,
                        config.hidden,
                        &mut allocs,
                    )
                })
                .transpose()?;
            Some(SharedExpertWeights {
                gate,
                up,
                down,
                gate_inp,
                intermediate: inter / shard.n_ranks(),
            })
        } else {
            None
        };
        ffn.push(Some(MoeWeights {
            ffn_norm,
            router,
            experts_gate,
            experts_up,
            experts_down,
            n_experts: config.num_experts,
            experts_per_tok: config.experts_per_tok,
            activation: Activation::SwiGLU,
            rms_eps: config.rms_eps,
            shared,
        }));
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
    Ok(Qwen35MoeV2Model {
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
) -> Result<Qwen35MoeV2Model> {
    load_with_shard(file, device, ShardMode::Replicated, layer_range, ctx_cap)
}

pub fn load_tp_shard_from_gguf(
    file: &GgufFile,
    device: &HipDevice,
    rank: usize,
    n_ranks: usize,
    layer_range: Option<(usize, usize)>,
    ctx_cap: Option<usize>,
) -> Result<Qwen35MoeV2Model> {
    load_with_shard(
        file,
        device,
        ShardMode::Tp { rank, n_ranks },
        layer_range,
        ctx_cap,
    )
}
