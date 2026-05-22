//! `Arch` impl for `flambeau_forward::Session<Qwen35MoeV2>`.
//! Hybrid (GDN + full-attn-gated alternation) + routed MoE FFN.
//! TP uses `GdnTpMode::KReplicated` ("rep_outer"). MoE experts
//! are currently replicated on every rank (loader is SD-only;
//! TP expert sharding is a follow-up).

use anyhow::Result;
use flambeau_backend_hip::HipDevice;
use flambeau_forward::ctx::{ForwardCtx, GdnDims};
use flambeau_forward::loader::ShardMode;
use flambeau_forward::runtime::Arch;
use flambeau_forward::{per_layer_kv_widths, ScratchConfig};
use flambeau_quant::GgufFile;

use crate::{forward, load_from_gguf, Qwen35MoeV2Model};

pub struct Qwen35MoeV2;

pub(crate) fn per_rank_gdn_dims(g: GdnDims, n_ranks: usize) -> GdnDims {
    let num_v_heads_local = g.num_v_heads / n_ranks;
    let num_k_heads_local = g.num_k_heads;
    let d_inner_local = num_v_heads_local * g.head_v_dim;
    let conv_channels_local = 2 * num_k_heads_local * g.head_k_dim + d_inner_local;
    GdnDims {
        d_inner: d_inner_local,
        num_v_heads: num_v_heads_local,
        num_k_heads: num_k_heads_local,
        head_k_dim: g.head_k_dim,
        head_v_dim: g.head_v_dim,
        conv_channels: conv_channels_local,
        conv_kernel: g.conv_kernel,
    }
}

impl Arch for Qwen35MoeV2 {
    type Model = Qwen35MoeV2Model;

    fn arch_tag() -> &'static str {
        "qwen35moe"
    }

    fn load(
        file: &GgufFile,
        device: &HipDevice,
        shard: ShardMode,
        layer_range: Option<(usize, usize)>,
        ctx_cap: Option<usize>,
    ) -> Result<Self::Model> {
        match shard {
            ShardMode::Replicated => load_from_gguf(file, device, layer_range, ctx_cap),
            ShardMode::Tp { rank, n_ranks } => crate::loader::load_tp_shard_from_gguf(
                file,
                device,
                rank,
                n_ranks,
                layer_range,
                ctx_cap,
            ),
        }
    }

    fn forward<C: ForwardCtx>(
        model: &Self::Model,
        ctx: &mut C,
        tokens: &[u32],
        positions: &[usize],
        slot_ids: &[usize],
    ) -> Result<()> {
        forward(model, ctx, tokens, positions, slot_ids)
    }

    fn scratch_config(
        model: &Self::Model,
        shard: ShardMode,
        prefill_ubatch: usize,
        max_slots: usize,
    ) -> ScratchConfig {
        let cfg = &model.config;
        let n_ranks = shard.n_ranks();
        let max_seq_len = cfg.context_length;
        let local_gdn = if n_ranks > 1 {
            per_rank_gdn_dims(cfg.gdn, n_ranks)
        } else {
            cfg.gdn
        };
        ScratchConfig {
            hidden: cfg.hidden,
            intermediate: cfg.expert_intermediate / n_ranks,
            q_width: (cfg.n_heads / n_ranks) * cfg.head_dim,
            kv_width: (cfg.n_kv_heads / n_ranks) * cfg.head_dim,
            vocab: cfg.vocab_size,
            max_seq_len,
            num_layers: cfg.num_layers,
            max_experts: cfg.num_experts,
            max_experts_per_tok: cfg.experts_per_tok,
            gdn: Some(local_gdn),
            per_layer_kv_widths: Some(per_layer_kv_widths(cfg, n_ranks)),
            attn_q_gated: true,
            shared_intermediate: cfg.shared_expert_intermediate,
            max_prefill_tokens: prefill_ubatch,
            max_slots,
            per_layer_embd: 0,
        }
    }

    fn dispose(model: &mut Self::Model, device: &HipDevice) -> Result<()> {
        model.dispose(device)
    }
}
