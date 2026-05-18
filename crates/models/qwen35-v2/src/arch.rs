//! `Arch` impl for `flambeau_forward::Session<Qwen35V2>`. qwen35-v2 is
//! hybrid (GDN + full-attn alternation) + dense-FFN. TP uses
//! `GdnTpMode::KReplicated` ("rep_outer"): every rank holds the full
//! Q/K slabs of GDN's fused-QKV tensors, V heads shard.

use anyhow::Result;
use flambeau_backend_hip::HipDevice;
use flambeau_forward::ctx::{ForwardCtx, GdnDims};
use flambeau_forward::loader::ShardMode;
use flambeau_forward::runtime::Arch;
use flambeau_forward::ScratchConfig;
use flambeau_quant::GgufFile;

use crate::{
    forward_one_token, load_from_gguf, load_tp_shard_from_gguf, Qwen35V2Model,
};

pub struct Qwen35V2;

/// Per-rank GdnDims under KReplicated TP. V-heads shard;
/// K-heads stay global on every rank.
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

impl Arch for Qwen35V2 {
    type Model = Qwen35V2Model;

    fn arch_tag() -> &'static str {
        "qwen3"
    }

    fn load(file: &GgufFile, device: &HipDevice, shard: ShardMode) -> Result<Self::Model> {
        match shard {
            ShardMode::Replicated => load_from_gguf(file, device),
            ShardMode::Tp { rank, n_ranks } => {
                load_tp_shard_from_gguf(file, device, rank, n_ranks)
            }
        }
    }

    fn forward<C: ForwardCtx>(
        model: &Self::Model,
        ctx: &mut C,
        token: u32,
        position: usize,
    ) -> Result<()> {
        forward_one_token(model, ctx, token, position)
    }

    fn scratch_config(model: &Self::Model, shard: ShardMode) -> ScratchConfig {
        let cfg = &model.config;
        let n_ranks = shard.n_ranks();
        let max_seq_len = 64.min(cfg.context_length);
        let local_gdn = if n_ranks > 1 {
            per_rank_gdn_dims(cfg.gdn, n_ranks)
        } else {
            cfg.gdn
        };
        ScratchConfig {
            hidden: cfg.hidden,
            intermediate: cfg.intermediate / n_ranks,
            q_width: (cfg.n_heads / n_ranks) * cfg.head_dim,
            kv_width: (cfg.n_kv_heads / n_ranks) * cfg.head_dim,
            vocab: cfg.vocab_size,
            max_seq_len,
            num_layers: cfg.num_layers,
            max_experts: 0,
            gdn: Some(local_gdn),
            per_layer_kv_widths: None,
        }
    }

    fn dispose(model: &mut Self::Model, device: &HipDevice) -> Result<()> {
        model.dispose(device)
    }
}
