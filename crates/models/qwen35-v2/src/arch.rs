//! `Arch` impl for `flambeau_forward::Session<Qwen35V2>`. qwen35-v2 is
//! hybrid (GDN + full-attn alternation) + dense-FFN. TP mode is
//! chosen by the per-rank geometry: prefer `KReplicated` (V-heads
//! shard, K-heads global) when the GDN kernel's `n_rep =
//! local_num_v_heads / num_k_heads` is a clean integer; otherwise
//! fall back to `FullShard` (both K and V heads shard). Qwen3.5-9B
//! (32 → 16 v-heads / 16 k-heads → n_rep=1) and Qwen3.6-35B-A3B
//! (same ratio) hit KReplicated; Qwen3.5/3.6-27B (48 → 24 v-heads /
//! 16 k-heads → 1.5) hits FullShard.

use anyhow::Result;
use flambeau_backend_hip::HipDevice;
use flambeau_forward::ctx::{ForwardCtx, GdnDims};
use flambeau_forward::loader::{GdnTpMode, ShardMode};
use flambeau_forward::runtime::Arch;
use flambeau_forward::ScratchConfig;
use flambeau_quant::GgufFile;

use crate::{
    forward_one_token, load_from_gguf, load_tp_shard_from_gguf, Qwen35V2Model,
};

pub struct Qwen35V2;

/// Pick the GDN TP mode based on the per-rank geometry.
pub(crate) fn gdn_tp_mode_for(g: GdnDims, n_ranks: usize) -> GdnTpMode {
    if n_ranks <= 1 {
        return GdnTpMode::KReplicated;
    }
    let local_v = g.num_v_heads / n_ranks;
    if local_v % g.num_k_heads == 0 {
        GdnTpMode::KReplicated
    } else {
        GdnTpMode::FullShard
    }
}

/// Per-rank GdnDims under TP. Math differs by mode: KReplicated
/// keeps K-heads global (only V shards); FullShard divides both.
pub(crate) fn per_rank_gdn_dims(g: GdnDims, n_ranks: usize) -> GdnDims {
    let mode = gdn_tp_mode_for(g, n_ranks);
    let num_v_heads_local = g.num_v_heads / n_ranks;
    let num_k_heads_local = match mode {
        GdnTpMode::KReplicated => g.num_k_heads,
        GdnTpMode::FullShard => g.num_k_heads / n_ranks,
    };
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
            attn_q_gated: true,
            shared_intermediate: 0,
        }
    }

    fn dispose(model: &mut Self::Model, device: &HipDevice) -> Result<()> {
        model.dispose(device)
    }
}
