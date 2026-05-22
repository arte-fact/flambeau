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
use flambeau_forward::{scratch_config_for, MoeShape, ScratchConfig, ScratchShape};
use flambeau_quant::GgufFile;

use crate::config::Qwen35V2Config;
use crate::{forward, load_from_gguf, load_tp_shard_from_gguf, Qwen35V2Model};

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

    fn load(
        file: &GgufFile,
        device: &HipDevice,
        shard: ShardMode,
        layer_range: Option<(usize, usize)>,
        ctx_cap: Option<usize>,
    ) -> Result<Self::Model> {
        match shard {
            ShardMode::Replicated => load_from_gguf(file, device, layer_range, ctx_cap),
            ShardMode::Tp { rank, n_ranks } => {
                load_tp_shard_from_gguf(file, device, rank, n_ranks, layer_range, ctx_cap)
            }
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
        scratch_config_for(&model.config, shard, prefill_ubatch, max_slots)
    }

    fn dispose(model: &mut Self::Model, device: &HipDevice) -> Result<()> {
        model.dispose(device)
    }
}

impl ScratchShape for Qwen35V2Config {
    fn hidden(&self) -> usize {
        self.hidden
    }
    fn vocab(&self) -> usize {
        self.vocab_size
    }
    fn max_seq_len(&self) -> usize {
        self.context_length
    }
    fn intermediate_per_rank(&self, n_ranks: usize) -> usize {
        self.intermediate / n_ranks
    }
    fn q_width_per_rank(&self, n_ranks: usize) -> usize {
        (self.n_heads / n_ranks) * self.head_dim
    }
    fn moe_per_rank(&self, _n_ranks: usize) -> Option<MoeShape> {
        None
    }
    fn gdn_per_rank(&self, n_ranks: usize) -> Option<GdnDims> {
        Some(if n_ranks > 1 {
            per_rank_gdn_dims(self.gdn, n_ranks)
        } else {
            self.gdn
        })
    }
    fn attn_q_gated(&self) -> bool {
        true
    }
}
