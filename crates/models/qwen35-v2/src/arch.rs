//! `Arch` impl for `flambeau_forward::Session<Qwen35V2>`. qwen35-v2 is
//! hybrid (GDN + full-attn alternation) + dense-FFN. GDN TP mode is
//! auto-picked by `flambeau_forward::loader::gdn_tp_mode_for` from
//! the per-rank geometry: Qwen3.5-9B / Qwen3.6-35B-A3B (32/16 v/k →
//! n_rep=1) hit `KReplicated`; Qwen3.5/3.6-27B (24/16 → 1.5) hit
//! `FullShard`.

use anyhow::Result;
use flambeau_core::Device;
use flambeau_forward::ctx::{ForwardCtx, GdnDims};
use flambeau_forward::loader::{per_rank_gdn_dims, ShardMode};
use flambeau_forward::runtime::Arch;
use flambeau_forward::{scratch_config_for, KvLayout, MoeShape, ScratchConfig, ScratchShape};
use flambeau_quant::GgufFile;

use crate::config::Qwen35V2Config;
use crate::{forward, load_from_gguf, load_tp_shard_from_gguf, Qwen35V2Model};

pub struct Qwen35V2;

impl Arch for Qwen35V2 {
    type Model = Qwen35V2Model;

    fn arch_tag() -> &'static str {
        "qwen3"
    }

    fn load(
        file: &GgufFile,
        device: &impl Device,
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

    fn forward_mixed<C: ForwardCtx>(
        model: &Self::Model,
        ctx: &mut C,
        tokens: &[u32],
        positions: &[usize],
        slot_ids: &[usize],
        prefill_rows: usize,
    ) -> Result<()> {
        crate::model::forward_mixed(model, ctx, tokens, positions, slot_ids, prefill_rows)
    }

    fn scratch_config(
        model: &Self::Model,
        shard: ShardMode,
        prefill_ubatch: usize,
        max_slots: usize,
        paged_kv_pages: Option<usize>,
        kv_layout: KvLayout,
    ) -> ScratchConfig {
        scratch_config_for(
            &model.config,
            shard,
            prefill_ubatch,
            max_slots,
            paged_kv_pages,
            kv_layout,
        )
    }

    fn dispose(model: &mut Self::Model, device: &impl Device) -> Result<()> {
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
        Some(per_rank_gdn_dims(self.gdn, n_ranks))
    }
    fn attn_q_gated(&self) -> bool {
        true
    }
}
