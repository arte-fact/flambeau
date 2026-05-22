//! `Arch` impl for `flambeau_forward::Session<Gemma4V2>`. Per-layer
//! head_dim alternation (SWA vs global) makes `q_width`/`kv_width`
//! the per-layer max; a follow-up phase swaps the pool to per-layer
//! KV cache sizing.

use anyhow::Result;
use flambeau_backend_hip::HipDevice;
use flambeau_forward::ctx::{ForwardCtx, GdnDims};
use flambeau_forward::loader::ShardMode;
use flambeau_forward::runtime::Arch;
use flambeau_forward::{scratch_config_for, MoeShape, ScratchConfig, ScratchShape};
use flambeau_quant::GgufFile;

use crate::config::Gemma4V2Config;
use crate::{forward, load_from_gguf, load_tp_shard_from_gguf, Gemma4V2Model};

pub struct Gemma4V2;

impl Arch for Gemma4V2 {
    type Model = Gemma4V2Model;

    fn arch_tag() -> &'static str {
        "gemma3"
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

impl ScratchShape for Gemma4V2Config {
    fn hidden(&self) -> usize {
        self.hidden
    }
    fn vocab(&self) -> usize {
        self.vocab_size
    }
    fn max_seq_len(&self) -> usize {
        self.context_length
    }
    // MoE variants route per-token compute through the experts with
    // `moe_intermediate`; the shared dense MLP uses the dense
    // `intermediate`. Builder picks the routed width here and the
    // shared width via `moe().shared_intermediate_per_rank`.
    fn intermediate_per_rank(&self, n_ranks: usize) -> usize {
        match self.moe {
            Some(m) => m.moe_intermediate / n_ranks,
            None => self.intermediate / n_ranks,
        }
    }
    fn q_width_per_rank(&self, n_ranks: usize) -> usize {
        self.attn
            .iter()
            .map(|a| (self.num_heads / n_ranks) * a.head_dim)
            .max()
            .unwrap_or(0)
    }
    fn moe_per_rank(&self, n_ranks: usize) -> Option<MoeShape> {
        self.moe.map(|m| MoeShape {
            num_experts: m.num_experts,
            experts_per_tok: m.experts_per_tok,
            // gemma4's shared MLP runs in parallel with the routed
            // experts and uses the dense `intermediate` width per rank.
            shared_intermediate_per_rank: self.intermediate / n_ranks,
        })
    }
    fn gdn_per_rank(&self, _n_ranks: usize) -> Option<GdnDims> {
        None
    }
    fn attn_q_gated(&self) -> bool {
        false
    }
    fn per_layer_embd(&self) -> usize {
        self.per_layer_embd.map_or(0, |p| p.pe)
    }
}
