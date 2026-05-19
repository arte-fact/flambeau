//! `Arch` impl for `flambeau_forward::Session<Gemma4V2>`. Per-layer
//! head_dim alternation (SWA vs global) makes `q_width`/`kv_width`
//! the per-layer max; a follow-up phase swaps the pool to per-layer
//! KV cache sizing.

use anyhow::Result;
use flambeau_backend_hip::HipDevice;
use flambeau_forward::ctx::ForwardCtx;
use flambeau_forward::loader::ShardMode;
use flambeau_forward::runtime::Arch;
use flambeau_forward::ScratchConfig;
use flambeau_quant::GgufFile;

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
        let cfg = &model.config;
        let n_ranks = match shard {
            ShardMode::Replicated => 1,
            ShardMode::Tp { n_ranks, .. } => n_ranks,
        };
        let max_seq_len = cfg.context_length;
        let per_layer_kv: Vec<usize> = cfg
            .attn
            .iter()
            .zip(cfg.num_kv_heads.iter())
            .map(|(a, &nkv)| (nkv / n_ranks) * a.head_dim)
            .collect();
        let q_width = cfg
            .attn
            .iter()
            .map(|a| (cfg.num_heads / n_ranks) * a.head_dim)
            .max()
            .unwrap_or(0);
        let kv_width = per_layer_kv.iter().copied().max().unwrap_or(0);
        // MoE variants size the pool's expert / shared scratch from
        // the GGUF; dense variants leave them at zero.
        let (max_experts, max_experts_per_tok, moe_intermediate) = match cfg.moe {
            Some(m) => (m.num_experts, m.experts_per_tok, m.moe_intermediate / n_ranks),
            None => (0, 0, 0),
        };
        // Shared-MLP scratch uses the dense intermediate width per
        // rank — same shape as a row-parallel SharedExpert. Dense
        // variants leave this at zero.
        let shared_intermediate = if cfg.moe.is_some() {
            cfg.intermediate / n_ranks
        } else {
            0
        };
        // The MoE moe_intermediate replaces the dense per-rank
        // intermediate for routed-expert scratch sizing. The shared
        // MLP path still uses the dense intermediate via
        // `shared_intermediate`.
        let intermediate = if cfg.moe.is_some() {
            moe_intermediate
        } else {
            cfg.intermediate / n_ranks
        };
        ScratchConfig {
            hidden: cfg.hidden,
            intermediate,
            q_width,
            kv_width,
            vocab: cfg.vocab_size,
            max_seq_len,
            num_layers: cfg.num_layers,
            max_experts,
            max_experts_per_tok,
            gdn: None,
            per_layer_kv_widths: Some(per_layer_kv),
            attn_q_gated: false,
            shared_intermediate,
            max_prefill_tokens: prefill_ubatch,
            max_slots,
        }
    }

    fn dispose(model: &mut Self::Model, device: &HipDevice) -> Result<()> {
        model.dispose(device)
    }
}
