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

use crate::{forward_one_token, load_from_gguf, load_tp_shard_from_gguf, Gemma4V2Model};

pub struct Gemma4V2;

impl Arch for Gemma4V2 {
    type Model = Gemma4V2Model;

    fn arch_tag() -> &'static str {
        "gemma3"
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
        let n_ranks = match shard {
            ShardMode::Replicated => 1,
            ShardMode::Tp { n_ranks, .. } => n_ranks,
        };
        let max_seq_len = 64.min(cfg.context_length);
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
        ScratchConfig {
            hidden: cfg.hidden,
            intermediate: cfg.intermediate / n_ranks,
            q_width,
            kv_width,
            vocab: cfg.vocab_size,
            max_seq_len,
            num_layers: cfg.num_layers,
            max_experts: 0,
            gdn: None,
            per_layer_kv_widths: Some(per_layer_kv),
        }
    }

    fn dispose(model: &mut Self::Model, device: &HipDevice) -> Result<()> {
        model.dispose(device)
    }
}
