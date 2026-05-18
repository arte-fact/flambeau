//! `Arch` impl: routes `flambeau_forward::Session<Qwen3V2>` through
//! this crate's loader / forward / dispose.

use anyhow::Result;
use flambeau_backend_hip::HipDevice;
use flambeau_forward::ctx::ForwardCtx;
use flambeau_forward::loader::ShardMode;
use flambeau_forward::runtime::Arch;
use flambeau_forward::ScratchConfig;
use flambeau_quant::GgufFile;

use crate::{forward_one_token, load_from_gguf, load_tp_shard_from_gguf, Qwen3V2Model};

pub struct Qwen3V2;

impl Arch for Qwen3V2 {
    type Model = Qwen3V2Model;

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
        let n_ranks = match shard {
            ShardMode::Replicated => 1,
            ShardMode::Tp { n_ranks, .. } => n_ranks,
        };
        let max_seq_len = 64.min(cfg.context_length);
        ScratchConfig {
            hidden: cfg.hidden,
            intermediate: cfg.intermediate / n_ranks,
            q_width: (cfg.n_heads / n_ranks) * cfg.head_dim,
            kv_width: (cfg.n_kv_heads / n_ranks) * cfg.head_dim,
            vocab: cfg.vocab_size,
            max_seq_len,
            num_layers: cfg.num_layers,
            max_experts: 0,
            gdn: None,
        }
    }

    fn dispose(model: &mut Self::Model, device: &HipDevice) -> Result<()> {
        model.dispose(device)
    }
}
