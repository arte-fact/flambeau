//! `Arch` impl for `flambeau_forward::Session<Qwen35V2>`. qwen35-v2 is
//! hybrid (GDN + full-attn alternation) and dense-FFN; the v2 crate
//! currently only supports `ShardMode::Replicated` because TP-aware
//! GDN isn't wired yet.

use anyhow::{anyhow, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_forward::ctx::ForwardCtx;
use flambeau_forward::loader::ShardMode;
use flambeau_forward::runtime::Arch;
use flambeau_forward::ScratchConfig;
use flambeau_quant::GgufFile;

use crate::{forward_one_token, load_from_gguf, Qwen35V2Model};

pub struct Qwen35V2;

impl Arch for Qwen35V2 {
    type Model = Qwen35V2Model;

    fn arch_tag() -> &'static str {
        "qwen3"
    }

    fn load(file: &GgufFile, device: &HipDevice, shard: ShardMode) -> Result<Self::Model> {
        match shard {
            ShardMode::Replicated => load_from_gguf(file, device),
            ShardMode::Tp { .. } => Err(anyhow!(
                "qwen35-v2: TP loader not implemented yet (TP-aware GDN prerequisite)"
            )),
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

    fn scratch_config(model: &Self::Model, _shard: ShardMode) -> ScratchConfig {
        let cfg = &model.config;
        let max_seq_len = 64.min(cfg.context_length);
        ScratchConfig {
            hidden: cfg.hidden,
            intermediate: cfg.intermediate,
            q_width: cfg.n_heads * cfg.head_dim,
            kv_width: cfg.n_kv_heads * cfg.head_dim,
            vocab: cfg.vocab_size,
            max_seq_len,
            num_layers: cfg.num_layers,
            max_experts: 0,
            gdn: Some(cfg.gdn),
        }
    }

    fn dispose(model: &mut Self::Model, device: &HipDevice) -> Result<()> {
        model.dispose(device)
    }
}
