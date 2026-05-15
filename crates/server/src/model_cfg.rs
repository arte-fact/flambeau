//! Model-agnostic config slice held on [`crate::ServerState`].
//!
//! The server's routes / handlers only need a handful of fields from
//! the loaded model: `arch`, `vocab_size`, `context_length`, and
//! `num_layers`. The arch-specific config types (`Qwen3MoEConfig`,
//! `Gemma4Config`, future others) live in their respective model
//! crates and stay there — the boot path converts whichever it
//! loaded into this neutral struct before storing it on
//! `ServerState`.
//!
//! This breaks the direct dependency from `routes.rs` on
//! `flambeau_qwen3_moe::Qwen3MoEConfig`, which is the first step in
//! the routes.rs unification (Phase 12 follow-up). Subsequent steps
//! generalise `Inflight`'s qwen3-moe-typed variants and the forward-
//! call dispatch.

use flambeau_qwen3_moe::Qwen3MoEConfig;

/// Server-side view of the loaded model's identity + capacity. All
/// numeric fields are `usize` to match the model-crate field types;
/// callers that send these over HTTP are responsible for casting to
/// `u32`/`u64` as the schema requires.
#[derive(Debug, Clone)]
pub struct ServerModelCfg {
    pub arch: String,
    pub vocab_size: usize,
    pub context_length: usize,
    pub num_layers: usize,
}

impl From<&Qwen3MoEConfig> for ServerModelCfg {
    fn from(cfg: &Qwen3MoEConfig) -> Self {
        Self {
            arch: cfg.arch.clone(),
            vocab_size: cfg.vocab_size,
            context_length: cfg.context_length,
            num_layers: cfg.num_layers,
        }
    }
}

#[cfg(feature = "hip")]
impl From<&flambeau_gemma4::Gemma4Config> for ServerModelCfg {
    fn from(cfg: &flambeau_gemma4::Gemma4Config) -> Self {
        Self {
            arch: cfg.arch.clone(),
            vocab_size: cfg.vocab_size,
            context_length: cfg.context_length,
            num_layers: cfg.num_layers,
        }
    }
}
