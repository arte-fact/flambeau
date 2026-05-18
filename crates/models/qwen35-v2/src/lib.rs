//! qwen35 hybrid arch (GDN + full-attn alternating + dense FFN) on
//! the v2 stack. The MoE variants (qwen35moe / qwen36moe / qwen3next)
//! will land here once `moe_ffn` integrates with the per-layer
//! dispatch.

#![cfg(feature = "hip")]

pub mod arch;
pub mod config;
pub mod loader;
pub mod model;

pub use arch::Qwen35V2;
pub use config::{Qwen35V2Config, Qwen35V2ConfigError};
pub use loader::{load_from_gguf, load_tp_shard_from_gguf, Qwen35V2Model};
pub use model::forward;
