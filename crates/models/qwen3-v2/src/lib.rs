//! qwen3 dense arch (v2 stack): full-attention every layer, dense
//! FFN, SwiGLU, optional Q/K norm + partial NeoX RoPE. Topology lives
//! in `flambeau-forward`.

#![cfg(feature = "hip")]

pub mod arch;
pub mod config;
pub mod loader;
pub mod model;

pub use arch::Qwen3V2;
pub use config::{Qwen3V2Config, Qwen3V2ConfigError};
pub use loader::{load_from_gguf, load_tp_shard_from_gguf, Qwen3V2Model};
pub use model::forward_one_token;
