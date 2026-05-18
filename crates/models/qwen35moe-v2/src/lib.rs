//! qwen35moe MoE hybrid arch on the v2 stack. Covers Qwen3.5-MoE
//! and Qwen3.6-35B-A3B (GGUF tag `qwen35moe`).

#![cfg(feature = "hip")]

pub mod arch;
pub mod config;
pub mod loader;
pub mod model;

pub use arch::Qwen35MoeV2;
pub use config::{Qwen35MoeV2Config, Qwen35MoeV2ConfigError};
pub use loader::{load_from_gguf, Qwen35MoeV2Model};
pub use model::forward_one_token;
