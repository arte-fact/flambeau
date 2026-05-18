//! Gemma-4 dense (31B-like variants) on the v2 stack. GELU-tanh,
//! final-logit softcap, tied LM head, post-embed scale = sqrt(n_embd),
//! per-layer SWA alternation, V-from-K (no attn_v on disk).
//!
//! MoE (26B-A4B) and per-layer-embd (E2B/E4B) variants are out of
//! scope for the first cut.

#![cfg(feature = "hip")]

pub mod config;
pub mod loader;
pub mod model;

pub use config::{Gemma4V2Config, Gemma4V2ConfigError};
pub use loader::{load_from_gguf, load_tp_shard_from_gguf, Gemma4V2Model};
pub use model::forward_one_token;
