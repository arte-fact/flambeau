//! Gemma-4 dense (31B-like variants) on the v2 stack.
//!
//! Deltas vs qwen3 dense:
//! - GELU-tanh FFN activation (not SwiGLU)
//! - Final-logit softcap (typically 30.0)
//! - Tied LM head (token_embd reused as lm_head)
//! - `inpL = inpL * sqrt(n_embd)` post-embed scale
//! - Per-layer SWA alternation: some layers are local (`window_size > 0`)
//!   with separate `head_dim_swa` / `rope_theta_swa` / `rotated_dims_swa`;
//!   others are global full-attention.
//!
//! MoE (26B-A4B) and per-layer-embd (E2B/E4B) variants are out of
//! scope for the first cut.

#![cfg(feature = "hip")]

pub mod config;
pub mod loader;
pub mod model;

pub use config::{Gemma4V2Config, Gemma4V2ConfigError};
pub use loader::{load_from_gguf, Gemma4V2Model};
pub use model::forward_one_token;
