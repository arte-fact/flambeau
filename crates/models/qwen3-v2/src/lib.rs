//! Qwen3 dense-attention thin model (v2 stack).
//!
//! Implements the qwen3 GGUF architecture (pure dense — full-attention
//! every layer, dense FFN, SwiGLU activation, optional Q/K norm,
//! optional partial NeoX RoPE). The model crate's job is small:
//!
//! 1. Parse the GGUF metadata into a `Qwen3V2Config`.
//! 2. Upload each tensor to device, wrap in the right `QuantWeight` /
//!    `Tensor<F16>`.
//! 3. Provide `forward_one_token<C: ForwardCtx>` that walks the
//!    compose vocabulary in the right order.
//!
//! Topology is the executor's concern (`flambeau-forward`); this crate
//! is topology-agnostic.

#![cfg(feature = "hip")]

pub mod config;
pub mod loader;
pub mod model;
pub mod tp_shard;

pub use config::{Qwen3V2Config, Qwen3V2ConfigError};
pub use loader::{load_from_gguf, Qwen3V2Model};
pub use model::forward_one_token;
pub use tp_shard::load_tp_shard_from_gguf;
