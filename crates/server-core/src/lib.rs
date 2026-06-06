//! Backend-neutral `Model` / `Session` / `SessionContext` traits and
//! the cross-arch `BatchSlot` struct used by the server's batched-decode
//! dispatcher.
//!
//! Arch crates (`flambeau-qwen3-moe`, `flambeau-gemma4`, future CUDA
//! backends) implement these traits without depending on
//! `flambeau-server` or each other. The concrete `ServerState` lives in
//! `flambeau-server` and implements `SessionContext` so per-arch
//! `Model::forward_decode_batched` impls receive the shared
//! infrastructure (cluster, inflight-pool size, arch-specific extras
//! via `Any` downcast) without back-referencing the server crate.

#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "hip")]
mod logits_sink;
#[cfg(feature = "hip")]
mod traits;

#[cfg(feature = "hip")]
pub use logits_sink::LogitsSink;
#[cfg(feature = "hip")]
pub use traits::{
    BatchSlot, MixedBatchDecodes, MixedBatchPrefill, Model, ReasoningMarkers, ReasoningStyle,
    Session, SessionContext,
};
