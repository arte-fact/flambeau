//! Topology executor. A model is `fn forward<C: ForwardCtx>(...)`,
//! monomorphised per topology. See `README.md` + `CLAUDE.md`.

#![cfg(feature = "hip")]

pub mod core;
pub mod ctx;
pub mod engine;
pub mod loader;
pub mod per_layer_embd;
pub mod runtime;

#[cfg(test)]
pub mod testing;

pub use core::{
    per_layer_kv_depths, per_layer_kv_widths, scratch_config_for, KvLayerShape, KvLayout, MoeShape,
    NoopHooks, ScratchConfig, ScratchPool, ScratchShape, TopologyHooks, Q8_0_BLOCK_BYTES,
};
pub use ctx::ForwardCtx;
pub use engine::{
    ForwardEngine, HybStage, HybridEngine, HybridForwardCtx, HybridHooks, PpEngine, PpForwardCtx,
    PpStage, SingleDeviceEngine, SingleDeviceForwardCtx, SoloStage, StageHooks, TpEngine,
    TpForwardCtx, TpHooks,
};
pub use per_layer_embd::{
    PerLayerEmbedBlock, PerLayerEmbedDecodeScratch, PerLayerEmbedLayerWeights,
};
pub use runtime::orchestrate::LaunchParams;
pub use runtime::{Arch, Session, Topology};
