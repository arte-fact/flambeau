//! Topology executor. A model is `fn forward<C: ForwardCtx>(...)`,
//! monomorphised per topology. See `README.md` + `CLAUDE.md`.

#![cfg(feature = "hip")]

pub mod core;
pub mod ctx;
pub mod engine;
pub mod loader;
pub mod runtime;

#[cfg(test)]
pub mod testing;

pub use core::{NoopHooks, ScratchConfig, ScratchPool, TopologyHooks};
pub use ctx::ForwardCtx;
pub use engine::{
    ForwardEngine, HybStage, HybridEngine, HybridForwardCtx, HybridHooks, PpEngine, PpStage,
    PpForwardCtx, SingleDeviceEngine, SingleDeviceForwardCtx, SoloStage, StageHooks, TpEngine,
    TpForwardCtx, TpHooks,
};
pub use runtime::{Arch, Session, Topology};
