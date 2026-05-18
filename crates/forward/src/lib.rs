//! Topology executor. A model is `fn forward<C: ForwardCtx>(...)`,
//! monomorphised per topology. See `README.md` + `CLAUDE.md`.

#![cfg(feature = "hip")]

pub mod core;
pub mod ctx;
pub mod engine;
pub mod hybrid;
pub mod loader;
pub mod pp;
pub mod runtime;
pub mod single_device;
pub mod tp;

#[cfg(test)]
pub mod testing;

pub use core::{NoopHooks, ScratchConfig, ScratchPool, TopologyHooks};
pub use ctx::ForwardCtx;
pub use engine::{
    ForwardEngine, HybStage, HybridEngine, PpEngine, PpStage, SingleDeviceEngine, SoloStage,
    StageHooks, TpEngine,
};
pub use hybrid::HybridForwardCtx;
pub use pp::PpForwardCtx;
pub use runtime::{Arch, Session, Topology};
pub use single_device::SingleDeviceForwardCtx;
pub use tp::TpForwardCtx;
