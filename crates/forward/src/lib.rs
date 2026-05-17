//! Topology executor. A model is `fn forward<C: ForwardCtx>(...)`,
//! monomorphised per topology. See `README.md` + `CLAUDE.md`.

#![cfg(feature = "hip")]

pub mod core;
pub mod ctx;
pub mod hybrid;
pub mod layer_range;
pub mod loader;
pub mod pp;
pub mod single_device;
pub mod tp;

#[cfg(test)]
pub mod testing;

pub use core::{NoopHooks, ScratchConfig, ScratchPool, TopologyHooks};
pub use ctx::ForwardCtx;
pub use hybrid::HybridForwardCtx;
pub use pp::PpForwardCtx;
pub use single_device::SingleDeviceForwardCtx;
pub use tp::TpForwardCtx;
