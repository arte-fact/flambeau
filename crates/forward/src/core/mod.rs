//! Shared composite engine. Each `ForwardCtx` impl is a thin wrapper:
//! state in `CoreState`, hooks via `TopologyHooks`, composite bodies
//! as free functions in `composites/`.

pub mod composites;
pub mod hooks;
pub mod scratch;
pub mod state;

pub use hooks::{NoopHooks, TopologyHooks};
pub use scratch::{ScratchConfig, ScratchPool};
pub use state::CoreState;
