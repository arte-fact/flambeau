//! Shared composite engine. Each `ForwardCtx` impl is a thin wrapper:
//! state in `CoreState`, hooks via `TopologyHooks`, composite bodies
//! as free functions in `composites/`.

pub mod composites;
pub mod hooks;
pub mod scratch;
pub mod state;

pub use hooks::{NoopHooks, TopologyHooks};
pub use scratch::{
    per_layer_kv_widths, scratch_config_for, KvLayerShape, KvLayout, MoeShape, ScratchConfig,
    ScratchPool, ScratchShape, Q8_0_BLOCK_BYTES,
};
pub use state::CoreState;
