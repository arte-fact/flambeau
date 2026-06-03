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

/// Per-token-batch carrier for composites that take aligned
/// `positions[i]` ↔ `slot_ids[i]` slices.
#[derive(Copy, Clone, Debug)]
pub struct TokenBatch<'a> {
    pub positions: &'a [usize],
    pub slot_ids: &'a [usize],
}

/// Mixed prefill+decode batch carrier — first `prefill_rows` entries are
/// the prefill rows (all share slot_ids[0]); the remaining are per-slot
/// decode rows.
#[derive(Copy, Clone, Debug)]
pub struct MixedBatch<'a> {
    pub positions: &'a [usize],
    pub slot_ids: &'a [usize],
    pub prefill_rows: usize,
}

/// GDN mixed-batch variant — no positions because GDN doesn't carry
/// per-token positions through the recurrent step.
#[derive(Copy, Clone, Debug)]
pub struct GdnMixedBatch<'a> {
    pub slot_ids: &'a [usize],
    pub prefill_rows: usize,
}
