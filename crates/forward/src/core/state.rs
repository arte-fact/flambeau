//! Per-request shared state held by every topology impl.
//!
//! `CoreState` is plumbed by-mut-ref into the composite free functions
//! in `core::composites`. The fields are intentionally `pub` (visible
//! to other modules in the forward crate) so composites can read /
//! write them without accessor noise. External crates see the typed
//! `ForwardCtx` trait surface only.

use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_ops::{HipOps, OpsRegistry};

use super::ScratchPool;

/// Per-request shared state — borrowed by-mut-ref into composite
/// helpers.
pub struct CoreState<'a> {
    pub device: &'a HipDevice,
    pub stream: &'a HipStream,
    pub reg: &'a OpsRegistry,
    pub pool: &'a mut ScratchPool,
    /// Host-side logits buffer populated by `output_head`. Sized lazily
    /// on the first `output_head` call (or by the model crate ahead of
    /// time).
    pub logits_host: Vec<f32>,
    /// First global layer index this rank owns. Composites that index
    /// `pool.kv_caches` subtract this offset so the model can pass a
    /// global `layer_idx` and PP/Hybrid still hit slot 0 of their
    /// owned-slice pool. SingleDevice / TP set this to 0.
    pub layer_idx_offset: usize,
}

impl<'a> CoreState<'a> {
    pub fn new(
        device: &'a HipDevice,
        stream: &'a HipStream,
        reg: &'a OpsRegistry,
        pool: &'a mut ScratchPool,
    ) -> Self {
        Self {
            device,
            stream,
            reg,
            pool,
            logits_host: Vec::new(),
            layer_idx_offset: 0,
        }
    }

    pub fn ops(&self) -> HipOps<'a> {
        HipOps::new(self.reg, self.stream)
    }

    pub fn hidden(&self) -> usize {
        self.pool.config.hidden
    }
}
