//! Per-request shared state plumbed into composite free functions.
//! Fields are `pub` so composites read/write without accessor noise;
//! external crates see only the `ForwardCtx` trait surface.

use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_ops::{HipOps, OpsRegistry};

use super::ScratchPool;

pub struct CoreState<'a> {
    pub device: &'a HipDevice,
    pub stream: &'a HipStream,
    pub reg: &'a OpsRegistry,
    pub pool: &'a mut ScratchPool,
    pub logits_host: Vec<f32>,
    /// First global layer index this rank owns. Composites that index
    /// `pool.kv_caches` subtract this so the model can pass a global
    /// `layer_idx` and PP/Hybrid hit slot 0 of their owned-slice pool.
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
