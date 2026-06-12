//! Per-request shared state plumbed into composite free functions.
//! Fields are `pub` so composites read/write without accessor noise;
//! external crates see only the `ForwardCtx` trait surface.

use flambeau_backend::{Backend, HipBackend};

use super::ScratchPool;

pub struct CoreState<'a, B: Backend = HipBackend> {
    pub device: &'a B::Device,
    pub stream: &'a B::Stream,
    pub reg: &'a B::Registry,
    pub pool: &'a mut ScratchPool,
    pub logits_host: Vec<f32>,
    /// First global layer index this rank owns. Composites that index
    /// `pool.kv_caches` subtract this so the model can pass a global
    /// `layer_idx` and PP/Hybrid hit slot 0 of their owned-slice pool.
    pub layer_idx_offset: usize,
}

impl<'a, B: Backend> CoreState<'a, B> {
    pub fn new(
        device: &'a B::Device,
        stream: &'a B::Stream,
        reg: &'a B::Registry,
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

    pub fn ops(&self) -> B::Ops<'a> {
        B::ops(self.reg, self.stream)
    }

    pub fn hidden(&self) -> usize {
        self.pool.config.hidden
    }
}
