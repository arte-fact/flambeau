//! Concrete `Model` + `Session` implementations for the qwen3-moe arch
//! topologies (`PpHipModel`, `TpHipModel`, `HybridHipModel`) and the
//! per-request `Qwen3MoeOwnedSession`.
//!
//! Trait defs (`Model`, `Session`, `SessionContext`, `BatchSlot`) live
//! in `flambeau-server-core` and are re-exported below for crate-local
//! callers.

#![cfg(feature = "hip")]

use std::any::Any;

use anyhow::Result;
use flambeau_backend_hip::HipCluster;
use flambeau_qwen3_moe::session::KvLayout;

pub use flambeau_server_core::{BatchSlot, Model, Session, SessionContext};

use crate::model::{HybridHipModel, Inflight, LoadedModel, PpHipModel, TpHipModel};

/// Self-sufficient qwen3-moe session: bundles an `Inflight`
/// (KV state + scratches) with a back-reference to its parent
/// `LoadedModel` and an `Arc<HipCluster>` clone so the trait methods
/// don't need a cluster passed in.
pub struct Qwen3MoeOwnedSession {
    pub model: LoadedModel,
    pub inflight: Inflight,
    pub cluster: std::sync::Arc<HipCluster>,
}

/// Build a per-request session bound to `model`.
pub fn create_qwen3moe_session(
    model: LoadedModel,
    cluster: std::sync::Arc<HipCluster>,
    prefill_ubatch: usize,
    kv_layout: KvLayout,
) -> Result<Box<dyn Session>> {
    let inflight = Inflight::new(&model, &cluster, prefill_ubatch, kv_layout)?;
    Ok(Box::new(Qwen3MoeOwnedSession {
        model,
        inflight,
        cluster,
    }))
}

impl Model for PpHipModel {
    fn topology(&self) -> &'static str {
        "pp"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn supports_scheduler_batching(&self) -> bool {
        true
    }
    fn forward_decode_batched(
        &self,
        ctx: &dyn SessionContext,
        inflights: &mut [&mut dyn Session],
        slots: &[BatchSlot],
        logits_refs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        crate::model::qwen3moe_forward_decode_batched(self, ctx, inflights, slots, logits_refs)
    }
}

impl Model for TpHipModel {
    fn topology(&self) -> &'static str {
        "tp"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn supports_scheduler_batching(&self) -> bool {
        true
    }
    fn requires_prefill_serialiser(&self) -> bool {
        true
    }
    fn requires_tp_prefill_scratch(&self) -> bool {
        true
    }
    fn forward_decode_batched(
        &self,
        ctx: &dyn SessionContext,
        inflights: &mut [&mut dyn Session],
        slots: &[BatchSlot],
        logits_refs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        crate::model::qwen3moe_forward_decode_batched(self, ctx, inflights, slots, logits_refs)
    }
}

impl Model for HybridHipModel {
    fn topology(&self) -> &'static str {
        "pp+tp"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn supports_scheduler_batching(&self) -> bool {
        true
    }
    fn requires_prefill_serialiser(&self) -> bool {
        true
    }
    fn forward_decode_batched(
        &self,
        ctx: &dyn SessionContext,
        inflights: &mut [&mut dyn Session],
        slots: &[BatchSlot],
        logits_refs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        crate::model::qwen3moe_forward_decode_batched(self, ctx, inflights, slots, logits_refs)
    }
}

impl Session for Qwen3MoeOwnedSession {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn reset_for_next_request(&mut self) -> Result<()> {
        let cluster = self.cluster.clone();
        self.inflight.reset_for_next_request(&cluster, &self.model)
    }

    fn dispose(self: Box<Self>) -> Result<()> {
        let Qwen3MoeOwnedSession {
            model,
            inflight,
            cluster,
        } = *self;
        inflight.dispose(&cluster, &model)
    }
}
