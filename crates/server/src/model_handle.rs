//! HIP-bound model + session traits.
//!
//! `flambeau-runtime`'s `Model` trait is backend-agnostic (capability
//! metadata only — a runtime↔backend-hip dependency cycle prevents
//! the runtime from holding `HipCluster`). The HIP binding lives here
//! in the server crate, alongside the qwen3-moe types it dispatches.
//!
//! Two traits:
//! * `HipModel` — load-time entry. Owns weights + per-topology
//!   auxiliary state (`BarP2pAllReduce`, sub-cluster handles, optional
//!   MTP). Reports topology + config.
//! * `HipSession` — per-request handle. Owns KV caches + scratches and
//!   a back-reference to its parent `HipModel`. `prefill_logits` /
//!   `decode_logits` advance generation; `dispose` frees device buffers.
//!
//! Spec-decode and KV snapshot/restore are PP- / qwen3-moe-specific
//! and stay as free helpers in `model.rs` (extension-trait wiring lands
//! when a second model crate needs them).

#![cfg(feature = "hip")]

use std::sync::Arc;

use anyhow::Result;
use flambeau_backend_hip::HipCluster;
use flambeau_qwen3_moe::forward::ShardedForwardPrefillScratchTp;
use flambeau_qwen3_moe::session::KvLayout;
use flambeau_qwen3_moe::Qwen3MoEConfig;

use crate::model::{BoundaryCallback, Inflight, LoadedModel};

pub trait HipModel: Send + Sync + 'static {
    fn config(&self) -> &Qwen3MoEConfig;
    /// Topology label for handler metrics: `"pp"`, `"tp"`, `"pp+tp"`.
    fn topology(&self) -> &'static str;
}

pub trait HipSession: Send {
    /// Ingest a (chunked-as-needed) prompt and write the last
    /// position's logits into `logits_out`. `start_position` is the
    /// absolute position of `prompt_ids[0]` inside the original full
    /// prompt — `0` for a fresh request, `> 0` after a prefix-cache
    /// restore covering `[0..start_position)`. `tp_pool_prefill` (TP
    /// only) lets the caller share a pre-allocated scratch when it
    /// already holds the prefill serialiser. `on_boundary` (PP / TP
    /// only) fires after every non-final chunk for prefix-cache
    /// snapshotting.
    #[allow(clippy::too_many_arguments)]
    fn prefill_logits(
        &mut self,
        cluster: &HipCluster,
        prompt_ids: &[u32],
        start_position: usize,
        logits_out: &mut Vec<f32>,
        tp_pool_prefill: Option<&mut ShardedForwardPrefillScratchTp>,
        on_boundary: Option<BoundaryCallback<'_>>,
        prefill_ubatch: usize,
    ) -> Result<()>;

    fn decode_logits(
        &mut self,
        cluster: &HipCluster,
        token: u32,
        position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()>;

    /// Skip the F32-logits DtoH; logits remain on the head rank's
    /// device pointer for the GPU sampler to consume in place. TP and
    /// Hybrid only — PP returns an error.
    fn decode_keep_logits_on_device(
        &mut self,
        cluster: &HipCluster,
        token: u32,
        position: usize,
    ) -> Result<()>;

    fn reset_for_next_request(&mut self, cluster: &HipCluster) -> Result<()>;

    fn dispose(self: Box<Self>, cluster: &HipCluster) -> Result<()>;
}

/// Self-sufficient session: bundles an `Inflight` with an `Arc` back-
/// reference to its parent `LoadedModel`. Constructed via
/// [`create_hip_session`]; the server holds it as `Box<dyn HipSession>`
/// so call sites stop matching on the topology variant.
pub struct OwnedHipSession {
    pub model: Arc<LoadedModel>,
    pub inflight: Inflight,
}

/// Build a per-request session bound to `model`. `prefill_ubatch` sizes
/// the PP prefill scratch (ignored for TP/Hybrid); `kv_layout` selects
/// between F16 / Q8 / turbo-quant KV.
pub fn create_hip_session(
    model: Arc<LoadedModel>,
    cluster: &HipCluster,
    prefill_ubatch: usize,
    kv_layout: KvLayout,
) -> Result<Box<dyn HipSession>> {
    let inflight = Inflight::new(&model, cluster, prefill_ubatch, kv_layout)?;
    Ok(Box::new(OwnedHipSession { model, inflight }))
}

impl HipModel for LoadedModel {
    fn config(&self) -> &Qwen3MoEConfig {
        LoadedModel::config(self)
    }
    fn topology(&self) -> &'static str {
        LoadedModel::topology(self)
    }
}

impl HipSession for OwnedHipSession {
    fn prefill_logits(
        &mut self,
        cluster: &HipCluster,
        prompt_ids: &[u32],
        start_position: usize,
        logits_out: &mut Vec<f32>,
        tp_pool_prefill: Option<&mut ShardedForwardPrefillScratchTp>,
        on_boundary: Option<BoundaryCallback<'_>>,
        prefill_ubatch: usize,
    ) -> Result<()> {
        crate::model::prefill_logits(
            &self.model,
            cluster,
            &mut self.inflight,
            prompt_ids,
            start_position,
            logits_out,
            tp_pool_prefill,
            on_boundary,
            prefill_ubatch,
        )
    }

    fn decode_logits(
        &mut self,
        cluster: &HipCluster,
        token: u32,
        position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        crate::model::decode_logits(
            &self.model,
            cluster,
            &mut self.inflight,
            token,
            position,
            logits_out,
        )
    }

    fn decode_keep_logits_on_device(
        &mut self,
        cluster: &HipCluster,
        token: u32,
        position: usize,
    ) -> Result<()> {
        crate::model::decode_keep_logits_on_device(
            &self.model,
            cluster,
            &mut self.inflight,
            token,
            position,
        )
    }

    fn reset_for_next_request(&mut self, cluster: &HipCluster) -> Result<()> {
        self.inflight.reset_for_next_request(cluster, &self.model)
    }

    fn dispose(self: Box<Self>, cluster: &HipCluster) -> Result<()> {
        let OwnedHipSession { model, inflight } = *self;
        inflight.dispose(cluster, &model)
    }
}
