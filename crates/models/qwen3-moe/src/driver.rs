//! Owned PP / TP driver wrappers that implement
//! [`flambeau_runtime::ModelDriver`].
//!
//! Qwen3-MoE's forward functions take `(model, session, cluster,
//! scratch)` tuples; nothing else owns those four together. The
//! server folds them into its own `LoadedModel` + `Inflight`
//! abstraction. For the CLI / parity-test / single-stream callers
//! that don't need the server's slot pool, this module bundles them
//! into one struct + exposes the trait.
//!
//! Disposal is split: the underlying qwen3-moe types take `self`
//! by value in their `dispose` methods (so they can move out their
//! tracked allocations). The trait wants `&mut self`, so the bundle
//! uses `Option<T>` slots and `Option::take`s them inside
//! `ModelDriver::dispose`.

#![cfg(feature = "hip")]

use anyhow::{anyhow, Result};
use flambeau_backend_hip::HipCluster;
use flambeau_quant::GgufFile;
use flambeau_runtime::{LayerAssignment, ModelDriver};

use crate::forward::pp::{
    forward_one_token_pp, forward_one_token_pp_logits, forward_prefill_pp,
    forward_prefill_pp_logits, ShardedForwardOneTokenScratch, ShardedForwardPrefillScratch,
};
use crate::session::KvLayout;
use crate::sharded::{Qwen3MoEShardedModel, Qwen3MoEShardedSession};

/// PP-sharded Qwen3-MoE driver — owns weights + KV + decode/prefill
/// scratches + cluster. Constructed by [`Self::load`]; consumed via
/// [`ModelDriver::dispose`] (preferred) or [`Drop`] (best-effort).
pub struct Qwen3MoEPpDriver {
    cluster: Option<HipCluster>,
    model: Option<Qwen3MoEShardedModel>,
    session: Option<Qwen3MoEShardedSession>,
    decode_scratch: Option<ShardedForwardOneTokenScratch>,
    prefill_scratch: Option<ShardedForwardPrefillScratch>,
}

impl Qwen3MoEPpDriver {
    /// Construct a single-stream PP driver from a GGUF + device list.
    /// Layers split contiguously across ranks. `max_tokens` sizes the
    /// prefill scratch's per-rank hidden buffer; pass the maximum
    /// prompt + decode length the caller expects.
    pub fn load(
        file: &GgufFile,
        device_ids: &[i32],
        max_tokens: usize,
        kv_layout: KvLayout,
    ) -> Result<Self> {
        let cluster = HipCluster::new(device_ids)?;
        let cfg = crate::config::Qwen3MoEConfig::from_gguf(file)?;
        let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
        let model = Qwen3MoEShardedModel::load(file, &cluster, &assignment)?;
        let session = Qwen3MoEShardedSession::new(&model, &cluster, kv_layout)?;
        let decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;
        let prefill_scratch = ShardedForwardPrefillScratch::new(&model, &cluster, max_tokens)?;
        Ok(Self {
            cluster: Some(cluster),
            model: Some(model),
            session: Some(session),
            decode_scratch: Some(decode_scratch),
            prefill_scratch: Some(prefill_scratch),
        })
    }
}

impl ModelDriver for Qwen3MoEPpDriver {
    fn forward_prefill(&mut self, tokens: &[u32], start_position: usize) -> Result<u32> {
        let model = self.model.as_ref().ok_or_else(|| anyhow!("driver disposed"))?;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("driver disposed"))?;
        let cluster = self
            .cluster
            .as_ref()
            .ok_or_else(|| anyhow!("driver disposed"))?;
        let scratch = self
            .prefill_scratch
            .as_mut()
            .ok_or_else(|| anyhow!("driver disposed"))?;
        forward_prefill_pp(model, session, cluster, scratch, tokens, start_position)
    }

    fn forward_one_token(&mut self, token_id: u32, position: usize) -> Result<u32> {
        let model = self.model.as_ref().ok_or_else(|| anyhow!("driver disposed"))?;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("driver disposed"))?;
        let cluster = self
            .cluster
            .as_ref()
            .ok_or_else(|| anyhow!("driver disposed"))?;
        let scratch = self
            .decode_scratch
            .as_mut()
            .ok_or_else(|| anyhow!("driver disposed"))?;
        forward_one_token_pp(model, session, cluster, scratch, token_id, position)
    }

    fn forward_prefill_logits(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        let model = self.model.as_ref().ok_or_else(|| anyhow!("driver disposed"))?;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("driver disposed"))?;
        let cluster = self
            .cluster
            .as_ref()
            .ok_or_else(|| anyhow!("driver disposed"))?;
        let scratch = self
            .prefill_scratch
            .as_mut()
            .ok_or_else(|| anyhow!("driver disposed"))?;
        forward_prefill_pp_logits(model, session, cluster, scratch, tokens, start_position, logits_out)
    }

    fn forward_one_token_logits(
        &mut self,
        token_id: u32,
        position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        let model = self.model.as_ref().ok_or_else(|| anyhow!("driver disposed"))?;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("driver disposed"))?;
        let cluster = self
            .cluster
            .as_ref()
            .ok_or_else(|| anyhow!("driver disposed"))?;
        let scratch = self
            .decode_scratch
            .as_mut()
            .ok_or_else(|| anyhow!("driver disposed"))?;
        forward_one_token_pp_logits(
            model, session, cluster, scratch, token_id, position, logits_out,
        )
    }

    fn vocab_size(&self) -> usize {
        self.model
            .as_ref()
            .map(|m| m.config.vocab_size)
            .unwrap_or(0)
    }

    fn dispose(&mut self) -> Result<()> {
        // Drop order matters: scratches first (they reference no
        // cluster/model state via Drop), then session (KV caches),
        // then model (weights), finally cluster.
        let cluster = self
            .cluster
            .as_ref()
            .ok_or_else(|| anyhow!("driver already disposed"))?;

        if let Some(s) = self.decode_scratch.take() {
            s.dispose(cluster)?;
        }
        if let Some(s) = self.prefill_scratch.take() {
            s.dispose(cluster)?;
        }
        if let Some(s) = self.session.take() {
            s.dispose(cluster)?;
        }
        if let Some(m) = self.model.take() {
            m.dispose(cluster)?;
        }
        let _ = self.cluster.take();
        Ok(())
    }
}

impl Drop for Qwen3MoEPpDriver {
    fn drop(&mut self) {
        if self.model.is_some() || self.session.is_some() {
            tracing::warn!("Qwen3MoEPpDriver dropped without dispose()");
        }
    }
}
