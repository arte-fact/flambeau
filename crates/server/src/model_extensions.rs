//! Per-capability extension traits over `HipModel` / `HipSession`.
//!
//! Some capabilities are model-specific:
//!
//! * **Spec-decode** is PP-only (greedy strict-match + rejection-sampling
//!   variants), and only when an MTP head is attached at boot. Future
//!   model crates without an MTP head do not implement `SpecDecodeModel`.
//! * **KV snapshot / restore** is the prefix-cache hook. PP and TP
//!   serialise per-rank caches against the global cluster; Hybrid walks
//!   per-stage sub-clusters. All three topologies support it; future
//!   models that ship without a recurrent / attention KV cache (none
//!   today) would not.
//!
//! Each trait is opt-in via a downcast on the parent `HipModel` /
//! `HipSession` trait — `model.as_spec_decode()` returns `None` for
//! topologies / configurations that don't support it. The server reads
//! the option and routes accordingly without matching on the topology
//! enum.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipCluster;
use flambeau_qwen3_moe::forward::{
    forward_speculative_pp_step, forward_speculative_pp_step_sampling, SpecStep,
};
use flambeau_qwen3_moe::session::{
    restore_layer_caches_from_host, snapshot_layer_caches_to_host, LayerCacheSnapshot,
};

use crate::model::{
    HybridHipModel, HybridHipSession, PpHipModel, PpHipSession, SpecDecodePp, TpHipSession,
};

/// PP + MTP capability. Models that load without an MTP head (everything
/// today except `flambeau-server` with `--spec-mtp`) return `None` from
/// `HipModel::as_spec_decode`; this trait is otherwise unreachable.
pub trait SpecDecodeModel: Send + Sync {
    /// Allocate per-request spec state — the MTP forward scratch + a
    /// placeholder for `h_for_mtp` that the caller sets after the
    /// first base prefill writes its hidden output to the last rank.
    fn create_spec_state(&self, cluster: &HipCluster) -> Result<SpecDecodePp>;

    /// One K=1 strict-match (greedy) macro step. Returns committed
    /// tokens + telemetry; updates `spec.h_for_mtp` in place.
    fn decode_spec_greedy(
        &self,
        cluster: &HipCluster,
        session: &mut PpHipSession,
        spec: &mut SpecDecodePp,
        last_token: u32,
        position: usize,
    ) -> Result<SpecStep>;

    /// One vLLM-canonical rejection-sampling macro step.
    #[allow(clippy::too_many_arguments)]
    fn decode_spec_sampling(
        &self,
        cluster: &HipCluster,
        session: &mut PpHipSession,
        spec: &mut SpecDecodePp,
        last_token: u32,
        position: usize,
        sampling: &flambeau_runtime::Sampling,
        rng: &mut flambeau_runtime::Rng,
        history: &[u32],
    ) -> Result<SpecStep>;
}

impl SpecDecodeModel for PpHipModel {
    fn create_spec_state(&self, cluster: &HipCluster) -> Result<SpecDecodePp> {
        SpecDecodePp::new(&self.model, cluster).context("alloc SpecDecodePp")
    }

    fn decode_spec_greedy(
        &self,
        cluster: &HipCluster,
        session: &mut PpHipSession,
        spec: &mut SpecDecodePp,
        last_token: u32,
        position: usize,
    ) -> Result<SpecStep> {
        let mtp = self
            .mtp
            .as_ref()
            .context("decode_spec_greedy: PpHipModel has no MTP attachment (FLAMBEAU_SPEC_MTP not set)")?;
        let last_rank = cluster.ranks() - 1;
        let last_shard = &self.model.shards[last_rank];
        let output_norm = last_shard
            .output_norm
            .as_ref()
            .context("PP last rank missing output_norm for spec-decode")?;
        let lm_head = last_shard
            .output
            .as_ref()
            .context("PP last rank missing lm_head for spec-decode")?;

        let (step, h_next) = forward_speculative_pp_step(
            &self.model,
            &mut session.session,
            cluster,
            &mut session.decode,
            &mut session.prefill,
            mtp,
            &spec.mtp_scratch,
            output_norm,
            lm_head,
            last_token,
            spec.h_for_mtp,
            position,
        )?;
        spec.h_for_mtp = h_next;
        Ok(step)
    }

    fn decode_spec_sampling(
        &self,
        cluster: &HipCluster,
        session: &mut PpHipSession,
        spec: &mut SpecDecodePp,
        last_token: u32,
        position: usize,
        sampling: &flambeau_runtime::Sampling,
        rng: &mut flambeau_runtime::Rng,
        history: &[u32],
    ) -> Result<SpecStep> {
        let mtp = self
            .mtp
            .as_ref()
            .context("decode_spec_sampling: PpHipModel has no MTP attachment")?;
        let last_rank = cluster.ranks() - 1;
        let last_shard = &self.model.shards[last_rank];
        let output_norm = last_shard
            .output_norm
            .as_ref()
            .context("PP last rank missing output_norm for spec-decode")?;
        let lm_head = last_shard
            .output
            .as_ref()
            .context("PP last rank missing lm_head for spec-decode")?;

        let (step, h_next) = forward_speculative_pp_step_sampling(
            &self.model,
            &mut session.session,
            cluster,
            &mut session.decode,
            &mut session.prefill,
            mtp,
            &spec.mtp_scratch,
            output_norm,
            lm_head,
            last_token,
            spec.h_for_mtp,
            position,
            sampling,
            rng,
            history,
        )?;
        spec.h_for_mtp = h_next;
        Ok(step)
    }
}

/// Host-side KV+GDN snapshot capability for the prefix cache.
///
/// PP and TP capture against the server-owned global `HipCluster`;
/// Hybrid additionally needs its parent model so each stage's
/// sub-cluster is reachable. Hybrid impls require the typed model
/// (`HybridHipModel`) — the trait method takes it as `&dyn HipModel`
/// and downcasts via `Any` so the call site can carry a single
/// model handle.
pub trait KvSnapshot {
    fn capture(&self, cluster: &HipCluster) -> Result<Vec<Vec<LayerCacheSnapshot>>>;

    fn restore(
        &mut self,
        cluster: &HipCluster,
        snapshot: &[Vec<LayerCacheSnapshot>],
    ) -> Result<()>;
}

impl KvSnapshot for PpHipSession {
    fn capture(&self, cluster: &HipCluster) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
        let mut out = Vec::with_capacity(self.session.per_rank.len());
        for (rank_idx, rank) in self.session.per_rank.iter().enumerate() {
            let device = cluster.device(rank_idx);
            let s = snapshot_layer_caches_to_host(&rank.caches, device)
                .with_context(|| format!("PP snapshot rank {rank_idx}"))?;
            out.push(s);
        }
        Ok(out)
    }

    fn restore(
        &mut self,
        cluster: &HipCluster,
        snapshot: &[Vec<LayerCacheSnapshot>],
    ) -> Result<()> {
        if snapshot.len() != self.session.per_rank.len() {
            bail!(
                "PP restore: snapshot rank count {} != session ranks {}",
                snapshot.len(),
                self.session.per_rank.len()
            );
        }
        for (rank_idx, rank_session) in self.session.per_rank.iter_mut().enumerate() {
            let device = cluster.device(rank_idx);
            restore_layer_caches_from_host(&snapshot[rank_idx], rank_session.caches_mut(), device)
                .with_context(|| format!("PP restore rank {rank_idx}"))?;
        }
        Ok(())
    }
}

impl KvSnapshot for TpHipSession {
    fn capture(&self, cluster: &HipCluster) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
        let mut out = Vec::with_capacity(self.session.caches.len());
        for (rank_idx, c) in self.session.caches.iter().enumerate() {
            let device = cluster.device(rank_idx);
            let s = snapshot_layer_caches_to_host(c, device)
                .with_context(|| format!("TP snapshot rank {rank_idx}"))?;
            out.push(s);
        }
        Ok(out)
    }

    fn restore(
        &mut self,
        cluster: &HipCluster,
        snapshot: &[Vec<LayerCacheSnapshot>],
    ) -> Result<()> {
        if snapshot.len() != self.session.caches.len() {
            bail!(
                "TP restore: snapshot rank count {} != session ranks {}",
                snapshot.len(),
                self.session.caches.len()
            );
        }
        for (rank_idx, rank_caches) in self.session.caches.iter_mut().enumerate() {
            let device = cluster.device(rank_idx);
            restore_layer_caches_from_host(&snapshot[rank_idx], rank_caches, device)
                .with_context(|| format!("TP restore rank {rank_idx}"))?;
        }
        Ok(())
    }
}

/// Hybrid capture/restore needs the parent `HybridHipModel` for its
/// per-stage `sub_cluster` handles. The blanket trait doesn't carry
/// it; use [`HybridHipSession::capture_with_model`] /
/// [`HybridHipSession::restore_with_model`] for hybrid sessions, or
/// drive the dispatch through the server's `KvSnapshotExt` helpers.
impl HybridHipSession {
    pub fn capture_with_model(
        &self,
        model: &HybridHipModel,
    ) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
        let mut out: Vec<Vec<LayerCacheSnapshot>> = Vec::with_capacity(
            self.session
                .stages
                .iter()
                .map(|s| s.caches.len())
                .sum::<usize>(),
        );
        for (stage_idx, stage_session) in self.session.stages.iter().enumerate() {
            let stage_model = model.model.stages.get(stage_idx).ok_or_else(|| {
                anyhow::anyhow!("hybrid snapshot: stage {stage_idx} missing in model")
            })?;
            for (tp_rank, layer_caches) in stage_session.caches.iter().enumerate() {
                let device = stage_model.sub_cluster.device(tp_rank);
                let s = snapshot_layer_caches_to_host(layer_caches, device)
                    .with_context(|| {
                        format!("Hybrid snapshot stage {stage_idx} rank {tp_rank}")
                    })?;
                out.push(s);
            }
        }
        Ok(out)
    }

    pub fn restore_with_model(
        &mut self,
        model: &HybridHipModel,
        snapshot: &[Vec<LayerCacheSnapshot>],
    ) -> Result<()> {
        let total_ranks: usize = self.session.stages.iter().map(|s| s.caches.len()).sum();
        if snapshot.len() != total_ranks {
            bail!(
                "Hybrid restore: snapshot global-rank count {} != session total ranks {}",
                snapshot.len(),
                total_ranks
            );
        }
        let mut g = 0usize;
        for (stage_idx, stage_session) in self.session.stages.iter_mut().enumerate() {
            let stage_model = model.model.stages.get(stage_idx).ok_or_else(|| {
                anyhow::anyhow!("hybrid restore: stage {stage_idx} missing in model")
            })?;
            for (tp_rank, layer_caches) in stage_session.caches.iter_mut().enumerate() {
                let device = stage_model.sub_cluster.device(tp_rank);
                restore_layer_caches_from_host(&snapshot[g], layer_caches, device)
                    .with_context(|| {
                        format!("Hybrid restore stage {stage_idx} rank {tp_rank}")
                    })?;
                g += 1;
            }
        }
        Ok(())
    }
}

