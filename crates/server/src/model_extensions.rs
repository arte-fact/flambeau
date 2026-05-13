//! Host-side KV+GDN snapshot capability for the prefix cache.
//!
//! PP and TP capture against the server-owned global `HipCluster`;
//! Hybrid additionally needs its parent model so each stage's
//! sub-cluster is reachable. Hybrid impls require the typed model
//! (`HybridHipModel`) — the blanket trait doesn't carry it; use
//! [`HybridHipSession::capture_with_model`] /
//! [`HybridHipSession::restore_with_model`] for hybrid sessions, or
//! drive the dispatch through the server's `KvSnapshotExt` helpers.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipCluster;
use flambeau_qwen3_moe::session::{
    restore_layer_caches_from_host, snapshot_layer_caches_to_host, LayerCacheSnapshot,
};

use crate::model::{HybridHipModel, HybridHipSession, PpHipSession, TpHipSession};

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
