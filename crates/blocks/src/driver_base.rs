//! Driver scaffolding shared by model crates.
//!
//! `TpCluster` and `HybridCluster` wrap the `HipCluster` +
//! `BarP2pAllReduce` construction ritual that every TP / pp+tp driver
//! re-derives by hand. They also encode the sub-cluster-before-global
//! construction ordering for hybrid as a type invariant (the gotcha
//! captured in `project_hybrid_cluster_order`: if the per-stage
//! sub-cluster + AR is built AFTER the global cluster, BAR1 probes
//! report all-zero off-diagonals and `peer_copy_via_host` silently
//! reads stale bytes).
//!
//! Arch drivers embed these and call `.cluster()` / `.ar()` /
//! `.stage(s)` accessors rather than carrying separate
//! `Arc<HipCluster>` + `BarP2pAllReduce` fields.

#![cfg(feature = "hip")]

use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use flambeau_backend_hip::{BarP2pAllReduce, HipCluster};

/// TP cluster: `HipCluster` over `tp_size` GPUs + a single
/// `BarP2pAllReduce` plumbed against the same cluster.
pub struct TpCluster {
    cluster: Arc<HipCluster>,
    ar: BarP2pAllReduce,
}

impl TpCluster {
    /// Build from an existing `HipCluster`. The cluster is consumed
    /// into an `Arc`; callers wanting shared ownership of the original
    /// can wrap it themselves and pass the `Arc` via [`Self::from_arc`].
    pub fn new(cluster: HipCluster) -> Result<Self> {
        Self::from_arc(Arc::new(cluster))
    }

    pub fn from_arc(cluster: Arc<HipCluster>) -> Result<Self> {
        let ar =
            BarP2pAllReduce::new(cluster.clone()).map_err(|e| anyhow!("BarP2pAllReduce: {e}"))?;
        Ok(Self { cluster, ar })
    }

    pub fn cluster(&self) -> &Arc<HipCluster> {
        &self.cluster
    }

    pub fn ar(&self) -> &BarP2pAllReduce {
        &self.ar
    }

    pub fn n_ranks(&self) -> usize {
        self.cluster.ranks()
    }
}

/// One pp+tp stage: a sub-cluster (tp_size ranks for this stage) + the
/// matching `BarP2pAllReduce`. Always constructed via
/// [`HybridCluster::new`] so the sub-cluster lives in the
/// before-global-cluster window.
pub struct HybridStageCluster {
    pub stage_idx: usize,
    pub sub_cluster: Arc<HipCluster>,
    pub ar: BarP2pAllReduce,
}

/// Hybrid (pp+tp) cluster: N sub-clusters (one per PP stage, each with
/// `tp_size` ranks for intra-stage TP) plus a single global cluster
/// spanning every rank for cross-stage PP handoff.
///
/// **Construction order matters:** every sub-cluster's
/// `BarP2pAllReduce` MUST be built before the global cluster's BAR1
/// peer-access matrix is probed. [`Self::new`] enforces this by taking
/// the per-stage sub-cluster Arcs first, building their ARs, and only
/// then accepting the global cluster.
pub struct HybridCluster {
    stages: Vec<HybridStageCluster>,
    global_cluster: Arc<HipCluster>,
    tp_size: usize,
}

impl HybridCluster {
    /// Build the hybrid cluster from its pieces. `sub_clusters` must
    /// already exist (caller built them via `HipCluster::new` per
    /// stage's device subset); this constructor wraps each in a
    /// `BarP2pAllReduce` before binding the `global_cluster`.
    pub fn new(
        sub_clusters: Vec<Arc<HipCluster>>,
        global_cluster: Arc<HipCluster>,
        tp_size: usize,
    ) -> Result<Self> {
        if sub_clusters.is_empty() {
            bail!("HybridCluster::new: 0 sub-clusters");
        }
        if tp_size == 0 {
            bail!("HybridCluster::new: tp_size=0");
        }
        let n_stages = sub_clusters.len();
        let expected_global = n_stages * tp_size;
        if global_cluster.ranks() != expected_global {
            bail!(
                "HybridCluster::new: global_cluster has {} ranks, expected n_stages*tp_size = {}",
                global_cluster.ranks(),
                expected_global,
            );
        }
        let mut stages = Vec::with_capacity(n_stages);
        for (stage_idx, sub) in sub_clusters.into_iter().enumerate() {
            if sub.ranks() != tp_size {
                bail!(
                    "HybridCluster::new: stage {stage_idx} sub-cluster has {} ranks, expected tp_size={tp_size}",
                    sub.ranks(),
                );
            }
            let ar = BarP2pAllReduce::new(sub.clone())
                .map_err(|e| anyhow!("BarP2pAllReduce stage {stage_idx}: {e}"))?;
            stages.push(HybridStageCluster {
                stage_idx,
                sub_cluster: sub,
                ar,
            });
        }
        Ok(Self {
            stages,
            global_cluster,
            tp_size,
        })
    }

    pub fn n_stages(&self) -> usize {
        self.stages.len()
    }

    pub fn tp_size(&self) -> usize {
        self.tp_size
    }

    pub fn global_cluster(&self) -> &Arc<HipCluster> {
        &self.global_cluster
    }

    pub fn stage(&self, stage_idx: usize) -> &HybridStageCluster {
        &self.stages[stage_idx]
    }

    /// Iterator-friendly access to every stage in stage_idx order.
    pub fn stages(&self) -> &[HybridStageCluster] {
        &self.stages
    }

    /// Mutable handle to a stage (for the rare case the arch driver
    /// needs `&mut` for stage-local mutable state — neither AR nor
    /// the cluster Arcs are normally re-borrowed mutably).
    pub fn stage_mut(&mut self, stage_idx: usize) -> &mut HybridStageCluster {
        &mut self.stages[stage_idx]
    }

    /// Global rank for `(stage, rank_in_stage)`. Matches the layout
    /// used by both `gemma4::hybrid` and `qwen3-moe::forward::hybrid`:
    /// `global_rank = stage_idx * tp_size + rank_in_stage`.
    pub fn global_rank_of(&self, stage_idx: usize, rank_in_stage: usize) -> usize {
        stage_idx * self.tp_size + rank_in_stage
    }
}
