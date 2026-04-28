//! AUTO-4b — hybrid PP-of-TP model loader.
//!
//! `Qwen3MoEHybridModel` composes `pp_size` `Qwen3MoETpModel` instances
//! (one per pipeline stage) on top of `pp_size` per-stage `HipCluster`s
//! of `tp_size` ranks each. Stages are uniform: each stage owns a
//! contiguous `num_layers / pp_size` slice of the layer list, plus
//! global tensors only where they're consumed:
//!
//!   - `token_embd` lives on **stage 0** only (the embed lookup runs
//!     before the layer loop on the first stage).
//!   - `output_norm` + `output` (LM head) live on the **last stage**
//!     only.
//!
//! This is the loader-only milestone for the hybrid path (task #65).
//! Forward composition is AUTO-4d (#67); session/scratch is AUTO-4c
//! (#66); server wiring is AUTO-4f (#69). The variant is selected
//! manually via `--mesh-mode pp+tp --pp-size N --tp-size M`; flambeau
//! does not autodetect topology — operators pick one with the AUTO-5
//! bracket-bench harness (#61).
//!
//! ### Device ordering
//!
//! `device_ids[0..tp_size]` form stage 0's TP subgroup,
//! `device_ids[tp_size..2*tp_size]` form stage 1's, and so on. The
//! caller is responsible for picking pairings that have a healthy
//! BAR1 peer-access matrix within each stage (the stage-internal
//! `BarP2pAllReduce` requires it). The cross-stage hand-off (one F16
//! hidden vector per token per hop) goes through `peer_copy_via_host`
//! on a **global** `HipCluster` constructed separately by the server.

use std::ops::Range;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipCluster;
use flambeau_core::{Device, Stream};
use flambeau_quant::GgufFile;

use crate::config::Qwen3MoEConfig;
use crate::forward::{ShardedForwardOneTokenScratchTp, ShardedForwardPrefillScratchTp};
use crate::session::{alloc_layer_cache_tp, dispose_layer_cache, LayerCache};
use crate::tp_layout::Qwen35DenseTpLayout;
use crate::tp_sharded::{Qwen3MoETpModel, TpLoadOpts};

/// Operator-supplied hybrid-mesh shape. Mirrors
/// [`flambeau_server::MeshMode::Hybrid`]; the loader keeps an owned
/// copy because the server may dispose its `MeshMode` before the
/// model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridMeshSpec {
    pub pp_size: u32,
    pub tp_size: u32,
}

impl HybridMeshSpec {
    /// Total ranks `= pp_size * tp_size`.
    pub fn total_ranks(self) -> usize {
        (self.pp_size as usize) * (self.tp_size as usize)
    }

    /// Validate against a model + device list. Both axes must be
    /// non-zero, total ranks must equal `device_ids.len()`, and the
    /// layer count must divide cleanly (uniform stages only at AUTO-4b).
    pub fn validate(self, num_layers: usize, num_devices: usize) -> Result<()> {
        if self.pp_size == 0 || self.tp_size == 0 {
            bail!(
                "HybridMeshSpec: pp_size={} tp_size={} (both must be > 0)",
                self.pp_size,
                self.tp_size
            );
        }
        if self.total_ranks() != num_devices {
            bail!(
                "HybridMeshSpec: pp_size={} * tp_size={} = {} but got {} devices",
                self.pp_size,
                self.tp_size,
                self.total_ranks(),
                num_devices
            );
        }
        if num_layers % (self.pp_size as usize) != 0 {
            bail!(
                "HybridMeshSpec: pp_size={} must divide num_layers={} (uniform stages only)",
                self.pp_size,
                num_layers
            );
        }
        Ok(())
    }

    /// Layer range owned by `stage` (0-indexed). Caller must have
    /// passed `validate` first.
    pub fn stage_layer_range(self, stage: u32, num_layers: usize) -> Range<usize> {
        let per_stage = num_layers / (self.pp_size as usize);
        let start = (stage as usize) * per_stage;
        let end = if stage + 1 == self.pp_size {
            num_layers
        } else {
            start + per_stage
        };
        start..end
    }

    /// Slice of `device_ids` belonging to `stage`'s TP subgroup.
    /// Caller must have passed `validate` first.
    pub fn stage_device_ids<'a>(self, stage: u32, all: &'a [i32]) -> &'a [i32] {
        let tp = self.tp_size as usize;
        let start = (stage as usize) * tp;
        &all[start..start + tp]
    }
}

/// One PP stage's worth of TP shards. Owns its sub-cluster (only the
/// `tp_size` devices that participate in this stage's TP group), the
/// per-stage TP layout, and a `Qwen3MoETpModel` populated with just
/// this stage's layer range plus whichever globals belong here.
pub struct Qwen3MoEHybridStage {
    pub stage_idx: u32,
    pub layer_range: Range<usize>,
    pub sub_cluster: Arc<HipCluster>,
    pub tp_model: Qwen3MoETpModel,
}

impl std::fmt::Debug for Qwen3MoEHybridStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3MoEHybridStage")
            .field("stage_idx", &self.stage_idx)
            .field("layer_range", &self.layer_range)
            .field("ranks", &self.sub_cluster.ranks())
            .field("has_token_embd", &self.tp_model.has_token_embd)
            .field("has_output_head", &self.tp_model.has_output_head)
            .finish()
    }
}

/// Hybrid PP-of-TP model — `Vec<Qwen3MoEHybridStage>`, one per PP
/// stage. Constructed once at `flambeau serve` startup and consumed by
/// the AUTO-4d hybrid forward path.
pub struct Qwen3MoEHybridModel {
    pub config: Qwen3MoEConfig,
    pub spec: HybridMeshSpec,
    pub stages: Vec<Qwen3MoEHybridStage>,
}

impl std::fmt::Debug for Qwen3MoEHybridModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3MoEHybridModel")
            .field("spec", &self.spec)
            .field("num_layers", &self.config.num_layers)
            .field("stages", &self.stages)
            .finish()
    }
}

impl Qwen3MoEHybridModel {
    /// Build the per-stage sub-clusters, slice the layer table per
    /// stage, and load every stage's weights.
    ///
    /// `device_ids` is in stage-major order: `device_ids[s*tp_size ..
    /// (s+1)*tp_size]` form stage `s`'s TP subgroup.
    pub fn load(
        file: &GgufFile,
        device_ids: &[i32],
        spec: HybridMeshSpec,
    ) -> Result<Self> {
        let config = Qwen3MoEConfig::from_gguf(file)
            .context("Qwen3MoEConfig::from_gguf for hybrid load")?;
        spec.validate(config.num_layers, device_ids.len())
            .context("HybridMeshSpec::validate")?;

        let mut stages: Vec<Qwen3MoEHybridStage> = Vec::with_capacity(spec.pp_size as usize);
        for stage_idx in 0..spec.pp_size {
            let stage_devices = spec.stage_device_ids(stage_idx, device_ids);
            let sub_cluster: Arc<HipCluster> = Arc::new(
                HipCluster::new(stage_devices)
                    .with_context(|| format!("HipCluster::new for stage {stage_idx}"))?,
            );
            // The intra-stage AR path requires a fully-connected peer
            // matrix among `stage_devices`. We don't probe it here —
            // `BarP2pAllReduce::new` (constructed in AUTO-4f) is the
            // right place; failure there surfaces a clear "stage N has
            // a broken peer pair" message to the operator.

            let tp_layout = Qwen35DenseTpLayout::new(&config, spec.tp_size)
                .with_context(|| format!("Qwen35DenseTpLayout::new for stage {stage_idx}"))?;

            let layer_range = spec.stage_layer_range(stage_idx, config.num_layers);
            let opts = TpLoadOpts {
                layer_range: Some(layer_range.clone()),
                load_token_embd: stage_idx == 0,
                load_output_head: stage_idx + 1 == spec.pp_size,
            };

            tracing::info!(
                stage = stage_idx,
                ?layer_range,
                tp_size = spec.tp_size,
                load_token_embd = opts.load_token_embd,
                load_output_head = opts.load_output_head,
                stage_devices = ?stage_devices,
                "loading hybrid stage"
            );

            let tp_model = Qwen3MoETpModel::load_with_opts(file, &sub_cluster, tp_layout, &opts)
                .with_context(|| format!("Qwen3MoETpModel::load_with_opts stage {stage_idx}"))?;

            stages.push(Qwen3MoEHybridStage {
                stage_idx,
                layer_range,
                sub_cluster,
                tp_model,
            });
        }

        Ok(Self {
            config,
            spec,
            stages,
        })
    }

    /// Total bytes uploaded across all stages and all per-stage ranks.
    pub fn total_bytes(&self) -> usize {
        self.stages.iter().map(|s| s.tp_model.total_bytes()).sum()
    }

    /// Bytes uploaded to a specific stage. Diagnostic.
    pub fn stage_bytes(&self, stage_idx: u32) -> Option<usize> {
        self.stages
            .iter()
            .find(|s| s.stage_idx == stage_idx)
            .map(|s| s.tp_model.total_bytes())
    }

    /// Free every stage's per-rank shards. Each stage disposes against
    /// its own sub-cluster (the global cluster is the server's
    /// concern).
    pub fn dispose(mut self) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for stage in self.stages.drain(..) {
            let Qwen3MoEHybridStage {
                sub_cluster,
                tp_model,
                ..
            } = stage;
            if let Err(e) = tp_model.dispose(&sub_cluster) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

// ────────────────────────────────────────────────────────────────────
// AUTO-4c — per-stage session + decode/prefill scratch
// ────────────────────────────────────────────────────────────────────

/// One PP stage's per-rank KV / GDN caches. `caches[rank][i]` holds the
/// cache for **layer `layer_range.start + i`**. Layers outside the
/// stage's range live on a different stage and are not allocated here.
///
/// Sizing inherits the same per-rank-TP shrinkage as
/// [`crate::Qwen3MoETpSession`]: `local_num_v_heads = num_v_heads /
/// tp_size` (GDN), `local_num_kv_heads = num_kv_heads / tp_size` for
/// full-attn (with the V2.4d-i2 KV-replication fallback when
/// `num_kv_heads % tp_size != 0`).
pub struct Qwen3MoEHybridStageSession {
    pub stage_idx: u32,
    pub layer_range: Range<usize>,
    /// `caches[rank][layer_offset]` where `layer_offset = il -
    /// layer_range.start`. Matches [`Qwen3MoEHybridStage::sub_cluster`]'s
    /// rank order.
    pub caches: Vec<Vec<LayerCache>>,
    disposed: bool,
}

impl Qwen3MoEHybridStageSession {
    /// Bytes of KV / GDN state held by this stage across all its TP
    /// ranks. Used by the AUTO-4c smoke cert.
    pub fn total_bytes(&self) -> usize {
        let mut total = 0usize;
        for rank_caches in &self.caches {
            for c in rank_caches {
                match c {
                    LayerCache::FullAttn(kv) => total += kv.bytes_per_tensor() * 2,
                    LayerCache::FullAttnQ8(kv) => total += kv.bytes_per_tensor() * 2,
                    LayerCache::Gdn(g) => total += g.state_bytes + g.conv_history_bytes,
                }
            }
        }
        total
    }

    /// Free this stage's per-rank caches against its sub-cluster.
    /// Mirrors [`Qwen3MoETpSession::dispose`]; the hybrid model owns the
    /// sub-cluster, so callers pass it through from the parent stage.
    pub fn dispose(mut self, sub_cluster: &HipCluster) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        let mut first_err: Option<anyhow::Error> = None;
        for (rank, layer_caches) in self.caches.drain(..).enumerate() {
            let device = sub_cluster.device(rank);
            device.bind()?;
            for cache in layer_caches {
                if let Err(e) = dispose_layer_cache(cache, device) {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

impl Drop for Qwen3MoEHybridStageSession {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::hybrid",
                stage = self.stage_idx,
                ?self.layer_range,
                "Qwen3MoEHybridStageSession dropped without dispose(sub_cluster); device buffers leaked"
            );
        }
    }
}

/// Per-request session for a hybrid PP+TP model. Holds one
/// [`Qwen3MoEHybridStageSession`] per stage. Constructed once per
/// inflight request and disposed when the request finishes.
pub struct Qwen3MoEHybridSession {
    pub stages: Vec<Qwen3MoEHybridStageSession>,
    disposed: bool,
}

impl Qwen3MoEHybridSession {
    /// Allocate per-stage / per-rank caches for `model`. Each stage
    /// allocates only its `layer_range`; layers belonging to other
    /// stages are not present.
    pub fn new(model: &Qwen3MoEHybridModel) -> Result<Self> {
        let cfg = &model.config;
        let tp_world = model.spec.tp_size;
        let mut stages: Vec<Qwen3MoEHybridStageSession> = Vec::with_capacity(model.stages.len());
        for stage in &model.stages {
            let mut per_rank: Vec<Vec<LayerCache>> =
                Vec::with_capacity(stage.sub_cluster.ranks());
            for rank_idx in 0..stage.sub_cluster.ranks() {
                let device = stage.sub_cluster.device(rank_idx);
                device.bind()?;
                let mut layer_caches: Vec<LayerCache> =
                    Vec::with_capacity(stage.layer_range.len());
                for il in stage.layer_range.clone() {
                    layer_caches
                        .push(alloc_layer_cache_tp(cfg, device, il, tp_world).with_context(
                            || {
                                format!(
                                    "alloc_layer_cache_tp stage={} rank={rank_idx} layer={il}",
                                    stage.stage_idx
                                )
                            },
                        )?);
                }
                device.default_stream().synchronize()?;
                per_rank.push(layer_caches);
            }
            stages.push(Qwen3MoEHybridStageSession {
                stage_idx: stage.stage_idx,
                layer_range: stage.layer_range.clone(),
                caches: per_rank,
                disposed: false,
            });
        }
        Ok(Self {
            stages,
            disposed: false,
        })
    }

    /// Total cache bytes across all stages.
    pub fn total_bytes(&self) -> usize {
        self.stages.iter().map(|s| s.total_bytes()).sum()
    }

    /// Dispose every stage's caches. Each stage disposes against its
    /// owning [`Qwen3MoEHybridStage::sub_cluster`]; the model is passed
    /// in so the session can find them.
    pub fn dispose(mut self, model: &Qwen3MoEHybridModel) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        if self.stages.len() != model.stages.len() {
            bail!(
                "Qwen3MoEHybridSession::dispose: session has {} stages but model has {}",
                self.stages.len(),
                model.stages.len()
            );
        }
        let mut first_err: Option<anyhow::Error> = None;
        for (stage_session, stage) in self.stages.drain(..).zip(model.stages.iter()) {
            if let Err(e) = stage_session.dispose(&stage.sub_cluster) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

impl Drop for Qwen3MoEHybridSession {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::hybrid",
                stages = self.stages.len(),
                "Qwen3MoEHybridSession dropped without dispose(model); device buffers leaked"
            );
        }
    }
}

/// Per-stage decode/prefill scratch for the hybrid forward path. Reuses
/// [`ShardedForwardOneTokenScratchTp`] verbatim because that scratch is
/// hidden-size-driven, not layer-count-driven — the same shape covers
/// any stage's layer range. The `head_stage` field marks the stage that
/// runs the LM head + argmax (always the last stage in AUTO-4a, since
/// that's where `output_norm` / `output` were uploaded by the loader).
///
/// Inter-stage hand-off is one F16 hidden vector per token per hop,
/// sent via [`HipCluster::peer_copy_via_host`] on the **global**
/// cluster (constructed by the server, not by this scratch). The
/// global cluster is the only place that knows about all
/// `pp_size * tp_size` ranks at once.
pub struct ShardedForwardOneTokenScratchHybrid {
    pub per_stage: Vec<ShardedForwardOneTokenScratchTp>,
    /// Stage that runs the LM head. V1: `pp_size - 1`.
    pub head_stage: u32,
    /// CN-80B-20 — per-stage-per-rank captured decode graph. Outer indexes
    /// stage, inner indexes rank within that stage's sub_cluster. Populated
    /// lazily on the first decode call when `FLAMBEAU_DECODE_GRAPH=1`.
    /// Iter 1 stores HipGraphExec only (no slot binding) — captured K/V
    /// append destinations and `n_tokens_kv` are frozen at capture time, so
    /// replays produce TIMING-MEANINGFUL but OUTPUT-WRONG results. Iter 2
    /// adds slot binding to make replay correctness-preserving.
    pub decode_graphs: Vec<Vec<Option<flambeau_backend_hip::HipGraphExec>>>,
    disposed: bool,
}

impl ShardedForwardOneTokenScratchHybrid {
    /// Allocate per-stage TP scratch for `model`. The LM head runs on
    /// the last stage (where the loader put `output_norm` + `output`).
    pub fn new(model: &Qwen3MoEHybridModel) -> Result<Self> {
        let cfg = &model.config;
        let head_stage = model.spec.pp_size - 1;
        let mut per_stage: Vec<ShardedForwardOneTokenScratchTp> =
            Vec::with_capacity(model.stages.len());
        for stage in &model.stages {
            // Only the head stage needs `OutputHeadScratch`. The TP
            // scratch's `new_with_head_rank` puts the LM-head buffers on
            // a single rank within the cluster; for non-head stages we
            // still pass rank 0 — the scratch is allocated but unused
            // and gets disposed cleanly. (Skipping it would require a
            // bool toggle in TP scratch construction; deferring that
            // micro-optimisation to a follow-up.)
            let scratch =
                ShardedForwardOneTokenScratchTp::new_with_head_rank(
                    cfg,
                    &stage.sub_cluster,
                    flambeau_runtime::RankId(0),
                )
                .with_context(|| {
                    format!(
                        "ShardedForwardOneTokenScratchTp::new stage={} (tp ranks={})",
                        stage.stage_idx,
                        stage.sub_cluster.ranks()
                    )
                })?;
            per_stage.push(scratch);
        }
        // CN-80B-20 — graph cache: outer per stage, inner per rank within
        // that stage's sub_cluster. Lazy population on first decode call.
        let decode_graphs: Vec<Vec<Option<flambeau_backend_hip::HipGraphExec>>> = model
            .stages
            .iter()
            .map(|stage| (0..stage.sub_cluster.ranks()).map(|_| None).collect())
            .collect();
        Ok(Self {
            per_stage,
            head_stage,
            decode_graphs,
            disposed: false,
        })
    }

    /// Per-stage scratch byte total (same convention as
    /// [`Qwen3MoEHybridSession::total_bytes`]).
    pub fn total_bytes(&self) -> usize {
        self.per_stage.iter().map(|s| s.rank_level_bytes()).sum()
    }

    /// Free every stage's scratch against its sub-cluster.
    pub fn dispose(mut self, model: &Qwen3MoEHybridModel) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        if self.per_stage.len() != model.stages.len() {
            bail!(
                "ShardedForwardOneTokenScratchHybrid::dispose: scratch has {} stages \
                 but model has {}",
                self.per_stage.len(),
                model.stages.len()
            );
        }
        let mut first_err: Option<anyhow::Error> = None;
        for (scratch, stage) in self.per_stage.drain(..).zip(model.stages.iter()) {
            if let Err(e) = scratch.dispose(&stage.sub_cluster) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

impl Drop for ShardedForwardOneTokenScratchHybrid {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::hybrid",
                stages = self.per_stage.len(),
                "ShardedForwardOneTokenScratchHybrid dropped without dispose(model); device buffers leaked"
            );
        }
    }
}

/// **AUTO-6e2** — per-stage prefill scratch for the hybrid (PP-of-TP)
/// L-batched driver. Mirrors [`ShardedForwardOneTokenScratchHybrid`]
/// but each stage's scratch is a [`ShardedForwardPrefillScratchTp`]
/// sized for `max_tokens`.
///
/// LM head buffers live only on `head_stage` (last stage) — non-head
/// stages still allocate `OutputHeadScratch` against rank 0 (cheap)
/// and dispose cleanly. Mirrors the per-token sibling's choice; could
/// be tightened with a bool toggle if it shows up on profiles.
pub struct ShardedForwardPrefillScratchHybrid {
    pub per_stage: Vec<ShardedForwardPrefillScratchTp>,
    pub head_stage: u32,
    pub max_tokens: usize,
    disposed: bool,
}

impl ShardedForwardPrefillScratchHybrid {
    pub fn new(model: &Qwen3MoEHybridModel, max_tokens: usize) -> Result<Self> {
        if max_tokens == 0 {
            bail!("ShardedForwardPrefillScratchHybrid::new max_tokens must be >= 1");
        }
        let cfg = &model.config;
        let head_stage = model.spec.pp_size - 1;
        let mut per_stage: Vec<ShardedForwardPrefillScratchTp> =
            Vec::with_capacity(model.stages.len());
        for stage in &model.stages {
            let scratch = ShardedForwardPrefillScratchTp::new_with_head_rank(
                cfg,
                &stage.sub_cluster,
                max_tokens,
                flambeau_runtime::RankId(0),
            )
            .with_context(|| {
                format!(
                    "ShardedForwardPrefillScratchTp::new stage={} (tp ranks={}, max_tokens={max_tokens})",
                    stage.stage_idx,
                    stage.sub_cluster.ranks()
                )
            })?;
            per_stage.push(scratch);
        }
        Ok(Self {
            per_stage,
            head_stage,
            max_tokens,
            disposed: false,
        })
    }

    pub fn total_bytes(&self) -> usize {
        self.per_stage.iter().map(|s| s.rank_level_bytes()).sum()
    }

    pub fn dispose(mut self, model: &Qwen3MoEHybridModel) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        if self.per_stage.len() != model.stages.len() {
            bail!(
                "ShardedForwardPrefillScratchHybrid::dispose: scratch has {} stages \
                 but model has {}",
                self.per_stage.len(),
                model.stages.len()
            );
        }
        let mut first_err: Option<anyhow::Error> = None;
        for (scratch, stage) in self.per_stage.drain(..).zip(model.stages.iter()) {
            if let Err(e) = scratch.dispose(&stage.sub_cluster) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

impl Drop for ShardedForwardPrefillScratchHybrid {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::hybrid",
                stages = self.per_stage.len(),
                max_tokens = self.max_tokens,
                "ShardedForwardPrefillScratchHybrid dropped without dispose(model); device buffers leaked"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_pp_size_must_divide_num_layers() {
        let spec = HybridMeshSpec {
            pp_size: 3,
            tp_size: 2,
        };
        // 6 ranks but 40 layers / 3 stages does not divide.
        let err = spec.validate(40, 6).unwrap_err();
        assert!(
            err.to_string().contains("must divide num_layers"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_total_ranks_must_match_devices() {
        let spec = HybridMeshSpec {
            pp_size: 2,
            tp_size: 2,
        };
        let err = spec.validate(40, 5).unwrap_err();
        assert!(err.to_string().contains("got 5 devices"), "got: {err}");
    }

    #[test]
    fn stage_layer_range_uniform() {
        let spec = HybridMeshSpec {
            pp_size: 4,
            tp_size: 2,
        };
        spec.validate(40, 8).unwrap();
        assert_eq!(spec.stage_layer_range(0, 40), 0..10);
        assert_eq!(spec.stage_layer_range(1, 40), 10..20);
        assert_eq!(spec.stage_layer_range(2, 40), 20..30);
        assert_eq!(spec.stage_layer_range(3, 40), 30..40);
    }

    #[test]
    fn stage_device_ids_stage_major() {
        let spec = HybridMeshSpec {
            pp_size: 2,
            tp_size: 2,
        };
        let devs = [0i32, 2, 1, 3];
        assert_eq!(spec.stage_device_ids(0, &devs), &[0, 2]);
        assert_eq!(spec.stage_device_ids(1, &devs), &[1, 3]);
    }
}
