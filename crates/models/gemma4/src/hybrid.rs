//! Hybrid PP-of-TP decode driver for Gemma 4 (pp2tp2 — 4-GPU target).
//!
//! Mesh shape: `n_stages * ranks_per_stage` devices. Stage `s`'s
//! sub-cluster holds `ranks_per_stage[s]` devices; the global cluster
//! covers all of them in the order `s * tp_size + r`. The driver runs
//! TP within a stage (Megatron-style sharding + `sum_tp2` AR) and PP
//! between stages (peer-copy-via-host of the F16 hidden residual).
//!
//! Cluster construction order matters (MEMORY.md `hybrid_cluster_order`):
//! per-stage sub-clusters MUST be created BEFORE the global cluster.
//! The driver assumes the caller already obeys this ordering when
//! handing in the cluster handles.
//!
//! Device mesh for the 4× MI50 rig is `hip:0,2,1,3` with `pp_size=2,
//! tp_size=2` (MEMORY.md `never_tp4_use_pp2tp2`) — the {2,3} link-
//! faulted pair is avoided by interleaving devices across stages.
//!
//! S10-A scope:
//! - Decode only (hybrid prefill = S10-B follow-up).
//! - pp2tp2 only (4 stages × N ranks deferred to S10-B; TP4 stages
//!   wait on #20).
//! - Dense FFN only; MoE deferred (#18 + S10-B integration).
//! - No shared-KV-on-hybrid (deferred), no per-layer-embed (#17), no
//!   layer_output_scale.

#![cfg(feature = "hip")]

use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use flambeau_backend_hip::{HipCluster, HipDevice};
use flambeau_blocks::{
    apply_layer_output_scale_f16, embed_token_host, forward_one_token_hybrid, tp_allreduce_sum,
    tp_allreduce_sum_f32_synced, tp_allreduce_sum_synced, upload_f16_ones, Activation, Buffer,
    DenseMlpDecodeScratch, DenseMlpTp, HybridCluster, HybridDecodeDriver, LmHead, OutputNorm,
    RawAllocTracker, RowParallel, StageCommon, StandardAttention, StandardAttentionDecodeScratch,
    TokenEmbd, TpRankCore, WeightHandle, WeightUploader, F16,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{HipOps, OpsRegistry};
use flambeau_ops::Ops;
use flambeau_quant::GgmlDType;
use flambeau_runtime::{F16Contig, KvCache};

use crate::config::Gemma4Config;
use crate::layer::Gemma4LayerWeights;
use crate::layout::{FfnKind, ModelLayout};
use crate::output_head::{forward_output_head, OutputHeadScratch};
use crate::weights_hip::DeviceTensor;

/// Per-rank state within a stage. Mirrors `Gemma4TpStage` but
/// parameterised by stage layers + their sharded weights.
/// Per-rank model state for a hybrid PP-of-TP stage (weights only —
/// Arc-shareable across multiple concurrent `Gemma4HybridSession`s).
pub struct HybridRankModel {
    /// Shared bookkeeping for **weight** allocations on this rank.
    pub common: StageCommon,
    pub layer_weights: Vec<Gemma4LayerWeights>,
    /// Stage 0 only.
    pub token_embd: Option<DeviceTensor>,
    pub token_embd_dims: Option<[usize; 2]>,
    /// Head stage only (replicated across that stage's ranks).
    pub output_norm: Option<DeviceTensor>,
    pub lm_head: Option<DeviceTensor>,
    pub lm_head_dims: Option<[usize; 2]>,
}

impl HybridRankModel {
    pub fn rank_in_stage(&self) -> usize {
        self.common.rank as usize
    }

    fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if self.common.is_disposed() {
            return Ok(());
        }
        self.common
            .dispose(device)
            .map_err(|e| anyhow!("hybrid model raw_alloc dispose: {e}"))?;
        for t in [
            std::mem::replace(&mut self.token_embd, None),
            std::mem::replace(&mut self.output_norm, None),
            std::mem::replace(&mut self.lm_head, None),
        ]
        .into_iter()
        .flatten()
        {
            if !t.ptr.is_null() && t.bytes > 0 {
                unsafe {
                    let _ = device.dealloc(t.ptr, t.bytes);
                }
            }
        }
        let _ = dummy_dt();
        Ok(())
    }
}

impl Drop for HybridRankModel {
    fn drop(&mut self) {
        self.common
            .warn_on_leak("flambeau_gemma4::hybrid::HybridRankModel");
    }
}

/// Per-rank session state for a hybrid PP-of-TP stage (KV + scratch +
/// per-request sync events).
pub struct HybridRankSession {
    /// Shared bookkeeping for **scratch + per-request** allocations.
    pub common: StageCommon,
    pub kv_caches: Vec<Option<KvCache<F16Contig, HipDevice>>>,
    pub hidden: DevicePtr,
    pub partial_attn: DevicePtr,
    pub partial_ffn: DevicePtr,
    /// F32 attention path scratch (allocated only on MoE models with
    /// full-attention layers).
    pub partial_attn_f32: DevicePtr,
    pub attn_normed_f32_tmp: DevicePtr,
    /// MoE FFN cascade scratch.
    pub tp_moe_scratch: Option<crate::tp_moe_upload::Gemma4TpMoeScratch>,
    /// Per-request TP sync (producer_done event).
    pub core: TpRankCore,
    scratch: HybridScratchPtrs,
    /// Head rank only.
    pub output_head_scratch: Option<OutputHeadScratch>,
    positions_host: Vec<i32>,
}

impl HybridRankSession {
    pub fn rank_in_stage(&self) -> usize {
        self.common.rank as usize
    }

    fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if self.common.is_disposed() {
            return Ok(());
        }
        let kvs = std::mem::take(&mut self.kv_caches);
        for kv in kvs.into_iter().flatten() {
            kv.dispose(device).map_err(|e| anyhow!("kv dispose: {e}"))?;
        }
        self.common
            .dispose(device)
            .map_err(|e| anyhow!("hybrid session raw_alloc dispose: {e}"))?;
        Ok(())
    }
}

impl Drop for HybridRankSession {
    fn drop(&mut self) {
        self.common
            .warn_on_leak("flambeau_gemma4::hybrid::HybridRankSession");
    }
}

/// Bundled per-rank state. Test fixtures + back-compat wrapper.
pub struct HybridRankState {
    pub model: HybridRankModel,
    pub session: HybridRankSession,
}

impl HybridRankState {
    pub fn rank_in_stage(&self) -> usize {
        self.model.rank_in_stage()
    }

    pub fn into_halves(self) -> (HybridRankModel, HybridRankSession) {
        (self.model, self.session)
    }
}

struct HybridScratchPtrs {
    x_q8_1: (DevicePtr, usize),
    mmvq_f32: (DevicePtr, usize),
    q_f16: (DevicePtr, usize),
    k_f16: (DevicePtr, usize),
    v_f16: (DevicePtr, usize),
    attn_out_local: (DevicePtr, usize),
    attn_residual_f16: (DevicePtr, usize),
    gate_f32: (DevicePtr, usize),
    up_f32: (DevicePtr, usize),
    activated_f16: (DevicePtr, usize),
    activated_q8_1: (DevicePtr, usize),
    positions: (DevicePtr, usize),
    v_ones_f16: (DevicePtr, usize),
    /// Splitk partials (engaged by `StandardAttention::forward_decode`
    /// when `n_tokens_kv > 256`). Sized for n_heads_local_max heads.
    splitk_partials_m: (DevicePtr, usize),
    splitk_partials_s: (DevicePtr, usize),
    splitk_partials_o: (DevicePtr, usize),
}

/// One PP stage — model half (weights + per-rank `OpsRegistry`).
pub struct Gemma4HybridModelStage {
    pub stage_idx: usize,
    /// Layers in this stage (global indices, ascending).
    pub layers_global: Vec<usize>,
    pub rank_state: Vec<HybridRankModel>,
    /// Per-rank `OpsRegistry` over the sub-cluster.
    pub regs: Vec<OpsRegistry>,
}

/// One PP stage — session half (per-rank KV + scratch + per-request
/// sync events).
pub struct Gemma4HybridSessionStage {
    pub stage_idx: usize,
    pub rank_state: Vec<HybridRankSession>,
}

/// Bundled per-stage state. Test fixtures + back-compat wrapper.
pub struct Gemma4HybridStage {
    pub model: Gemma4HybridModelStage,
    pub session: Gemma4HybridSessionStage,
}

impl Gemma4HybridStage {
    pub fn into_halves(self) -> (Gemma4HybridModelStage, Gemma4HybridSessionStage) {
        (self.model, self.session)
    }
}

/// Hybrid model — Arc-shareable. Owns the [`HybridCluster`] +
/// per-stage layer/weights/registries.
pub struct Gemma4HybridModel {
    pub hc: HybridCluster,
    pub cfg: Gemma4Config,
    pub layout: ModelLayout,
    pub stages: Vec<Gemma4HybridModelStage>,
    pub layer_to_stage: Vec<usize>,
    pub head_stage_idx: usize,
    pub head_rank_in_head_stage_idx: usize,
}

impl Gemma4HybridModel {
    pub fn dispose(&mut self) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for stage in self.stages.iter_mut() {
            let sub = &self.hc.stage(stage.stage_idx).sub_cluster;
            for (r, rank_state) in stage.rank_state.iter_mut().enumerate() {
                let dev = sub.device(r);
                if let Err(e) = rank_state.dispose(dev) {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

impl Drop for Gemma4HybridModel {
    fn drop(&mut self) {
        let leaked = self.stages.iter().any(|s| {
            s.rank_state
                .iter()
                .any(|r| !r.common.is_disposed())
        });
        if leaked {
            tracing::warn!("Gemma4HybridModel dropped without dispose()");
        }
    }
}

/// Per-request hybrid session. Owns per-stage per-rank KV + scratch.
pub struct Gemma4HybridSession {
    pub stages: Vec<Gemma4HybridSessionStage>,
    pub logits_host: Vec<f32>,
}

impl Gemma4HybridSession {
    pub fn dispose(&mut self, model: &Gemma4HybridModel) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for stage in self.stages.iter_mut() {
            let sub = &model.hc.stage(stage.stage_idx).sub_cluster;
            for (r, rank_state) in stage.rank_state.iter_mut().enumerate() {
                let dev = sub.device(r);
                if let Err(e) = rank_state.dispose(dev) {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

/// Hybrid driver. Bundles `Arc<Gemma4HybridModel>` (shared weights)
/// and `Gemma4HybridSession` (per-request KV+scratch).
pub struct Gemma4HybridDriver {
    pub model: std::sync::Arc<Gemma4HybridModel>,
    pub session: Gemma4HybridSession,
}

fn dummy_dt() -> DeviceTensor {
    DeviceTensor {
        ptr: DevicePtr::NULL,
        dtype: GgmlDType::F32,
        bytes: 0,
    }
}

/// Even-split layer-to-stage assignment with shared-KV stage-boundary
/// validation (mirrors `partition_layers` in `pp.rs`).
pub fn partition_layers_pp(n_stages: usize, layout: &ModelLayout) -> Result<Vec<usize>> {
    if n_stages == 0 {
        bail!("partition_layers_pp: n_stages=0");
    }
    let n = layout.layers.len();
    if n == 0 {
        bail!("partition_layers_pp: 0 layers");
    }
    let per_stage = n.div_ceil(n_stages);
    let mut out = vec![0usize; n];
    for (i, _) in layout.layers.iter().enumerate() {
        out[i] = (i / per_stage).min(n_stages - 1);
    }
    for spec in &layout.layers {
        if !spec.has_kv {
            let src = spec.kv_share_src.ok_or_else(|| {
                anyhow!(
                    "partition_layers_pp: tail layer {} has no kv_share_src",
                    spec.index
                )
            })?;
            if out[spec.index] != out[src] {
                bail!(
                    "partition_layers_pp: shared-KV tail layer {} on stage {} but \
                     kv_share_src layer {} on stage {}; tail must be in same stage",
                    spec.index,
                    out[spec.index],
                    src,
                    out[src]
                );
            }
        }
    }
    Ok(out)
}

impl HybridRankState {
    #[allow(clippy::too_many_arguments)]
    pub fn from_pieces(
        device: &HipDevice,
        rank_in_stage: usize,
        cfg: &Gemma4Config,
        layout: &ModelLayout,
        layers_global: &[usize],
        tp_size: usize,
        layer_weights: Vec<Gemma4LayerWeights>,
        token_embd: Option<DeviceTensor>,
        token_embd_dims: Option<[usize; 2]>,
        output_norm: Option<DeviceTensor>,
        lm_head: Option<DeviceTensor>,
        lm_head_dims: Option<[usize; 2]>,
        is_head_rank: bool,
        max_tokens: usize,
        weight_alloc: RawAllocTracker,
    ) -> Result<Self> {
        device.bind()?;
        if layer_weights.len() != layers_global.len() {
            bail!(
                "HybridRankState: expected {} layer weights, got {}",
                layers_global.len(),
                layer_weights.len()
            );
        }
        let model = HybridRankModel::from_pieces(
            device,
            rank_in_stage,
            layer_weights,
            token_embd,
            token_embd_dims,
            output_norm,
            lm_head,
            lm_head_dims,
            weight_alloc,
        )?;
        let session = HybridRankSession::from_pieces(
            device,
            rank_in_stage,
            cfg,
            layout,
            layers_global,
            tp_size,
            is_head_rank,
            max_tokens,
        )?;
        Ok(Self { model, session })
    }

    fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        self.session.dispose(device)?;
        self.model.dispose(device)?;
        Ok(())
    }
}

impl HybridRankModel {
    #[allow(clippy::too_many_arguments)]
    fn from_pieces(
        device: &HipDevice,
        rank_in_stage: usize,
        layer_weights: Vec<Gemma4LayerWeights>,
        token_embd: Option<DeviceTensor>,
        token_embd_dims: Option<[usize; 2]>,
        output_norm: Option<DeviceTensor>,
        lm_head: Option<DeviceTensor>,
        lm_head_dims: Option<[usize; 2]>,
        weight_alloc: RawAllocTracker,
    ) -> Result<Self> {
        let mut common = StageCommon::new(rank_in_stage as u32, device.id());
        common.raw_alloc = weight_alloc;
        Ok(Self {
            common,
            layer_weights,
            token_embd,
            token_embd_dims,
            output_norm,
            lm_head,
            lm_head_dims,
        })
    }
}

impl HybridRankSession {
    #[allow(clippy::too_many_arguments)]
    pub fn from_pieces(
        device: &HipDevice,
        rank_in_stage: usize,
        cfg: &Gemma4Config,
        layout: &ModelLayout,
        layers_global: &[usize],
        tp_size: usize,
        is_head_rank: bool,
        max_tokens: usize,
    ) -> Result<Self> {
        device.bind()?;
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim.max(cfg.swa.head_dim_swa);
        let n_heads_local_max = cfg.num_heads / tp_size;
        let n_kv_local_max = layers_global
            .iter()
            .map(|&gi| cfg.num_kv_heads[gi] / tp_size)
            .max()
            .unwrap_or(0);
        let q_width_local_max = n_heads_local_max * head_dim;
        let kv_width_local_max = n_kv_local_max * head_dim;
        let ff_len = cfg.feed_forward_length;
        let ff_len_local = ff_len / tp_size;

        let mut kv_caches = Vec::with_capacity(layers_global.len());
        for &gi in layers_global {
            let spec = &layout.layers[gi];
            if !spec.has_kv {
                bail!("HybridRankSession: shared-KV tail layer {gi} not supported in S10-H");
            }
            let n_kv_local = spec.n_kv_heads / tp_size;
            let kv = KvCache::<F16Contig, HipDevice>::new(
                device,
                n_kv_local,
                spec.head_dim,
                max_tokens,
            )
            .map_err(|e| anyhow!("kv alloc layer {gi}: {e}"))?;
            kv_caches.push(Some(kv));
        }

        let mut common = StageCommon::new(rank_in_stage as u32, device.id());
        let raw_alloc = &mut common.raw_alloc;

        let mmvq_max = q_width_local_max
            .max(kv_width_local_max)
            .max(hidden)
            .max(ff_len_local);
        let x_q8_1_n = hidden.max(ff_len_local).div_ceil(32) * 32;
        let activated_q8_1_n = ff_len_local.div_ceil(32) * 32;

        let v_ones_ptr = upload_f16_ones(device, head_dim)?;
        raw_alloc.track(v_ones_ptr, head_dim * 2);

        let splitk_ms_n = n_heads_local_max * flambeau_blocks::MAX_SPLITK_CHUNKS;
        let splitk_o_n = n_heads_local_max * flambeau_blocks::MAX_SPLITK_CHUNKS * head_dim;

        let scratch = HybridScratchPtrs {
            x_q8_1: raw_alloc.alloc_q8_1(device, x_q8_1_n)?,
            mmvq_f32: raw_alloc.alloc_f32(device, mmvq_max)?,
            q_f16: raw_alloc.alloc_f16(device, q_width_local_max)?,
            k_f16: raw_alloc.alloc_f16(device, kv_width_local_max)?,
            v_f16: raw_alloc.alloc_f16(device, kv_width_local_max)?,
            attn_out_local: raw_alloc.alloc_f16(device, q_width_local_max.max(hidden))?,
            attn_residual_f16: raw_alloc.alloc_f16(device, hidden)?,
            gate_f32: raw_alloc.alloc_f32(device, ff_len_local)?,
            up_f32: raw_alloc.alloc_f32(device, ff_len_local)?,
            activated_f16: raw_alloc.alloc_f16(device, ff_len_local)?,
            activated_q8_1: raw_alloc.alloc_q8_1(device, activated_q8_1_n)?,
            positions: raw_alloc.alloc_i32(device, 1)?,
            v_ones_f16: (v_ones_ptr, head_dim * 2),
            splitk_partials_m: raw_alloc.alloc_f32(device, splitk_ms_n)?,
            splitk_partials_s: raw_alloc.alloc_f32(device, splitk_ms_n)?,
            splitk_partials_o: raw_alloc.alloc_f32(device, splitk_o_n)?,
        };
        let hidden_ptr = raw_alloc.alloc_f16(device, hidden)?.0;
        let partial_attn = raw_alloc.alloc_f16(device, hidden)?.0;
        let partial_ffn = raw_alloc.alloc_f16(device, hidden)?.0;

        let has_full_attn_layers = cfg.moe.is_some()
            && layers_global.iter().any(|&gi| !layout.layers[gi].is_swa);
        let (partial_attn_f32, attn_normed_f32_tmp) = if has_full_attn_layers {
            (
                raw_alloc.alloc_f32(device, hidden)?.0,
                raw_alloc.alloc_f32(device, hidden)?.0,
            )
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };

        let tp_moe_scratch = if let Some(moe_dims) = cfg.moe {
            let local_inter = moe_dims.moe_intermediate_size / tp_size;
            Some(crate::tp_moe_upload::Gemma4TpMoeScratch::alloc(
                device,
                hidden,
                local_inter,
                moe_dims.num_experts,
                moe_dims.num_experts_per_tok,
                raw_alloc,
            )?)
        } else {
            None
        };

        let output_head_scratch = if is_head_rank {
            Some(OutputHeadScratch {
                x_norm_f16: raw_alloc.alloc_f16(device, hidden)?.0,
                x_q8_1: raw_alloc.alloc_q8_1(device, x_q8_1_n)?.0,
                logits_f32: raw_alloc.alloc_f32(device, cfg.vocab_size)?.0,
            })
        } else {
            None
        };

        let core = TpRankCore::new(rank_in_stage, device.id())
            .map_err(|e| anyhow!("HybridRankSession core: {e}"))?;

        Ok(Self {
            common,
            kv_caches,
            hidden: hidden_ptr,
            partial_attn,
            partial_ffn,
            partial_attn_f32,
            attn_normed_f32_tmp,
            tp_moe_scratch,
            core,
            scratch,
            output_head_scratch,
            positions_host: vec![0i32; 1],
        })
    }
}

impl Gemma4HybridStage {
    /// Build the per-stage rank state + OpsRegistry vector. The
    /// sub-cluster + AR live on the driver's [`HybridCluster`]; pass
    /// the same `sub_cluster` here for the bind/registry walk.
    pub fn new(
        stage_idx: usize,
        sub_cluster: &Arc<HipCluster>,
        layers_global: Vec<usize>,
        rank_state: Vec<HybridRankState>,
    ) -> Result<Self> {
        let n_ranks = sub_cluster.ranks();
        if rank_state.len() != n_ranks {
            bail!(
                "Gemma4HybridStage: sub_cluster has {n_ranks} ranks but {} rank states",
                rank_state.len()
            );
        }
        let mut regs = Vec::with_capacity(n_ranks);
        for r in 0..n_ranks {
            let dev = sub_cluster.device(r);
            dev.bind()?;
            regs.push(OpsRegistry::new(dev).map_err(|e| anyhow!("registry stage {stage_idx} rank {r}: {e}"))?);
        }
        let mut model_rank = Vec::with_capacity(n_ranks);
        let mut session_rank = Vec::with_capacity(n_ranks);
        for rs in rank_state.into_iter() {
            let (m, s) = rs.into_halves();
            model_rank.push(m);
            session_rank.push(s);
        }
        Ok(Self {
            model: Gemma4HybridModelStage {
                stage_idx,
                layers_global: layers_global.clone(),
                rank_state: model_rank,
                regs,
            },
            session: Gemma4HybridSessionStage {
                stage_idx,
                rank_state: session_rank,
            },
        })
    }
}

impl Gemma4HybridDriver {
    /// `tp_size` is uniform across stages in S10-A. `global_cluster`
    /// has `n_stages * tp_size` ranks. `stages[s]` has its own
    /// sub-cluster of `tp_size` ranks. The global rank for
    /// `(stage=s, rank=r)` is `s * tp_size + r`.
    pub fn from_pieces(
        hc: HybridCluster,
        cfg: Gemma4Config,
        layout: ModelLayout,
        stages: Vec<Gemma4HybridStage>,
        layer_to_stage: Vec<usize>,
        head_stage_idx: usize,
        head_rank_in_head_stage_idx: usize,
    ) -> Result<Self> {
        let n_stages = stages.len();
        if n_stages == 0 {
            bail!("Gemma4HybridDriver: 0 stages");
        }
        if n_stages != hc.n_stages() {
            bail!(
                "Gemma4HybridDriver: stages {n_stages} != HybridCluster n_stages {}",
                hc.n_stages()
            );
        }
        if head_stage_idx >= n_stages {
            bail!("Gemma4HybridDriver: head_stage_idx {head_stage_idx} OOB");
        }
        if head_rank_in_head_stage_idx >= hc.tp_size() {
            bail!(
                "Gemma4HybridDriver: head_rank_in_head_stage_idx {head_rank_in_head_stage_idx} OOB"
            );
        }
        let logits_host = vec![0.0f32; cfg.vocab_size];

        let mut model_stages = Vec::with_capacity(n_stages);
        let mut session_stages = Vec::with_capacity(n_stages);
        for stage in stages.into_iter() {
            let (m, s) = stage.into_halves();
            model_stages.push(m);
            session_stages.push(s);
        }
        let model = Gemma4HybridModel {
            hc,
            cfg,
            layout,
            stages: model_stages,
            layer_to_stage,
            head_stage_idx,
            head_rank_in_head_stage_idx,
        };
        let session = Gemma4HybridSession {
            stages: session_stages,
            logits_host,
        };
        Ok(Self {
            model: std::sync::Arc::new(model),
            session,
        })
    }

    /// Real-GGUF upload entry. Uses [`partition_layers_pp`] for the
    /// stage assignment; head stage = last stage, head rank = 0 (the
    /// TP convention — every rank in the head stage has output_norm /
    /// lm_head replicated, but rank 0 owns the head-rank slot).
    pub fn upload(
        file: &flambeau_quant::GgufFile,
        cfg: Gemma4Config,
        layout: ModelLayout,
        hc: HybridCluster,
        max_tokens: usize,
    ) -> Result<Self> {
        let n_stages = hc.n_stages();
        let tp_size = hc.tp_size();
        if n_stages == 0 || tp_size == 0 {
            bail!("Gemma4HybridDriver::upload: n_stages={n_stages} tp_size={tp_size}");
        }
        // Same shardability invariants as TP path.
        crate::tp::Gemma4TpStage::validate_shardable(&cfg, tp_size)?;
        if let Some(moe) = &cfg.moe {
            if moe.moe_intermediate_size % tp_size != 0 {
                bail!(
                    "Gemma4HybridDriver::upload: moe_intermediate_size {} not divisible by tp_size {}",
                    moe.moe_intermediate_size,
                    tp_size
                );
            }
        }
        if cfg.per_layer_embed.is_some() {
            bail!("Gemma4HybridDriver::upload: per-layer-embd is followup work");
        }
        for spec in &layout.layers {
            if !spec.has_kv {
                bail!(
                    "Gemma4HybridDriver::upload: shared-KV tail layer {} unsupported (S10-H)",
                    spec.index
                );
            }
        }

        let layer_to_stage = partition_layers_pp(n_stages, &layout)?;
        // Per-stage `layers_global` (ascending).
        let mut layers_per_stage: Vec<Vec<usize>> = vec![Vec::new(); n_stages];
        for (i, &s) in layer_to_stage.iter().enumerate() {
            layers_per_stage[s].push(i);
        }
        let head_stage_idx = n_stages - 1;
        let head_rank_in_head_stage_idx = 0usize;

        let mut stages: Vec<Gemma4HybridStage> = Vec::with_capacity(n_stages);
        for s in 0..n_stages {
            let sub_cluster = hc.stage(s).sub_cluster.clone();
            let layers_global = layers_per_stage[s].clone();
            let is_stage_0 = s == 0;
            let is_head_stage = s == head_stage_idx;
            let mut rank_state: Vec<HybridRankState> = Vec::with_capacity(tp_size);
            for r in 0..tp_size {
                let device = sub_cluster.device(r);
                device.bind()?;
                let is_head_rank = is_head_stage && r == head_rank_in_head_stage_idx;
                let rs = upload_one_hybrid_stage_rank(
                    file,
                    &cfg,
                    &layout,
                    &layers_global,
                    r,
                    tp_size,
                    device,
                    max_tokens,
                    is_stage_0,
                    is_head_rank,
                )
                .with_context(|| format!("stage {s} rank {r} hybrid upload"))?;
                rank_state.push(rs);
            }
            let stage = Gemma4HybridStage::new(s, &sub_cluster, layers_global, rank_state)?;
            stages.push(stage);
        }

        Self::from_pieces(
            hc,
            cfg,
            layout,
            stages,
            layer_to_stage,
            head_stage_idx,
            head_rank_in_head_stage_idx,
        )
    }

    pub fn tp_size(&self) -> usize {
        self.model.hc.tp_size()
    }

    pub fn global_cluster(&self) -> &Arc<HipCluster> {
        self.model.hc.global_cluster()
    }

    /// Map (stage, rank_in_stage) → global rank in `global_cluster`.
    pub fn global_rank(&self, stage: usize, rank_in_stage: usize) -> usize {
        self.model.hc.global_rank_of(stage, rank_in_stage)
    }

    pub fn dispose(&mut self) -> Result<()> {
        self.session.dispose(&self.model)?;
        match std::sync::Arc::get_mut(&mut self.model) {
            Some(m) => m.dispose()?,
            None => {
                tracing::warn!(
                    "Gemma4HybridDriver::dispose: model Arc has other refs; \
                     model weights leak until all Sessions drop"
                );
            }
        }
        Ok(())
    }

    pub fn forward_one_token(&mut self, token_id: u32, position: usize) -> Result<u32> {
        forward_one_token_hybrid(self, token_id, position)?;
        let mut best_i = 0u32;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in self.session.logits_host.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i as u32;
            }
        }
        Ok(best_i)
    }
}

impl flambeau_runtime::ModelDriver for Gemma4HybridDriver {
    fn forward_prefill(&mut self, tokens: &[u32], start_position: usize) -> Result<u32> {
        // Hybrid has no batched-prefill entry; feed tokens
        // one-at-a-time through forward_one_token. Returns the
        // argmax of the final token.
        if tokens.is_empty() {
            bail!("Gemma4HybridDriver::forward_prefill: empty tokens");
        }
        let mut last: u32 = 0;
        for (i, &t) in tokens.iter().enumerate() {
            last = Gemma4HybridDriver::forward_one_token(self, t, start_position + i)?;
        }
        Ok(last)
    }
    fn forward_one_token(&mut self, token_id: u32, position: usize) -> Result<u32> {
        Gemma4HybridDriver::forward_one_token(self, token_id, position)
    }
    fn forward_prefill_logits(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        if tokens.is_empty() {
            bail!("Gemma4HybridDriver::forward_prefill_logits: empty tokens");
        }
        for (i, &t) in tokens.iter().enumerate() {
            let _ = Gemma4HybridDriver::forward_one_token(self, t, start_position + i)?;
        }
        logits_out.clear();
        logits_out.extend_from_slice(&self.session.logits_host);
        Ok(())
    }
    fn forward_one_token_logits(
        &mut self,
        token_id: u32,
        position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        let _ = Gemma4HybridDriver::forward_one_token(self, token_id, position)?;
        logits_out.clear();
        logits_out.extend_from_slice(&self.session.logits_host);
        Ok(())
    }
    fn vocab_size(&self) -> usize {
        self.model.cfg.vocab_size
    }
    fn dispose(&mut self) -> Result<()> {
        Gemma4HybridDriver::dispose(self)
    }
}

impl Drop for Gemma4HybridDriver {
    fn drop(&mut self) {
        // Best-effort; stages warn themselves if leaked.
    }
}

#[allow(clippy::too_many_arguments)]
fn upload_one_hybrid_stage_rank(
    file: &flambeau_quant::GgufFile,
    cfg: &Gemma4Config,
    layout: &ModelLayout,
    layers_global: &[usize],
    rank: usize,
    n_ranks: usize,
    device: &HipDevice,
    max_tokens: usize,
    is_stage_0: bool,
    is_head_rank: bool,
) -> Result<HybridRankState> {
    let stream = device.default_stream();
    let mut tracker = RawAllocTracker::new();

    // Globals via typed roles. Stage 0 owns token_embd; head-stage's
    // head rank owns output_norm + (untied) lm_head. `world=1` for
    // globals: the per-rank replica is uploaded as-is.
    let token_embd_dims: [usize; 2] = {
        let info = file
            .tensors
            .get("token_embd.weight")
            .ok_or_else(|| anyhow!("token_embd missing"))?;
        [info.dims[0] as usize, info.dims[1] as usize]
    };
    let mut up = WeightUploader {
        device,
        stream,
        tracker: &mut tracker,
        file,
        cfg,
        world: 1,
        rank: 0,
    };
    // Stage 0 ranks need token_embd for input embedding lookup. The
    // head-stage's head rank ALSO needs token_embd when tied
    // (`output_head_for_hybrid` reads `lm_head.or(token_embd)`), so
    // upload a per-rank replica there too (matches PP's pattern).
    let needs_token_embd = is_stage_0 || (is_head_rank && cfg.tied_lm_head);
    let token_embd = if needs_token_embd {
        Some(crate::tp::uploaded_to_device_tensor(
            up.upload_required::<TokenEmbd>(0)?,
        ))
    } else {
        None
    };
    let (output_norm, lm_head) = if is_head_rank {
        let on = crate::tp::uploaded_to_device_tensor(up.upload_required::<OutputNorm>(0)?);
        let lm = if cfg.tied_lm_head {
            None
        } else {
            up.upload::<LmHead>(0)?
                .map(crate::tp::uploaded_to_device_tensor)
        };
        (Some(on), lm)
    } else {
        (None, None)
    };
    drop(up);

    // Per-assigned-layer sharded weights.
    let mut layer_weights = Vec::with_capacity(layers_global.len());
    for &gi in layers_global {
        let spec = &layout.layers[gi];
        let lw = crate::tp::upload_layer_tp(
            file, spec, cfg, rank, n_ranks, device, stream, &mut tracker,
        )
        .with_context(|| format!("layer {gi}"))?;
        layer_weights.push(lw);
    }

    let lm_head_dims = if is_head_rank {
        Some(token_embd_dims)
    } else {
        None
    };

    let token_embd_dims_opt = if needs_token_embd {
        Some(token_embd_dims)
    } else {
        None
    };

    HybridRankState::from_pieces(
        device,
        rank,
        cfg,
        layout,
        layers_global,
        n_ranks,
        layer_weights,
        token_embd,
        token_embd_dims_opt,
        output_norm,
        lm_head,
        lm_head_dims,
        is_head_rank,
        max_tokens,
        tracker,
    )
}

fn forward_layer_decode_hybrid(
    driver: &mut Gemma4HybridDriver,
    stage_idx: usize,
    il_in_stage: usize,
    position: usize,
) -> Result<()> {
    let global_il = driver.model.stages[stage_idx].layers_global[il_in_stage];
    if driver.model.layout.layers[global_il].ffn_kind == FfnKind::Moe {
        return forward_layer_decode_hybrid_moe(driver, stage_idx, il_in_stage, position);
    }
    let cfg = driver.model.cfg.clone();
    // Split-borrow: `driver.model.stages` (shared) + `driver.session.stages`
    // (mut) live on distinct fields and can be borrowed independently
    // from `driver.model.hc` (shared).
    let hc = &driver.model.hc;
    let stage_cluster = hc.stage(stage_idx);
    let sub_cluster = &stage_cluster.sub_cluster;
    let ar = &stage_cluster.ar;
    let model_stage = &driver.model.stages[stage_idx];
    let session_stage = &mut driver.session.stages[stage_idx];
    let n_ranks = sub_cluster.ranks();
    if n_ranks != 2 {
        bail!("forward_layer_decode_hybrid: TP{n_ranks} not supported in S10-A; only tp2");
    }
    let spec = driver.model.layout.layers[global_il];

    let hidden = cfg.hidden_size;
    let head_dim = spec.head_dim;
    let n_heads_local = spec.n_heads / n_ranks;
    let n_kv_local = spec.n_kv_heads / n_ranks;
    let q_width_local = n_heads_local * head_dim;
    let kv_width_local = n_kv_local * head_dim;
    let ff_len = cfg.feed_forward_length;
    let ff_len_local = ff_len / n_ranks;
    let window: i32 = spec.window as i32;
    let softmax_scale: f32 = 1.0;
    let rms_eps = cfg.rms_norm_eps;

    // Phase 1: per-rank pre-AR fragment. Same delegation pattern as
    // `tp::forward_layer_decode_tp` Phase 1 — full kernel sequence
    // (norm → Q proj → K proj → V (proj or alt-V from K) →
    // Q/K/V per-head norm → RoPE Q+K → KV append → attention →
    // output projection → cast) handed off to
    // `flambeau_blocks::StandardAttention::forward_decode`.
    for r in 0..n_ranks {
        let dev = sub_cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &model_stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_rs = &model_stage.rank_state[r]; let session_rs = &mut session_stage.rank_state[r];
        let weights = &model_rs.layer_weights[il_in_stage];
        let x_in = session_rs.hidden;
        let attn_k = weights
            .attn_k
            .as_ref()
            .ok_or_else(|| anyhow!("layer {global_il}: attn_k missing"))?;
        let attn_k_norm_w = weights
            .attn_k_norm
            .ok_or_else(|| anyhow!("layer {global_il}: attn_k_norm missing"))?;

        let attn_q_handle = WeightHandle {
            ptr: weights.attn_q.ptr,
            dtype: weights.attn_q.dtype,
            dims: [q_width_local, hidden],
        };
        let attn_k_handle = WeightHandle {
            ptr: attn_k.ptr,
            dtype: attn_k.dtype,
            dims: [kv_width_local, hidden],
        };
        let attn_v_handle = weights.attn_v.as_ref().map(|v| WeightHandle {
            ptr: v.ptr,
            dtype: v.dtype,
            dims: [kv_width_local, hidden],
        });
        let attn_output_handle = WeightHandle {
            ptr: weights.attn_output.ptr,
            dtype: weights.attn_output.dtype,
            dims: [hidden, q_width_local],
        };
        let block = StandardAttention::new(
            attn_q_handle,
            attn_k_handle,
            attn_v_handle,
            attn_output_handle,
            weights.attn_norm,
            weights.attn_q_norm,
            attn_k_norm_w,
            hidden,
            n_heads_local,
            n_kv_local,
            head_dim,
            rms_eps,
            spec.rope_freq_base,
            spec.rope_dim,
            /* gated = */ false,
        )
        .context("StandardAttention::new (gemma4 hybrid)")?
        .with_softmax_scale(softmax_scale)
        .with_v_norm_w(session_rs.scratch.v_ones_f16.0);
        let block = if window > 0 {
            block.with_window_size(window as u32)
        } else {
            block
        };

        let kv = session_rs.kv_caches[il_in_stage]
            .as_mut()
            .expect("S10-A requires per-layer KV");
        let mut std_scratch = StandardAttentionDecodeScratch {
            x_q8_1: session_rs.scratch.x_q8_1.0,
            mmvq_f32: session_rs.scratch.mmvq_f32.0,
            q_fused_f16: DevicePtr(0),
            q_f16: session_rs.scratch.q_f16.0,
            gate_f16: DevicePtr(0),
            k_f16: session_rs.scratch.k_f16.0,
            v_f16: session_rs.scratch.v_f16.0,
            k_q8_0: DevicePtr(0),
            v_q8_0: DevicePtr(0),
            attn_out_f16: session_rs.scratch.attn_out_local.0,
            gated_out_f16: DevicePtr(0),
            positions: session_rs.scratch.positions.0,
            positions_host: &mut session_rs.positions_host,
            splitk_partials_m: session_rs.scratch.splitk_partials_m.0,
            splitk_partials_s: session_rs.scratch.splitk_partials_s.0,
            splitk_partials_o: session_rs.scratch.splitk_partials_o.0,
        };
        block.forward_decode(
            &ops,
            dev,
            stream,
            x_in,
            session_rs.partial_attn,
            kv,
            &mut std_scratch,
            position,
            /* slots = */ None,
        )
        .context("StandardAttention::forward_decode (gemma4 hybrid)")?;
    }

    // Phase 2: typed AR over the stage's sub-cluster — same pattern as
    // gemma4 tp.rs Phase-2 (commit 20ce15b) but per-stage.
    // SAFETY (Buffer::from_raw_unchecked + tp_allreduce_sum): each
    // partial_attn is hidden F16 elems on its rank's device; streams
    // outlive the AR; subsequent reads are serialised on each rank's
    // default stream.
    let _replicated = unsafe {
        let partials: [Buffer<F16, RowParallel<0>>; 2] = [
            Buffer::from_raw_unchecked(session_stage.rank_state[0].partial_attn, hidden),
            Buffer::from_raw_unchecked(session_stage.rank_state[1].partial_attn, hidden),
        ];
        let streams: [&_; 2] = [
            sub_cluster.device(0).default_stream(),
            sub_cluster.device(1).default_stream(),
        ];
        tp_allreduce_sum::<0>(ar, &partials, &streams)
    }
    .map_err(|e| anyhow!("AR sum attn stage {stage_idx}: {e}"))?;

    // Phase 3: per-rank post_attention_norm + residual.
    for r in 0..n_ranks {
        let dev = sub_cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &model_stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_rs = &model_stage.rank_state[r]; let session_rs = &mut session_stage.rank_state[r];
        let weights = &model_rs.layer_weights[il_in_stage];
        let scratch = &mut session_rs.scratch;
        ops.rmsnorm_f16(
            session_rs.partial_attn,
            weights.post_attention_norm,
            scratch.attn_out_local.0,
            1,
            hidden,
            rms_eps,
        )?;
        ops.add_f16(
            session_rs.hidden,
            scratch.attn_out_local.0,
            scratch.attn_residual_f16.0,
            hidden,
        )?;
    }

    // Phase 4: per-rank FFN pre-AR fragment. Same delegation pattern
    // as `tp::forward_layer_decode_tp` Phase 4 — ffn_norm + quant
    // inline (block API doesn't own the norm), gate/up/activation/
    // down/cast through `DenseMlpTp`.
    for r in 0..n_ranks {
        let dev = sub_cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &model_stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_rs = &model_stage.rank_state[r]; let session_rs = &mut session_stage.rank_state[r];
        let weights = &model_rs.layer_weights[il_in_stage];
        let scratch = &mut session_rs.scratch;
        ops.rmsnorm_quant_q8_1(
            scratch.attn_residual_f16.0,
            weights.ffn_norm,
            scratch.x_q8_1.0,
            1,
            hidden,
            rms_eps,
        )?;
        let ffn_gate = WeightHandle {
            ptr: weights.ffn_gate.ptr,
            dtype: weights.ffn_gate.dtype,
            dims: [ff_len_local, hidden],
        };
        let ffn_up = WeightHandle {
            ptr: weights.ffn_up.ptr,
            dtype: weights.ffn_up.dtype,
            dims: [ff_len_local, hidden],
        };
        let ffn_down = WeightHandle {
            ptr: weights.ffn_down.ptr,
            dtype: weights.ffn_down.dtype,
            dims: [hidden, ff_len_local],
        };
        let block = DenseMlpTp::new(
            ffn_gate, ffn_up, ffn_down, hidden, ff_len_local, Activation::Gelu,
        )
        .context("DenseMlpTp::new (gemma4 hybrid)")?;
        let block_scratch = DenseMlpDecodeScratch {
            x_q8_1: scratch.x_q8_1.0,
            gate_f32: scratch.gate_f32.0,
            up_f32: scratch.up_f32.0,
            activated_f16: scratch.activated_f16.0,
            activated_q8_1: scratch.activated_q8_1.0,
            down_f32: scratch.mmvq_f32.0,
            down_f16: DevicePtr(0),
        };
        block.forward_decode(
            &ops,
            DevicePtr(0),
            session_rs.partial_ffn,
            block_scratch,
            /* pre_quantized = */ true,
        )
        .context("DenseMlpTp::forward_decode (gemma4 hybrid)")?;
    }

    // Phase 5: typed AR FFN — same pattern as Phase 2 above.
    // SAFETY: same as Phase 2.
    let _replicated_ffn = unsafe {
        let partials_ffn: [Buffer<F16, RowParallel<0>>; 2] = [
            Buffer::from_raw_unchecked(session_stage.rank_state[0].partial_ffn, hidden),
            Buffer::from_raw_unchecked(session_stage.rank_state[1].partial_ffn, hidden),
        ];
        let streams: [&_; 2] = [
            sub_cluster.device(0).default_stream(),
            sub_cluster.device(1).default_stream(),
        ];
        tp_allreduce_sum::<0>(ar, &partials_ffn, &streams)
    }
    .map_err(|e| anyhow!("AR sum ffn stage {stage_idx}: {e}"))?;

    // Phase 6: per-rank post_ffw_norm + residual.
    for r in 0..n_ranks {
        let dev = sub_cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &model_stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_rs = &model_stage.rank_state[r]; let session_rs = &mut session_stage.rank_state[r];
        let weights = &model_rs.layer_weights[il_in_stage];
        let scratch = &mut session_rs.scratch;
        ops.rmsnorm_f16(
            session_rs.partial_ffn,
            weights.post_ffw_norm,
            scratch.attn_out_local.0,
            1,
            hidden,
            rms_eps,
        )?;
        ops.add_f16(
            scratch.attn_residual_f16.0,
            scratch.attn_out_local.0,
            session_rs.hidden,
            hidden,
        )?;
    }
    Ok(())
}

/// MoE-aware decode-layer composer for hybrid (pp+tp). Mirrors
/// `gemma4::tp::forward_decode_layer_tp_moe` but every AR is per-stage
/// (against the sub-cluster + per-stage `BarP2pAllReduce`). Full-attn
/// layers (head_dim=512 on 26B-A4B) go through the F32 attention output
/// path; SWA layers stay on the F16 path.
fn forward_layer_decode_hybrid_moe(
    driver: &mut Gemma4HybridDriver,
    stage_idx: usize,
    il_in_stage: usize,
    position: usize,
) -> Result<()> {
    let cfg = driver.model.cfg.clone();
    let hc = &driver.model.hc;
    let stage_cluster = hc.stage(stage_idx);
    let sub_cluster = &stage_cluster.sub_cluster;
    let ar = &stage_cluster.ar;
    let model_stage = &driver.model.stages[stage_idx];
    let session_stage = &mut driver.session.stages[stage_idx];
    let n_ranks = sub_cluster.ranks();
    if n_ranks != 2 {
        bail!("forward_layer_decode_hybrid_moe: TP{n_ranks} unsupported in S10-H; only tp2");
    }
    let global_il = model_stage.layers_global[il_in_stage];
    let spec = driver.model.layout.layers[global_il];

    let hidden = cfg.hidden_size;
    let head_dim = spec.head_dim;
    let n_heads_local = spec.n_heads / n_ranks;
    let n_kv_local = spec.n_kv_heads / n_ranks;
    let ff_len = cfg.feed_forward_length;
    let ff_len_local = ff_len / n_ranks;
    let window: i32 = spec.window as i32;
    let rms_eps = cfg.rms_norm_eps;
    let is_full_attn = !spec.is_swa;

    // Phase 1: per-rank attention. Full-attn uses F32 output_proj.
    for r in 0..n_ranks {
        let dev = sub_cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &model_stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_rs = &model_stage.rank_state[r]; let session_rs = &mut session_stage.rank_state[r];
        let weights = &model_rs.layer_weights[il_in_stage];
        let x_in = session_rs.hidden;
        let block = weights
            .build_attn_block(
                &spec,
                hidden,
                n_heads_local,
                n_kv_local,
                head_dim,
                rms_eps,
                session_rs.scratch.v_ones_f16.0,
            )?;
        let block = if is_full_attn {
            block.with_f32_output_proj(true)
        } else {
            block
        };
        let block = if window > 0 {
            block.with_window_size(window as u32)
        } else {
            block
        };
        let delta_out = if is_full_attn {
            session_rs.partial_attn_f32
        } else {
            session_rs.partial_attn
        };
        let kv = session_rs.kv_caches[il_in_stage]
            .as_mut()
            .expect("S10-H requires per-layer KV");
        let mut std_scratch = StandardAttentionDecodeScratch {
            x_q8_1: session_rs.scratch.x_q8_1.0,
            mmvq_f32: session_rs.scratch.mmvq_f32.0,
            q_fused_f16: DevicePtr(0),
            q_f16: session_rs.scratch.q_f16.0,
            gate_f16: DevicePtr(0),
            k_f16: session_rs.scratch.k_f16.0,
            v_f16: session_rs.scratch.v_f16.0,
            k_q8_0: DevicePtr(0),
            v_q8_0: DevicePtr(0),
            attn_out_f16: session_rs.scratch.attn_out_local.0,
            gated_out_f16: DevicePtr(0),
            positions: session_rs.scratch.positions.0,
            positions_host: &mut session_rs.positions_host,
            splitk_partials_m: session_rs.scratch.splitk_partials_m.0,
            splitk_partials_s: session_rs.scratch.splitk_partials_s.0,
            splitk_partials_o: session_rs.scratch.splitk_partials_o.0,
        };
        block
            .forward_decode(
                &ops, dev, stream, x_in, delta_out, kv, &mut std_scratch, position,
                /* slots = */ None,
            )
            .context("StandardAttention::forward_decode (gemma4 hybrid MoE)")?;
    }

    // Phase 2: AR-sum attention partial (F32 on full-attn, F16 on SWA).
    {
        let cores: Vec<&TpRankCore> = session_stage.rank_state.iter().map(|rs| &rs.core).collect();
        let streams: [&_; 2] = [
            sub_cluster.device(0).default_stream(),
            sub_cluster.device(1).default_stream(),
        ];
        if is_full_attn {
            let partials: [DevicePtr; 2] = [
                session_stage.rank_state[0].partial_attn_f32,
                session_stage.rank_state[1].partial_attn_f32,
            ];
            // SAFETY: partial_attn_f32 owns `hidden` F32 elements per
            // rank; streams correspond to those ranks; cores carry the
            // producer_done events the synced helper records before AR.
            unsafe {
                tp_allreduce_sum_f32_synced(ar, sub_cluster, &cores, &partials, hidden, &streams)
            }
            .context("hybrid MoE AR sum partial_attn_f32 (full-attn)")?;
        } else {
            // SAFETY: partial_attn is `hidden` F16 elements per rank;
            // same ordering contract as the F32 branch.
            let _ = unsafe {
                let partials: [Buffer<F16, RowParallel<0>>; 2] = [
                    Buffer::from_raw_unchecked(session_stage.rank_state[0].partial_attn, hidden),
                    Buffer::from_raw_unchecked(session_stage.rank_state[1].partial_attn, hidden),
                ];
                tp_allreduce_sum_synced::<0>(ar, sub_cluster, &cores, &partials, &streams)
            }
            .context("hybrid MoE AR sum partial_attn (SWA)")?;
        }
    }

    // Phase 3: per-rank post_attention_norm + residual → attn_residual_f16.
    for r in 0..n_ranks {
        let dev = sub_cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &model_stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_rs = &model_stage.rank_state[r]; let session_rs = &mut session_stage.rank_state[r];
        let weights = &model_rs.layer_weights[il_in_stage];
        let scratch = &mut session_rs.scratch;
        if is_full_attn {
            let norm_f32 = weights.post_attention_norm_f32.ok_or_else(|| {
                anyhow!("layer {global_il} rank {r}: post_attention_norm_f32 missing (hybrid MoE full-attn)")
            })?;
            ops.rmsnorm_f32(
                session_rs.partial_attn_f32,
                norm_f32,
                session_rs.attn_normed_f32_tmp,
                1,
                hidden,
                rms_eps,
            )
            .context("hybrid MoE F32 post_attention_norm")?;
            ops.cast_f32_to_f16(session_rs.attn_normed_f32_tmp, scratch.attn_out_local.0, hidden)
                .context("hybrid MoE F32→F16 cast post-attn-norm")?;
            ops.add_f16(
                session_rs.hidden,
                scratch.attn_out_local.0,
                scratch.attn_residual_f16.0,
                hidden,
            )
            .context("hybrid MoE F16 residual add (full-attn)")?;
        } else {
            ops.rmsnorm_f16(
                session_rs.partial_attn,
                weights.post_attention_norm,
                scratch.attn_out_local.0,
                1,
                hidden,
                rms_eps,
            )
            .context("hybrid MoE F16 post_attention_norm (SWA)")?;
            ops.add_f16(
                session_rs.hidden,
                scratch.attn_out_local.0,
                scratch.attn_residual_f16.0,
                hidden,
            )
            .context("hybrid MoE F16 residual add (SWA)")?;
        }
    }

    // Phase 4: per-rank MoE FFN → partial_shared_mlp_f32 + partial_moe_f32.
    for r in 0..n_ranks {
        let dev = sub_cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &model_stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_rs = &model_stage.rank_state[r]; let session_rs = &mut session_stage.rank_state[r];
        let layer = &model_rs.layer_weights[il_in_stage];
        let tp_moe = layer.tp_moe.as_ref().ok_or_else(|| {
            anyhow!("layer {global_il} rank {r}: tp_moe weights missing for MoE layer (hybrid)")
        })?;
        let tp_moe_scratch = session_rs.tp_moe_scratch.as_ref().ok_or_else(|| {
            anyhow!("rank {r}: tp_moe_scratch missing for hybrid MoE forward")
        })?;
        let scratch = &session_rs.scratch;
        crate::tp_moe_upload::forward_ffn_moe_tp_per_rank(
            &ops,
            layer,
            tp_moe,
            tp_moe_scratch,
            scratch.x_q8_1.0,
            scratch.gate_f32.0,
            scratch.up_f32.0,
            scratch.activated_f16.0,
            scratch.activated_q8_1.0,
            scratch.mmvq_f32.0,
            scratch.attn_residual_f16.0,
            hidden,
            ff_len_local,
            rms_eps,
        )
        .with_context(|| format!("hybrid MoE per-rank FFN layer {global_il} rank {r}"))?;
    }

    // Phase 5a: AR-sum partial_shared_mlp_f32 (shared MLP, F32).
    {
        let cores: Vec<&TpRankCore> = session_stage.rank_state.iter().map(|rs| &rs.core).collect();
        let streams: [&_; 2] = [
            sub_cluster.device(0).default_stream(),
            sub_cluster.device(1).default_stream(),
        ];
        let sm_partials: [DevicePtr; 2] = [
            session_stage.rank_state[0]
                .tp_moe_scratch
                .as_ref()
                .ok_or_else(|| anyhow!("rank 0: tp_moe_scratch missing in hybrid Phase 5a"))?
                .partial_shared_mlp_f32,
            session_stage.rank_state[1]
                .tp_moe_scratch
                .as_ref()
                .ok_or_else(|| anyhow!("rank 1: tp_moe_scratch missing in hybrid Phase 5a"))?
                .partial_shared_mlp_f32,
        ];
        // SAFETY: matches Phase 2 F32 contract.
        unsafe {
            tp_allreduce_sum_f32_synced(ar, sub_cluster, &cores, &sm_partials, hidden, &streams)
        }
        .context("hybrid MoE AR sum partial_shared_mlp_f32")?;
    }

    // Phase 5b: per-rank rmsnorm_f32(partial_shared_mlp_f32, post_ffw_norm_1_f32) → cur_mlp_f32.
    for r in 0..n_ranks {
        let dev = sub_cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &model_stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_rs = &model_stage.rank_state[r]; let session_rs = &session_stage.rank_state[r];
        let tp_moe = model_rs.layer_weights[il_in_stage]
            .tp_moe
            .as_ref()
            .ok_or_else(|| anyhow!("layer {global_il} rank {r}: tp_moe missing in hybrid Phase 5b"))?;
        let tp_moe_scratch = session_rs
            .tp_moe_scratch
            .as_ref()
            .ok_or_else(|| anyhow!("rank {r}: tp_moe_scratch missing in hybrid Phase 5b"))?;
        ops.rmsnorm_f32(
            tp_moe_scratch.partial_shared_mlp_f32,
            tp_moe.post_ffw_norm_1_f32,
            tp_moe_scratch.cur_mlp_f32,
            1,
            hidden,
            rms_eps,
        )
        .context("hybrid MoE post_ffw_norm_1 (F32)")?;
    }

    // Phase 5c: AR-sum partial_moe_f32 (routed-MoE, F32).
    {
        let cores: Vec<&TpRankCore> = session_stage.rank_state.iter().map(|rs| &rs.core).collect();
        let streams: [&_; 2] = [
            sub_cluster.device(0).default_stream(),
            sub_cluster.device(1).default_stream(),
        ];
        let moe_partials: [DevicePtr; 2] = [
            session_stage.rank_state[0]
                .tp_moe_scratch
                .as_ref()
                .ok_or_else(|| anyhow!("rank 0: tp_moe_scratch missing in hybrid Phase 5c"))?
                .partial_moe_f32,
            session_stage.rank_state[1]
                .tp_moe_scratch
                .as_ref()
                .ok_or_else(|| anyhow!("rank 1: tp_moe_scratch missing in hybrid Phase 5c"))?
                .partial_moe_f32,
        ];
        // SAFETY: matches Phase 2 F32 contract.
        unsafe {
            tp_allreduce_sum_f32_synced(ar, sub_cluster, &cores, &moe_partials, hidden, &streams)
        }
        .context("hybrid MoE AR sum partial_moe_f32 (routed)")?;
    }

    // Phase 5d: per-rank rmsnorm_f32(partial_moe_f32, post_ffw_norm_2_f32) → cur_moe_f32.
    for r in 0..n_ranks {
        let dev = sub_cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &model_stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_rs = &model_stage.rank_state[r]; let session_rs = &session_stage.rank_state[r];
        let tp_moe = model_rs.layer_weights[il_in_stage]
            .tp_moe
            .as_ref()
            .ok_or_else(|| anyhow!("layer {global_il} rank {r}: tp_moe missing in hybrid Phase 5d"))?;
        let tp_moe_scratch = session_rs
            .tp_moe_scratch
            .as_ref()
            .ok_or_else(|| anyhow!("rank {r}: tp_moe_scratch missing in hybrid Phase 5d"))?;
        ops.rmsnorm_f32(
            tp_moe_scratch.partial_moe_f32,
            tp_moe.post_ffw_norm_2_f32,
            tp_moe_scratch.cur_moe_f32,
            1,
            hidden,
            rms_eps,
        )
        .context("hybrid MoE post_ffw_norm_2 (F32)")?;
    }

    // Phase 5e: per-rank cur_combined_f32 = cur_mlp_f32 + cur_moe_f32.
    for r in 0..n_ranks {
        let dev = sub_cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &model_stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_rs = &model_stage.rank_state[r]; let session_rs = &session_stage.rank_state[r];
        let tp_moe_scratch = session_rs
            .tp_moe_scratch
            .as_ref()
            .ok_or_else(|| anyhow!("rank {r}: tp_moe_scratch missing in hybrid Phase 5e"))?;
        ops.add_f32(
            tp_moe_scratch.cur_mlp_f32,
            tp_moe_scratch.cur_moe_f32,
            tp_moe_scratch.cur_combined_f32,
            hidden,
        )
        .context("hybrid MoE combine cur_mlp + cur_moe (F32)")?;
    }

    // Phase 6: per-rank post_ffw_norm (F32) → cast F16 → residual add → layer_output_scale.
    for r in 0..n_ranks {
        let dev = sub_cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &model_stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_rs = &model_stage.rank_state[r]; let session_rs = &mut session_stage.rank_state[r];
        let layer = &model_rs.layer_weights[il_in_stage];
        let tp_moe = layer
            .tp_moe
            .as_ref()
            .ok_or_else(|| anyhow!("layer {global_il} rank {r}: tp_moe missing in hybrid Phase 6"))?;
        let tp_moe_scratch = session_rs
            .tp_moe_scratch
            .as_ref()
            .ok_or_else(|| anyhow!("rank {r}: tp_moe_scratch missing in hybrid Phase 6"))?;
        let scratch = &session_rs.scratch;
        ops.rmsnorm_f32(
            tp_moe_scratch.cur_combined_f32,
            tp_moe.post_ffw_norm_f32,
            tp_moe_scratch.tmp_f32,
            1,
            hidden,
            rms_eps,
        )
        .context("hybrid MoE post_ffw_norm (F32)")?;
        ops.cast_f32_to_f16(tp_moe_scratch.tmp_f32, scratch.attn_out_local.0, hidden)
            .context("hybrid MoE Phase 6 cast normed F32→F16")?;
        ops.add_f16(
            scratch.attn_residual_f16.0,
            scratch.attn_out_local.0,
            session_rs.hidden,
            hidden,
        )
        .context("hybrid MoE Phase 6 residual add")?;
        apply_layer_output_scale_f16(&ops, session_rs.hidden, hidden, layer.layer_output_scale)
            .context("hybrid MoE layer_output_scale")?;
    }

    Ok(())
}

impl HybridDecodeDriver for Gemma4HybridDriver {
    fn n_stages(&self) -> usize {
        self.model.stages.len()
    }

    fn ranks_per_stage(&self, stage: usize) -> usize {
        self.model.hc.stage(stage).sub_cluster.ranks()
    }

    fn n_layers_in_stage(&self, stage: usize) -> usize {
        self.model.stages[stage].layers_global.len()
    }

    fn head_stage(&self) -> usize {
        self.model.head_stage_idx
    }

    fn head_rank_in_head_stage(&self) -> usize {
        self.model.head_rank_in_head_stage_idx
    }

    fn bind(&self, stage: usize, rank: usize) -> Result<()> {
        self.model.hc.stage(stage).sub_cluster.device(rank).bind()?;
        Ok(())
    }

    fn embed_token(&mut self, stage: usize, rank: usize, token_id: u32) -> Result<()> {
        let cfg = self.model.cfg.clone();
        let device = self.model.hc.stage(stage).sub_cluster.device(rank);
        let model_stage = &self.model.stages[stage];
        let session_stage = &mut self.session.stages[stage];
        let stream = device.default_stream();
        let reg = &model_stage.regs[rank];
        let model_rs = &model_stage.rank_state[rank];
        let session_rs = &mut session_stage.rank_state[rank];
        let tok = model_rs
            .token_embd
            .as_ref()
            .ok_or_else(|| anyhow!("embed_token: stage {stage} rank {rank} no token_embd"))?;
        embed_token_host(
            device,
            stream,
            tok.ptr,
            tok.dtype,
            tok.bytes,
            cfg.vocab_size,
            cfg.hidden_size,
            token_id,
            session_rs.hidden,
        )?;
        // Gemma4 input scale: `inpL = scale(inpL, sqrt(n_embd))`
        // (gemma4-iswa.cpp:20). PP / TP apply this; the hybrid path
        // was missing it — caught by `parity_26b_a4b_q8_0_pp2tp2`
        // (MEMORY.md `parity_vs_argmax_in_vocab`).
        let ops = HipOps::new(reg, stream);
        ops.scale_f16(
            session_rs.hidden,
            session_rs.hidden,
            cfg.hidden_size,
            (cfg.hidden_size as f32).sqrt(),
        )
        .context("hybrid embed_token sqrt(n_embd) scale")
    }

    fn forward_layer_decode(
        &mut self,
        stage: usize,
        il_in_stage: usize,
        position: usize,
    ) -> Result<()> {
        forward_layer_decode_hybrid(self, stage, il_in_stage, position)
    }

    fn handoff_stage_to_next(&mut self, stage: usize) -> Result<()> {
        let hidden_bytes = self.model.cfg.hidden_size * 2;
        let src_global_rank = self.global_rank(stage, 0);
        let src_ptr = self.session.stages[stage].rank_state[0].hidden;
        let dst_stage = stage + 1;
        let dst_n_ranks = self.model.hc.stage(dst_stage).sub_cluster.ranks();
        // Drain the source stage's sub-cluster streams before peer_copy
        // — sub_cluster.default_stream() and global_cluster.default_stream()
        // are different `HipStream` handles for the same physical device
        // (MEMORY.md `hipcluster_stream_handles`). Without this, the
        // peer_copy_via_host DtoH (on global_cluster's stream) races
        // pending sub-cluster work and reads stale `session_rs.hidden` —
        // produces nondeterministic gibberish (#275 pattern; surfaced
        // by `parity_26b_a4b_q8_0_pp2tp2` where the heavier F32 MoE
        // cascade widened the race window).
        let src_sub = self.model.hc.stage(stage).sub_cluster.clone();
        for r in 0..src_sub.ranks() {
            let dev = src_sub.device(r);
            dev.bind()?;
            dev.default_stream().synchronize()?;
        }
        for dst_r in 0..dst_n_ranks {
            let dst_global = self.global_rank(dst_stage, dst_r);
            let dst_ptr = self.session.stages[dst_stage].rank_state[dst_r].hidden;
            // SAFETY: each `hidden` is a `hidden*2` byte F16 alloc on
            // its rank's device; the global cluster's
            // peer_copy_via_host validates source/destination ranks.
            unsafe {
                self.model.hc
                    .global_cluster()
                    .peer_copy_via_host(dst_ptr, dst_global, src_ptr, src_global_rank, hidden_bytes)
                    .map_err(|e| {
                        anyhow!(
                            "peer_copy stage {stage}→{dst_stage} rank 0 → rank {dst_r}: {e}"
                        )
                    })?;
            }
        }
        Ok(())
    }

    fn output_head(&mut self) -> Result<()> {
        let stage = self.model.head_stage_idx;
        let rank = self.model.head_rank_in_head_stage_idx;
        let cfg = &self.model.cfg;
        let device = self.model.hc.stage(stage).sub_cluster.device(rank);
        let model_stage = &self.model.stages[stage];
        let session_stage = &mut self.session.stages[stage];
        let stream = device.default_stream();
        let reg = &model_stage.regs[rank];
        let ops = HipOps::new(reg, stream);
        let model_rs = &model_stage.rank_state[rank];
        let session_rs = &mut session_stage.rank_state[rank];
        let scratch = session_rs
            .output_head_scratch
            .as_mut()
            .ok_or_else(|| anyhow!("output_head: missing scratch"))?;
        let lm_head_t = model_rs
            .lm_head
            .as_ref()
            .or(model_rs.token_embd.as_ref())
            .ok_or_else(|| anyhow!("output_head: missing LM head weight"))?;
        let lm_head_dims = model_rs
            .lm_head_dims
            .or(model_rs.token_embd_dims)
            .ok_or_else(|| anyhow!("output_head: missing dims"))?;
        let lm_head: WeightHandle = lm_head_t.as_weight_handle(lm_head_dims)?;
        let on = model_rs
            .output_norm
            .as_ref()
            .ok_or_else(|| anyhow!("output_head: missing output_norm"))?;
        let logits = forward_output_head(
            &ops,
            session_rs.hidden,
            on.ptr,
            lm_head,
            cfg.final_logit_softcap,
            scratch,
            cfg.hidden_size,
            cfg.vocab_size,
            cfg.rms_norm_eps,
        )?;
        // SAFETY: logits is vocab*4 device bytes; host buffer matches.
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToHost,
                DevicePtr(self.session.logits_host.as_mut_ptr() as usize),
                logits,
                cfg.vocab_size * 4,
            )?;
        }
        stream.synchronize()?;
        Ok(())
    }
}
