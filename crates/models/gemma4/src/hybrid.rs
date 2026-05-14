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
    embed_token_host, forward_one_token_hybrid, tp_allreduce_sum, upload_f16_ones, Activation,
    Buffer, DenseMlpDecodeScratch, DenseMlpTp, HybridCluster, HybridDecodeDriver, RawAllocTracker,
    RowParallel, StandardAttention, StandardAttentionDecodeScratch, WeightHandle, F16,
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
pub struct HybridRankState {
    pub rank_in_stage: usize,
    pub layer_weights: Vec<Gemma4LayerWeights>,
    pub kv_caches: Vec<Option<KvCache<F16Contig, HipDevice>>>,
    /// Stage 0 only.
    pub token_embd: Option<DeviceTensor>,
    pub token_embd_dims: Option<[usize; 2]>,
    /// Head stage only (replicated across that stage's ranks).
    pub output_norm: Option<DeviceTensor>,
    pub lm_head: Option<DeviceTensor>,
    pub lm_head_dims: Option<[usize; 2]>,
    pub hidden: DevicePtr,
    pub partial_attn: DevicePtr,
    pub partial_ffn: DevicePtr,
    scratch: HybridScratchPtrs,
    /// Head rank only.
    pub output_head_scratch: Option<OutputHeadScratch>,
    positions_host: Vec<i32>,
    raw_alloc: RawAllocTracker,
    disposed: bool,
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

/// One PP stage. Per-stage TP machinery (sub_cluster + AR) lives on
/// the driver's [`HybridCluster`]; this struct only carries the
/// stage's layer assignment + per-rank state + registries.
pub struct Gemma4HybridStage {
    pub stage_idx: usize,
    /// Layers in this stage (global indices, ascending).
    pub layers_global: Vec<usize>,
    pub rank_state: Vec<HybridRankState>,
    /// Per-rank `OpsRegistry` over the sub-cluster.
    regs: Vec<OpsRegistry>,
}

/// Hybrid driver. Owns the [`HybridCluster`] (sub-clusters + global
/// cluster + per-stage `BarP2pAllReduce`, all built in the correct
/// construction order) and per-stage layer / rank-state machinery.
pub struct Gemma4HybridDriver {
    pub hc: HybridCluster,
    pub cfg: Gemma4Config,
    pub layout: ModelLayout,
    pub stages: Vec<Gemma4HybridStage>,
    pub layer_to_stage: Vec<usize>,
    pub head_stage_idx: usize,
    pub head_rank_in_head_stage_idx: usize,
    logits_host: Vec<f32>,
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
    ) -> Result<Self> {
        device.bind()?;
        if layer_weights.len() != layers_global.len() {
            bail!(
                "HybridRankState: expected {} layer weights, got {}",
                layers_global.len(),
                layer_weights.len()
            );
        }
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim.max(cfg.head_dim_swa);
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
                bail!("HybridRankState: shared-KV tail layer {gi} not supported in S10-A");
            }
            if spec.ffn_kind != FfnKind::Dense {
                bail!("HybridRankState: MoE layer {gi} not supported in S10-A");
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

        let mut raw_alloc = RawAllocTracker::new();

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
        let output_head_scratch = if is_head_rank {
            Some(OutputHeadScratch {
                x_norm_f16: raw_alloc.alloc_f16(device, hidden)?.0,
                x_q8_1: raw_alloc.alloc_q8_1(device, x_q8_1_n)?.0,
                logits_f32: raw_alloc.alloc_f32(device, cfg.vocab_size)?.0,
            })
        } else {
            None
        };

        Ok(Self {
            rank_in_stage,
            layer_weights,
            kv_caches,
            token_embd,
            token_embd_dims,
            output_norm,
            lm_head,
            lm_head_dims,
            hidden: hidden_ptr,
            partial_attn,
            partial_ffn,
            scratch,
            output_head_scratch,
            positions_host: vec![0i32; 1],
            raw_alloc,
            disposed: false,
        })
    }

    fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        let kvs = std::mem::take(&mut self.kv_caches);
        for kv in kvs.into_iter().flatten() {
            kv.dispose(device).map_err(|e| anyhow!("kv dispose: {e}"))?;
        }
        self.raw_alloc
            .dispose(device)
            .map_err(|e| anyhow!("raw_alloc dispose: {e}"))?;
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
        let _ = dummy_dt(); // keep helper referenced for future use
        Ok(())
    }
}

impl Drop for HybridRankState {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                "HybridRankState rank-in-stage {} dropped without dispose()",
                self.rank_in_stage
            );
        }
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
        Ok(Self {
            stage_idx,
            layers_global,
            rank_state,
            regs,
        })
    }

    fn dispose(&mut self, sub_cluster: &Arc<HipCluster>) -> Result<()> {
        let n = sub_cluster.ranks();
        for r in 0..n {
            let dev = sub_cluster.device(r);
            self.rank_state[r].dispose(dev)?;
        }
        Ok(())
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
        Ok(Self {
            hc,
            cfg,
            layout,
            stages,
            layer_to_stage,
            head_stage_idx,
            head_rank_in_head_stage_idx,
            logits_host,
        })
    }

    pub fn tp_size(&self) -> usize {
        self.hc.tp_size()
    }

    pub fn global_cluster(&self) -> &Arc<HipCluster> {
        self.hc.global_cluster()
    }

    /// Map (stage, rank_in_stage) → global rank in `global_cluster`.
    pub fn global_rank(&self, stage: usize, rank_in_stage: usize) -> usize {
        self.hc.global_rank_of(stage, rank_in_stage)
    }

    pub fn dispose(&mut self) -> Result<()> {
        for stage in &mut self.stages {
            let sub = self.hc.stage(stage.stage_idx).sub_cluster.clone();
            stage.dispose(&sub)?;
        }
        Ok(())
    }

    pub fn forward_one_token(&mut self, token_id: u32, position: usize) -> Result<u32> {
        forward_one_token_hybrid(self, token_id, position)?;
        let mut best_i = 0u32;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in self.logits_host.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i as u32;
            }
        }
        Ok(best_i)
    }
}

impl Drop for Gemma4HybridDriver {
    fn drop(&mut self) {
        // Best-effort; stages warn themselves if leaked.
    }
}

fn forward_layer_decode_hybrid(
    driver: &mut Gemma4HybridDriver,
    stage_idx: usize,
    il_in_stage: usize,
    position: usize,
) -> Result<()> {
    let cfg = driver.cfg.clone();
    // Split-borrow: `driver.stages` (mut) and `driver.hc` (shared) live
    // on distinct fields and can be borrowed independently.
    let hc = &driver.hc;
    let stage_cluster = hc.stage(stage_idx);
    let sub_cluster = &stage_cluster.sub_cluster;
    let ar = &stage_cluster.ar;
    let stage = &mut driver.stages[stage_idx];
    let n_ranks = sub_cluster.ranks();
    if n_ranks != 2 {
        bail!("forward_layer_decode_hybrid: TP{n_ranks} not supported in S10-A; only tp2");
    }
    let global_il = stage.layers_global[il_in_stage];
    let spec = driver.layout.layers[global_il];

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
        let reg = &stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let rs = &mut stage.rank_state[r];
        let weights = &rs.layer_weights[il_in_stage];
        let x_in = rs.hidden;
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
        .with_v_norm_w(rs.scratch.v_ones_f16.0);
        let block = if window > 0 {
            block.with_window_size(window as u32)
        } else {
            block
        };

        let kv = rs.kv_caches[il_in_stage]
            .as_mut()
            .expect("S10-A requires per-layer KV");
        let mut std_scratch = StandardAttentionDecodeScratch {
            x_q8_1: rs.scratch.x_q8_1.0,
            mmvq_f32: rs.scratch.mmvq_f32.0,
            q_fused_f16: DevicePtr(0),
            q_f16: rs.scratch.q_f16.0,
            gate_f16: DevicePtr(0),
            k_f16: rs.scratch.k_f16.0,
            v_f16: rs.scratch.v_f16.0,
            k_q8_0: DevicePtr(0),
            v_q8_0: DevicePtr(0),
            attn_out_f16: rs.scratch.attn_out_local.0,
            gated_out_f16: DevicePtr(0),
            positions: rs.scratch.positions.0,
            positions_host: &mut rs.positions_host,
            splitk_partials_m: rs.scratch.splitk_partials_m.0,
            splitk_partials_s: rs.scratch.splitk_partials_s.0,
            splitk_partials_o: rs.scratch.splitk_partials_o.0,
        };
        block.forward_decode(
            &ops,
            dev,
            stream,
            x_in,
            rs.partial_attn,
            kv,
            &mut std_scratch,
            position,
            /* slots = */ None,
        )
        .context("StandardAttention::forward_decode (gemma4 hybrid)")?;
    }

    // Phase 2: typed AR over the stage's sub-cluster — same pattern as
    // gemma4 tp.rs Phase-2 (commit 20ce15b) but per-stage.
    let partials: [Buffer<F16, RowParallel<0>>; 2] = [
        Buffer::from_raw_unchecked(stage.rank_state[0].partial_attn, hidden),
        Buffer::from_raw_unchecked(stage.rank_state[1].partial_attn, hidden),
    ];
    let streams: [&_; 2] = [
        sub_cluster.device(0).default_stream(),
        sub_cluster.device(1).default_stream(),
    ];
    // SAFETY: each partial_attn is hidden F16 elems on its rank's device.
    let _replicated = unsafe {
        tp_allreduce_sum::<0>(ar, &partials, &streams)
    }
    .map_err(|e| anyhow!("AR sum attn stage {stage_idx}: {e}"))?;

    // Phase 3: per-rank post_attention_norm + residual.
    for r in 0..n_ranks {
        let dev = sub_cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let rs = &mut stage.rank_state[r];
        let weights = &rs.layer_weights[il_in_stage];
        let scratch = &mut rs.scratch;
        ops.rmsnorm_f16(
            rs.partial_attn,
            weights.post_attention_norm,
            scratch.attn_out_local.0,
            1,
            hidden,
            rms_eps,
        )?;
        ops.add_f16(
            rs.hidden,
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
        let reg = &stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let rs = &mut stage.rank_state[r];
        let weights = &rs.layer_weights[il_in_stage];
        let scratch = &mut rs.scratch;
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
            rs.partial_ffn,
            block_scratch,
            /* pre_quantized = */ true,
        )
        .context("DenseMlpTp::forward_decode (gemma4 hybrid)")?;
    }

    // Phase 5: typed AR FFN — same pattern as Phase 2 above.
    let partials_ffn: [Buffer<F16, RowParallel<0>>; 2] = [
        Buffer::from_raw_unchecked(stage.rank_state[0].partial_ffn, hidden),
        Buffer::from_raw_unchecked(stage.rank_state[1].partial_ffn, hidden),
    ];
    let streams: [&_; 2] = [
        sub_cluster.device(0).default_stream(),
        sub_cluster.device(1).default_stream(),
    ];
    // SAFETY: same as phase 2.
    let _replicated_ffn = unsafe {
        tp_allreduce_sum::<0>(ar, &partials_ffn, &streams)
    }
    .map_err(|e| anyhow!("AR sum ffn stage {stage_idx}: {e}"))?;

    // Phase 6: per-rank post_ffw_norm + residual.
    for r in 0..n_ranks {
        let dev = sub_cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &stage.regs[r];
        let ops = HipOps::new(reg, stream);
        let rs = &mut stage.rank_state[r];
        let weights = &rs.layer_weights[il_in_stage];
        let scratch = &mut rs.scratch;
        ops.rmsnorm_f16(
            rs.partial_ffn,
            weights.post_ffw_norm,
            scratch.attn_out_local.0,
            1,
            hidden,
            rms_eps,
        )?;
        ops.add_f16(
            scratch.attn_residual_f16.0,
            scratch.attn_out_local.0,
            rs.hidden,
            hidden,
        )?;
    }
    Ok(())
}

impl HybridDecodeDriver for Gemma4HybridDriver {
    fn n_stages(&self) -> usize {
        self.stages.len()
    }

    fn ranks_per_stage(&self, stage: usize) -> usize {
        self.hc.stage(stage).sub_cluster.ranks()
    }

    fn n_layers_in_stage(&self, stage: usize) -> usize {
        self.stages[stage].layers_global.len()
    }

    fn head_stage(&self) -> usize {
        self.head_stage_idx
    }

    fn head_rank_in_head_stage(&self) -> usize {
        self.head_rank_in_head_stage_idx
    }

    fn bind(&self, stage: usize, rank: usize) -> Result<()> {
        self.hc.stage(stage).sub_cluster.device(rank).bind()?;
        Ok(())
    }

    fn embed_token(&mut self, stage: usize, rank: usize, token_id: u32) -> Result<()> {
        let cfg = self.cfg.clone();
        let device = self.hc.stage(stage).sub_cluster.device(rank);
        let stage_ref = &mut self.stages[stage];
        let stream = device.default_stream();
        let rs = &mut stage_ref.rank_state[rank];
        let tok = rs
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
            rs.hidden,
        )
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
        let hidden_bytes = self.cfg.hidden_size * 2;
        let src_global_rank = self.global_rank(stage, 0);
        let src_ptr = self.stages[stage].rank_state[0].hidden;
        let dst_stage = stage + 1;
        let dst_n_ranks = self.hc.stage(dst_stage).sub_cluster.ranks();
        for dst_r in 0..dst_n_ranks {
            let dst_global = self.global_rank(dst_stage, dst_r);
            let dst_ptr = self.stages[dst_stage].rank_state[dst_r].hidden;
            // SAFETY: each `hidden` is a `hidden*2` byte F16 alloc on
            // its rank's device; the global cluster's
            // peer_copy_via_host validates source/destination ranks.
            unsafe {
                self.hc
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
        let stage = self.head_stage_idx;
        let rank = self.head_rank_in_head_stage_idx;
        let cfg = &self.cfg;
        let device = self.hc.stage(stage).sub_cluster.device(rank);
        let stage_ref = &mut self.stages[stage];
        let stream = device.default_stream();
        let reg = &stage_ref.regs[rank];
        let ops = HipOps::new(reg, stream);
        let rs = &mut stage_ref.rank_state[rank];
        let scratch = rs
            .output_head_scratch
            .as_mut()
            .ok_or_else(|| anyhow!("output_head: missing scratch"))?;
        let lm_head_t = rs
            .lm_head
            .as_ref()
            .or(rs.token_embd.as_ref())
            .ok_or_else(|| anyhow!("output_head: missing LM head weight"))?;
        let lm_head_dims = rs
            .lm_head_dims
            .or(rs.token_embd_dims)
            .ok_or_else(|| anyhow!("output_head: missing dims"))?;
        let lm_head: WeightHandle = lm_head_t.as_weight_handle(lm_head_dims)?;
        let on = rs
            .output_norm
            .as_ref()
            .ok_or_else(|| anyhow!("output_head: missing output_norm"))?;
        let logits = forward_output_head(
            &ops,
            rs.hidden,
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
                DevicePtr(self.logits_host.as_mut_ptr() as usize),
                logits,
                cfg.vocab_size * 4,
            )?;
        }
        stream.synchronize()?;
        Ok(())
    }
}
