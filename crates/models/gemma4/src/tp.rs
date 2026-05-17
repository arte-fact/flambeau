//! Tensor-parallel decode driver for Gemma 4.
//!
//! Megatron-style sharding:
//! - Q / K / V projections are **column-parallel** — each rank owns a
//!   contiguous head-shard of `n_heads / n_ranks` Q heads and
//!   `n_kv_heads / n_ranks` KV heads.
//! - Per-head Q / K / V RMSNorms operate on local heads only.
//! - RoPE applies to each rank's local Q + K (V is not rotated).
//! - KV cache holds **only this rank's KV head shard** (full sequence,
//!   sharded heads).
//! - Attention runs on local Q/K/V → partial `attn_out` of size
//!   `n_heads_local * head_dim`.
//! - Output projection is **row-parallel** — each rank multiplies its
//!   `attn_out` against `attn_output[:, rank_q_offset:rank_q_offset + q_width_local]`
//!   to produce a partial hidden contribution. AR-sum across ranks
//!   completes the projection.
//! - `post_attention_norm` + residual add then run replicated per rank.
//! - FFN gate / up are column-parallel; FFN down is row-parallel +
//!   AR-summed; `post_ffw_norm` + residual add replicate.
//!
//! Limitations of S9-A:
//! - Decode only (TP prefill = S9-B).
//! - Dense FFN only (MoE TP path lands with S6-B + #18).
//! - Shared-KV tail layers bail (S9-B follow-up: K-replicated path for
//!   cross-shard KV reads).
//! - `attn_v.is_none()` (alt-attention V=K): V copy is done at the
//!   pre-norm step from K's local shard — works without changes.
//! - Per-layer side-channel embedding bails (S5-B-2 follow-up).
//! - Layer output scale is not applied.

#![cfg(feature = "hip")]

use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use flambeau_backend_hip::{HipCluster, HipDevice};
use flambeau_blocks::{
    embed_token_host, forward_one_token_tp, post_norm_residual_f16, upload_f16_ones, AttnK,
    AttnKNorm, AttnNorm, AttnOutput, AttnQ, AttnQNorm, AttnV, FfnDown, FfnGate, FfnNorm, FfnUp,
    LayerComposerTp, LmHead, OutputNorm, PostAttnNorm, PostFfwNorm, StageCommon, TokenEmbd,
    TpRankCore, Activation, DenseMlpDecodeScratch, DenseMlpTp, RawAllocTracker,
    StandardAttentionDecodeScratch, TpCluster, TpDecodeDriver, UploadedTensor, WeightHandle,
    WeightUploader,
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

/// Per-rank model state for a TP stage (weights only — Arc-shareable
/// across multiple concurrent `Gemma4TpSession` slots).
pub struct Gemma4TpModelStage {
    /// Shared bookkeeping for **weight** allocations on this rank.
    /// `common.raw_alloc` holds every device alloc that backs a
    /// `Gemma4LayerWeights` / `token_embd` / `output_norm` / `lm_head`
    /// tensor; `dispose(device)` frees them all in one pass.
    pub common: StageCommon,
    /// Per-layer sharded weights; ALL ranks carry weights for ALL layers.
    pub layer_weights: Vec<Gemma4LayerWeights>,
    /// Replicated token_embd (each rank holds a full copy).
    pub token_embd: DeviceTensor,
    pub token_embd_dims: [usize; 2],
    /// Replicated output_norm.
    pub output_norm: DeviceTensor,
    /// Replicated LM head (gemma4 ties to `token_embd`).
    pub lm_head: Option<DeviceTensor>,
    /// Per-layer F32 copy of `post_attention_norm`. Populated only
    /// for full-attention layers (SWA layers use the F16 weight on
    /// `layer.layer_weights[il].post_attention_norm`). `DevicePtr::NULL`
    /// for SWA layers and for non-MoE models. Points into the layer-
    /// weight uploads (model-side), not a per-session allocation.
    pub post_attention_norm_f32: Vec<DevicePtr>,
}

impl Gemma4TpModelStage {
    pub fn rank(&self) -> usize {
        self.common.rank as usize
    }

    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if self.common.is_disposed() {
            return Ok(());
        }
        self.common
            .dispose(device)
            .map_err(|e| anyhow!("model raw_alloc dispose: {e}"))?;
        for t in [
            std::mem::replace(&mut self.token_embd, dummy_dt()),
            std::mem::replace(&mut self.output_norm, dummy_dt()),
        ]
        .into_iter()
        .chain(self.lm_head.take())
        {
            if !t.ptr.is_null() && t.bytes > 0 {
                unsafe {
                    let _ = device.dealloc(t.ptr, t.bytes);
                }
            }
        }
        Ok(())
    }
}

impl Drop for Gemma4TpModelStage {
    fn drop(&mut self) {
        self.common
            .warn_on_leak("flambeau_gemma4::tp::Gemma4TpModelStage");
    }
}

/// Per-rank session state for a TP stage (KV caches + scratch + per-
/// request sync events — per request).
pub struct Gemma4TpSessionStage {
    /// Shared bookkeeping for **scratch + per-request** allocations.
    pub common: StageCommon,
    /// Per-layer KV cache (each holds this rank's local KV head shard).
    pub kv_caches: Vec<Option<KvCache<F16Contig, HipDevice>>>,
    /// F16 [hidden] hidden buffer holding the current residual stream.
    pub hidden: DevicePtr,
    /// F16 [hidden] buffer holding this rank's partial attn-out
    /// contribution after the row-parallel output proj.
    pub partial_attn: DevicePtr,
    /// F32 [hidden] buffer for the row-parallel attn-output partial
    /// on **full-attention** layers (head_dim=512 on 26B-A4B). Allocated
    /// only on MoE models (no full-attn layers ⇒ stays `DevicePtr::NULL`).
    pub partial_attn_f32: DevicePtr,
    /// F32 [hidden] staging buffer for the F32 rmsnorm output on
    /// full-attention layers.
    pub attn_normed_f32_tmp: DevicePtr,
    /// F16 [hidden] buffer holding this rank's partial FFN-out
    /// contribution after the row-parallel down proj.
    pub partial_ffn: DevicePtr,
    /// Layer scratch — sized to the local head shard.
    scratch: TpScratchPtrs,
    /// Optional output-head scratch (head rank only).
    pub output_head_scratch: Option<OutputHeadScratch>,
    /// Per-rank MoE scratch — allocated when any layer is MoE.
    pub tp_moe_scratch: Option<crate::tp_moe_upload::Gemma4TpMoeScratch>,
    /// Universal TP per-rank sync identity (rank id, device id,
    /// `producer_done_event`). Per-request because the event handle
    /// tracks this session's stream progress; concurrent sessions need
    /// distinct events.
    pub core: TpRankCore,
    positions_host: Vec<i32>,
}

impl Gemma4TpSessionStage {
    pub fn rank(&self) -> usize {
        self.common.rank as usize
    }

    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if self.common.is_disposed() {
            return Ok(());
        }
        let kvs = std::mem::take(&mut self.kv_caches);
        for kv in kvs.into_iter().flatten() {
            kv.dispose(device).map_err(|e| anyhow!("kv dispose: {e}"))?;
        }
        self.common
            .dispose(device)
            .map_err(|e| anyhow!("session raw_alloc dispose: {e}"))?;
        Ok(())
    }
}

impl Drop for Gemma4TpSessionStage {
    fn drop(&mut self) {
        self.common
            .warn_on_leak("flambeau_gemma4::tp::Gemma4TpSessionStage");
    }
}

/// Per-rank state for a TP stage. Bundles model + session halves;
/// test fixtures + `Gemma4TpDriver::from_pieces` construct this.
pub struct Gemma4TpStage {
    pub model: Gemma4TpModelStage,
    pub session: Gemma4TpSessionStage,
}

impl Gemma4TpStage {
    pub fn rank(&self) -> usize {
        self.model.rank()
    }

    pub fn into_halves(self) -> (Gemma4TpModelStage, Gemma4TpSessionStage) {
        (self.model, self.session)
    }
}

struct TpScratchPtrs {
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
    /// when `n_tokens_kv > 256`). F32 buffers shared across all
    /// per-rank layers; sized for the widest layer.
    splitk_partials_m: (DevicePtr, usize),
    splitk_partials_s: (DevicePtr, usize),
    splitk_partials_o: (DevicePtr, usize),
}

/// TP model — Arc-shareable across concurrent `Gemma4TpSession`s.
/// Owns the `TpCluster` (cluster + BarP2pAllReduce), per-rank weights,
/// and per-rank `OpsRegistry`.
pub struct Gemma4TpModel {
    pub tp: TpCluster,
    pub cfg: Gemma4Config,
    pub layout: ModelLayout,
    pub stages: Vec<Gemma4TpModelStage>,
    /// Rank that runs the LM head; head_rank == 0 in V1.
    pub head_rank: usize,
    pub regs: Vec<OpsRegistry>,
}

impl Gemma4TpModel {
    pub fn dispose(&mut self) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for (rank, stage) in self.stages.iter_mut().enumerate() {
            let dev = self.tp.cluster().device(rank);
            if let Err(e) = stage.dispose(dev) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

impl Drop for Gemma4TpModel {
    fn drop(&mut self) {
        if self
            .stages
            .iter()
            .any(|s| !s.common.is_disposed())
        {
            tracing::warn!("Gemma4TpModel dropped without dispose()");
        }
    }
}

/// Per-request TP session. Allocates per-rank KV + scratch sized for
/// `max_tokens` against the model's cluster.
pub struct Gemma4TpSession {
    pub stages: Vec<Gemma4TpSessionStage>,
    pub logits_host: Vec<f32>,
}

impl Gemma4TpSession {
    pub fn dispose(&mut self, model: &Gemma4TpModel) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for (rank, stage) in self.stages.iter_mut().enumerate() {
            let dev = model.tp.cluster().device(rank);
            if let Err(e) = stage.dispose(dev) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

/// TP driver. Bundles `Arc<Gemma4TpModel>` (shared weights) and
/// `Gemma4TpSession` (per-request KV+scratch). Single-request callers
/// construct one Driver; multi-slot callers build one `Arc<Model>` and
/// per-slot Sessions (each wrapped in a Driver via `Arc::clone`).
pub struct Gemma4TpDriver {
    pub model: std::sync::Arc<Gemma4TpModel>,
    pub session: Gemma4TpSession,
}

impl Gemma4TpStage {
    /// Validate that `n_heads` and `n_kv_heads` shard cleanly across `n_ranks`.
    pub fn validate_shardable(cfg: &Gemma4Config, n_ranks: usize) -> Result<()> {
        if n_ranks == 0 {
            bail!("Gemma4TpStage: n_ranks=0");
        }
        if cfg.num_heads % n_ranks != 0 {
            bail!(
                "Gemma4TpStage: n_heads {} not divisible by n_ranks {}",
                cfg.num_heads,
                n_ranks
            );
        }
        for (il, &kv) in cfg.num_kv_heads.iter().enumerate() {
            if kv % n_ranks != 0 {
                bail!(
                    "Gemma4TpStage: layer {il} n_kv_heads {kv} not divisible by n_ranks {n_ranks}"
                );
            }
        }
        Ok(())
    }

    /// Build per-rank scratch / KV caches / hidden buffers. Weights +
    /// global tensors are passed in pre-allocated (sliced by caller).
    /// `weight_alloc` is the [`RawAllocTracker`] the caller used to
    /// upload the weight tensors — the weights tracker is moved into
    /// the model half; scratch allocations land in the session half's
    /// separate tracker. Pass `RawAllocTracker::new()` when no weights
    /// are pre-tracked (synthetic-weight tests).
    #[allow(clippy::too_many_arguments)]
    pub fn from_pieces(
        device: &HipDevice,
        rank: usize,
        cfg: &Gemma4Config,
        layout: &ModelLayout,
        n_ranks: usize,
        layer_weights: Vec<Gemma4LayerWeights>,
        token_embd: DeviceTensor,
        token_embd_dims: [usize; 2],
        output_norm: DeviceTensor,
        lm_head: Option<DeviceTensor>,
        is_head_rank: bool,
        max_tokens: usize,
        weight_alloc: RawAllocTracker,
    ) -> Result<Self> {
        let model = Gemma4TpModelStage::from_pieces(
            device,
            rank,
            cfg,
            layer_weights,
            token_embd,
            token_embd_dims,
            output_norm,
            lm_head,
            weight_alloc,
        )?;
        let session = Gemma4TpSessionStage::from_pieces(
            device,
            rank,
            cfg,
            layout,
            n_ranks,
            is_head_rank,
            max_tokens,
        )?;
        Ok(Self { model, session })
    }

    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        self.session.dispose(device)?;
        self.model.dispose(device)?;
        Ok(())
    }
}

impl Gemma4TpModelStage {
    /// Build a TP model stage. `weight_alloc` (the tracker the caller
    /// populated during upload) is adopted into `common.raw_alloc`.
    #[allow(clippy::too_many_arguments)]
    fn from_pieces(
        device: &HipDevice,
        rank: usize,
        cfg: &Gemma4Config,
        layer_weights: Vec<Gemma4LayerWeights>,
        token_embd: DeviceTensor,
        token_embd_dims: [usize; 2],
        output_norm: DeviceTensor,
        lm_head: Option<DeviceTensor>,
        weight_alloc: RawAllocTracker,
    ) -> Result<Self> {
        device.bind()?;
        if layer_weights.len() != cfg.num_layers {
            bail!(
                "Gemma4TpModelStage: expected {} layer weights, got {}",
                cfg.num_layers,
                layer_weights.len()
            );
        }
        let mut common = StageCommon::new(rank as u32, device.id());
        common.raw_alloc = weight_alloc;
        let post_attention_norm_f32: Vec<DevicePtr> = layer_weights
            .iter()
            .map(|lw| lw.post_attention_norm_f32.unwrap_or(DevicePtr::NULL))
            .collect();
        Ok(Self {
            common,
            layer_weights,
            token_embd,
            token_embd_dims,
            output_norm,
            lm_head,
            post_attention_norm_f32,
        })
    }
}

impl Gemma4TpSessionStage {
    /// Build the per-request session stage. Allocates KV caches +
    /// per-rank scratch (`hidden`, `partial_attn`, `partial_ffn`,
    /// optional F32 attention scratch on MoE models, optional output-
    /// head scratch on head rank, optional MoE scratch) sized for
    /// `max_tokens` against `device`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_pieces(
        device: &HipDevice,
        rank: usize,
        cfg: &Gemma4Config,
        layout: &ModelLayout,
        n_ranks: usize,
        is_head_rank: bool,
        max_tokens: usize,
    ) -> Result<Self> {
        device.bind()?;

        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim.max(cfg.swa.head_dim_swa);
        let n_heads_local_max = cfg.num_heads / n_ranks;
        let n_kv_local_max = cfg
            .num_kv_heads
            .iter()
            .map(|&kv| kv / n_ranks)
            .max()
            .unwrap_or(0);
        let q_width_local_max = n_heads_local_max * head_dim;
        let kv_width_local_max = n_kv_local_max * head_dim;
        let ff_len = cfg.feed_forward_length;
        let ff_len_local = ff_len / n_ranks;

        // KV caches per layer with local head shard.
        let mut kv_caches = Vec::with_capacity(cfg.num_layers);
        for spec in &layout.layers {
            if !spec.has_kv {
                bail!(
                    "Gemma4TpSessionStage: shared-KV tail layer {}",
                    spec.index
                );
            }
            let n_kv_local = spec.n_kv_heads / n_ranks;
            let kv =
                KvCache::<F16Contig, HipDevice>::new(device, n_kv_local, spec.head_dim, max_tokens)
                    .map_err(|e| anyhow!("kv alloc layer {} rank {rank}: {e}", spec.index))?;
            kv_caches.push(Some(kv));
        }

        let mut common = StageCommon::new(rank as u32, device.id());
        let raw_alloc = &mut common.raw_alloc;

        let mmvq_max = q_width_local_max
            .max(kv_width_local_max)
            .max(hidden)
            .max(ff_len_local);
        let q8_1_n = hidden.max(ff_len_local).div_ceil(32) * 32;

        let v_ones_ptr = upload_f16_ones(device, head_dim)?;
        raw_alloc.track(v_ones_ptr, head_dim * 2);

        let splitk_ms_n = n_heads_local_max * flambeau_blocks::MAX_SPLITK_CHUNKS;
        let splitk_o_n = n_heads_local_max * flambeau_blocks::MAX_SPLITK_CHUNKS * head_dim;

        let scratch = TpScratchPtrs {
            x_q8_1: raw_alloc.alloc_q8_1(device, q8_1_n)?,
            mmvq_f32: raw_alloc.alloc_f32(device, mmvq_max)?,
            q_f16: raw_alloc.alloc_f16(device, q_width_local_max)?,
            k_f16: raw_alloc.alloc_f16(device, kv_width_local_max)?,
            v_f16: raw_alloc.alloc_f16(device, kv_width_local_max)?,
            attn_out_local: raw_alloc.alloc_f16(device, q_width_local_max.max(hidden))?,
            attn_residual_f16: raw_alloc.alloc_f16(device, hidden)?,
            gate_f32: raw_alloc.alloc_f32(device, ff_len_local)?,
            up_f32: raw_alloc.alloc_f32(device, ff_len_local)?,
            activated_f16: raw_alloc.alloc_f16(device, ff_len_local)?,
            activated_q8_1: raw_alloc.alloc_q8_1(device, ff_len_local.div_ceil(32) * 32)?,
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
            && layout.layers.iter().any(|s| !s.is_swa);
        let (partial_attn_f32, attn_normed_f32_tmp) = if has_full_attn_layers {
            (
                raw_alloc.alloc_f32(device, hidden)?.0,
                raw_alloc.alloc_f32(device, hidden)?.0,
            )
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };

        let output_head_scratch = if is_head_rank {
            Some(OutputHeadScratch {
                x_norm_f16: raw_alloc.alloc_f16(device, hidden)?.0,
                x_q8_1: raw_alloc.alloc_q8_1(device, q8_1_n)?.0,
                logits_f32: raw_alloc.alloc_f32(device, cfg.vocab_size)?.0,
            })
        } else {
            None
        };

        let tp_moe_scratch = if let Some(moe_dims) = cfg.moe {
            let local_inter = moe_dims.moe_intermediate_size / n_ranks;
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

        let core = TpRankCore::new(rank, device.id())?;

        Ok(Self {
            common,
            kv_caches,
            hidden: hidden_ptr,
            partial_attn,
            partial_attn_f32,
            attn_normed_f32_tmp,
            partial_ffn,
            scratch,
            output_head_scratch,
            tp_moe_scratch,
            core,
            positions_host: vec![0i32; 1],
        })
    }
}

fn dummy_dt() -> DeviceTensor {
    DeviceTensor {
        ptr: DevicePtr::NULL,
        dtype: GgmlDType::F32,
        bytes: 0,
    }
}

impl Gemma4TpDriver {
    pub fn from_pieces(
        cluster: Arc<HipCluster>,
        cfg: Gemma4Config,
        layout: ModelLayout,
        stages: Vec<Gemma4TpStage>,
        head_rank: usize,
    ) -> Result<Self> {
        let n_ranks = stages.len();
        if n_ranks == 0 {
            bail!("Gemma4TpDriver: 0 stages");
        }
        if cluster.ranks() != n_ranks {
            bail!(
                "Gemma4TpDriver: cluster ranks {} != stages {}",
                cluster.ranks(),
                n_ranks
            );
        }
        if head_rank >= n_ranks {
            bail!("Gemma4TpDriver: head_rank {head_rank} out of range");
        }
        Gemma4TpStage::validate_shardable(&cfg, n_ranks)?;

        let tp = TpCluster::from_arc(cluster)?;
        let mut regs = Vec::with_capacity(n_ranks);
        for r in 0..n_ranks {
            let dev = tp.cluster().device(r);
            dev.bind()?;
            regs.push(OpsRegistry::new(dev).map_err(|e| anyhow!("registry rank {r}: {e}"))?);
        }
        let logits_host = vec![0.0f32; cfg.vocab_size];

        let mut model_stages = Vec::with_capacity(n_ranks);
        let mut session_stages = Vec::with_capacity(n_ranks);
        for stage in stages.into_iter() {
            let (m, s) = stage.into_halves();
            model_stages.push(m);
            session_stages.push(s);
        }
        let model = Gemma4TpModel {
            tp,
            cfg,
            layout,
            stages: model_stages,
            head_rank,
            regs,
        };
        let session = Gemma4TpSession {
            stages: session_stages,
            logits_host,
        };
        Ok(Self {
            model: Arc::new(model),
            session,
        })
    }

    pub fn dispose(&mut self) -> Result<()> {
        self.session.dispose(&self.model)?;
        match Arc::get_mut(&mut self.model) {
            Some(m) => m.dispose()?,
            None => {
                tracing::warn!(
                    "Gemma4TpDriver::dispose: model Arc has other refs; \
                     model weights leak until all Sessions drop"
                );
            }
        }
        Ok(())
    }

    /// Upload `file` across `cluster.ranks()` TP ranks. Column-parallel
    /// weights (Q / K / V / gate / up) are sliced contiguously by output
    /// dim per rank; row-parallel weights (attn_output / ffn_down) are
    /// gathered host-side then uploaded once per rank. Replicated
    /// globals (token_embd, output_norm, LM head) are uploaded
    /// independently on every rank. Norms cast F32 → F16 at upload.
    /// Per-layer-embd and shared-KV-tail and MoE FFN are not yet
    /// supported in the TP upload path (mirrors the synthetic
    /// `from_pieces` constraints).
    pub fn upload(
        file: &flambeau_quant::GgufFile,
        cfg: Gemma4Config,
        layout: ModelLayout,
        cluster: Arc<HipCluster>,
        max_tokens: usize,
    ) -> Result<Self> {
        let n_ranks = cluster.ranks();
        if n_ranks == 0 {
            bail!("Gemma4TpDriver::upload: cluster has 0 ranks");
        }
        Gemma4TpStage::validate_shardable(&cfg, n_ranks)?;
        // MoE TP upload runs `upload_moe_layer_tp` per rank inside
        // `upload_one_tp_stage` when `cfg.moe.is_some()` AND the layer
        // spec is FfnKind::Moe. Validate divisibility here so the
        // per-layer upload doesn't fail mid-stage.
        if let Some(moe) = &cfg.moe {
            if moe.moe_intermediate_size % n_ranks != 0 {
                bail!(
                    "Gemma4TpDriver::upload: moe_intermediate_size {} not divisible by n_ranks {}",
                    moe.moe_intermediate_size,
                    n_ranks
                );
            }
        }
        if cfg.per_layer_embed.is_some() {
            bail!("Gemma4TpDriver::upload: per-layer-embd TP path is followup work");
        }
        for spec in &layout.layers {
            if !spec.has_kv {
                bail!(
                    "Gemma4TpDriver::upload: shared-KV tail layer {}",
                    spec.index
                );
            }
        }

        let mut stages: Vec<Gemma4TpStage> = Vec::with_capacity(n_ranks);
        for rank in 0..n_ranks {
            let device = cluster.device(rank);
            device.bind()?;
            let stage =
                upload_one_tp_stage(file, &cfg, &layout, rank, n_ranks, device, max_tokens)
                    .with_context(|| format!("rank {rank} TP upload"))?;
            stages.push(stage);
        }
        Self::from_pieces(cluster, cfg, layout, stages, 0)
    }

    pub fn forward_one_token(&mut self, token_id: u32, position: usize) -> Result<u32> {
        forward_one_token_tp(self, token_id, position)?;
        // Argmax host-side.
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

impl flambeau_runtime::ModelDriver for Gemma4TpDriver {
    fn forward_prefill(&mut self, tokens: &[u32], start_position: usize) -> Result<u32> {
        // TP driver has no batched-prefill entry; feed tokens one at a
        // time through `forward_one_token` and return the argmax of
        // the final token. Matches `flambeau_decode_tp` in the parity
        // tests.
        if tokens.is_empty() {
            bail!("Gemma4TpDriver::forward_prefill: empty tokens");
        }
        let mut last: u32 = 0;
        for (i, &t) in tokens.iter().enumerate() {
            last = Gemma4TpDriver::forward_one_token(self, t, start_position + i)?;
        }
        Ok(last)
    }
    fn forward_one_token(&mut self, token_id: u32, position: usize) -> Result<u32> {
        Gemma4TpDriver::forward_one_token(self, token_id, position)
    }
    fn forward_prefill_logits(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        if tokens.is_empty() {
            bail!("Gemma4TpDriver::forward_prefill_logits: empty tokens");
        }
        for (i, &t) in tokens.iter().enumerate() {
            let _ = Gemma4TpDriver::forward_one_token(self, t, start_position + i)?;
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
        let _ = Gemma4TpDriver::forward_one_token(self, token_id, position)?;
        logits_out.clear();
        logits_out.extend_from_slice(&self.session.logits_host);
        Ok(())
    }
    fn vocab_size(&self) -> usize {
        self.model.cfg.vocab_size
    }
    fn dispose(&mut self) -> Result<()> {
        Gemma4TpDriver::dispose(self)
    }
}

impl Drop for Gemma4TpDriver {
    fn drop(&mut self) {
        let model_leaked = self
            .model
            .stages
            .iter()
            .any(|s| !s.common.is_disposed());
        let session_leaked = self
            .session
            .stages
            .iter()
            .any(|s| !s.common.is_disposed());
        if model_leaked || session_leaked {
            tracing::warn!("Gemma4TpDriver dropped without dispose()");
        }
    }
}

// ---------------------------------------------------------------------------
// Per-layer TP composition
// ---------------------------------------------------------------------------

/// One layer's TP decode. Phases (driven by
/// [`flambeau_blocks::forward_decode_layer_tp`]):
/// 1. Per-rank: attn-norm + Q/K/V proj + per-head norms + RoPE + KV
///    append + local attention + row-parallel output proj (partial).
/// 2. Barrier-fused AR-sum partial_attn → Replicated.
/// 3. Per-rank: post_attention_norm + residual add → attn_residual.
/// 4. Per-rank: ffn_norm + gate/up + GELU + row-parallel down (partial).
/// 5. Barrier-fused AR-sum partial_ffn → Replicated.
/// 6. Per-rank: post_ffw_norm + residual add → next-layer hidden.
fn forward_layer_decode_tp(
    driver: &mut Gemma4TpDriver,
    il: usize,
    position: usize,
) -> Result<()> {
    let n_ranks = driver.model.stages.len();
    // MoE layers use the parallel-branch composer (3 ARs/layer);
    // dense layers use the shared `LayerComposerTp` (2 ARs/layer).
    if driver.model.layout.layers[il].ffn_kind == FfnKind::Moe {
        forward_decode_layer_tp_moe(driver, il, position)
    } else {
        flambeau_blocks::forward_decode_layer_tp(driver, position, il)
    }
}

/// MoE-aware decode-layer composer for gemma4 26B-A4B. Re-uses the
/// dense composer's attention phases (1-3) via the `LayerComposerTp`
/// trait methods on `driver`, then forks the FFN half into the
/// 5-phase parallel-branch shape:
///
/// 4. Per-rank: `forward_ffn_moe_tp_per_rank` writes BOTH the shared
///    MLP partial (`stage.partial_ffn`) and the routed-MoE partial
///    (`tp_moe_scratch.partial_moe_f16`).
/// 5a. Barrier-fused AR-sum on `partial_ffn` (shared-MLP).
/// 5b. Per-rank: `cur_mlp_f16 = rmsnorm(partial_ffn, post_ffw_norm_1)`.
/// 5c. Barrier-fused AR-sum on `partial_moe_f16` (routed-MoE).
/// 5d. Per-rank: `cur_moe_f16 = rmsnorm(partial_moe, post_ffw_norm_2)`.
/// 5e. Per-rank: `cur_combined = cur_mlp + cur_moe`.
/// 6. Per-rank: `post_ffw_norm + residual_add` → next-layer hidden
///    (with optional `layer_output_scale`).
fn forward_decode_layer_tp_moe(
    driver: &mut Gemma4TpDriver,
    il: usize,
    position: usize,
) -> Result<()> {
    use flambeau_blocks::{
        tp_allreduce_sum_f32_synced, tp_allreduce_sum_synced, Buffer, RowParallel, F16,
    };

    let n = driver.model.stages.len();
    let hidden = driver.model.cfg.hidden_size;
    let rms_eps = driver.model.cfg.rms_norm_eps;
    let ff_len_local = driver.model.cfg.feed_forward_length / n;
    let is_full_attn = !driver.model.layout.layers[il].is_swa;

    // Phase 1: per-rank attention. Full-attention layers (head_dim=512
    // on 26B-A4B) use the F32 output_proj path so the row-parallel
    // partial doesn't saturate F16 when V has a sqrt(head_dim)≈22
    // spike (see feedback_gemma4_attn_output_proj_f16_saturate).
    if is_full_attn {
        for r in 0..n {
            forward_attn_f32_output(driver, r, position, il)?;
        }
    } else {
        for r in 0..n {
            <Gemma4TpDriver as LayerComposerTp>::forward_attn(driver, r, position, il)?;
        }
    }
    // Phase 2: AR-sum partial_attn across ranks. F32 path on
    // full-attention layers (so the F32 mmvq output stays bounded
    // through AR; cast to F16 happens only after post-norm absorbs
    // the spike).
    {
        let cluster = driver.model.tp.cluster();
        let streams: Vec<&_> = (0..n).map(|r| cluster.device(r).default_stream()).collect();
        let cores: Vec<&TpRankCore> = driver.session.stages.iter().map(|s| &s.core).collect();
        if is_full_attn {
            let partials: Vec<DevicePtr> = driver
                .session
                .stages
                .iter()
                .map(|s| s.partial_attn_f32)
                .collect();
            // SAFETY: partial_attn_f32 is hidden F32 elems per rank;
            // synced helper orders BAR1 reads behind producer events.
            unsafe {
                tp_allreduce_sum_f32_synced(
                    driver.model.tp.ar(),
                    cluster,
                    &cores,
                    &partials,
                    hidden,
                    &streams,
                )
            }
            .context("MoE AR sum partial_attn_f32 (full-attn)")?;
        } else {
            // SAFETY: partial_attn is hidden F16 elems per rank; streams
            // outlive this call; synced helper adds the cross-rank edge.
            let _ = unsafe {
                let partials: Vec<Buffer<F16, RowParallel<0>>> = driver
                    .session
                    .stages
                    .iter()
                    .map(|s| Buffer::from_raw_unchecked(s.partial_attn, hidden))
                    .collect();
                tp_allreduce_sum_synced::<0>(
                    driver.model.tp.ar(),
                    cluster,
                    &cores,
                    &partials,
                    &streams,
                )
            }
            .context("MoE AR sum partial_attn (SWA)")?;
        }
    }
    // Phase 3: post-attn-norm + residual add → attn_residual_f16.
    // Full-attn: F32 rmsnorm + cast + F16 add. SWA: trait method.
    if is_full_attn {
        for r in 0..n {
            post_norm_residual_attn_f32(driver, r, il)?;
        }
    } else {
        for r in 0..n {
            <Gemma4TpDriver as LayerComposerTp>::post_norm_residual_attn(driver, r, il)?;
        }
    }

    // ---- FFN half (Phases 4 / 5a-e / 6) — MoE-specific. ----
    // Phase 4: per-rank MoE FFN forward → two partials.
    for r in 0..n {
        let dev = driver.model.tp.cluster().device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &driver.model.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_stage = &driver.model.stages[r];
        let session_stage = &driver.session.stages[r];
        let layer = &model_stage.layer_weights[il];
        let tp_moe = layer
            .tp_moe
            .as_ref()
            .ok_or_else(|| anyhow!("layer {il} rank {r}: tp_moe weights missing for MoE layer"))?;
        let tp_moe_scratch = session_stage
            .tp_moe_scratch
            .as_ref()
            .ok_or_else(|| anyhow!("rank {r}: tp_moe_scratch missing for MoE forward"))?;
        let scratch = &session_stage.scratch;
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
        .with_context(|| format!("MoE per-rank FFN layer {il} rank {r}"))?;
    }

    // Phase 5a: AR-sum partial_shared_mlp_f32 (shared MLP, F32 path).
    // F32 AR to match the F32 down qmatmul output from
    // `DenseMlpTp::forward_decode_f32`. Pairs with the F32 MoE branch
    // and F32 attention residual on head_dim=512 + Q8_0 paths.
    {
        let cluster = driver.model.tp.cluster();
        let streams: Vec<&_> = (0..n).map(|r| cluster.device(r).default_stream()).collect();
        let cores: Vec<&TpRankCore> = driver.session.stages.iter().map(|s| &s.core).collect();
        let sm_partials: Vec<DevicePtr> = driver
            .session
            .stages
            .iter()
            .enumerate()
            .map(|(r, s)| -> Result<DevicePtr> {
                Ok(s.tp_moe_scratch
                    .as_ref()
                    .ok_or_else(|| anyhow!("rank {r}: tp_moe_scratch missing in Phase 5a"))?
                    .partial_shared_mlp_f32)
            })
            .collect::<Result<_>>()?;
        // SAFETY: partial_shared_mlp_f32 buffers own hidden*4 bytes per
        // rank; streams correspond to those ranks; cores carry the
        // producer_done events that the synced helper records.
        unsafe {
            tp_allreduce_sum_f32_synced(
                driver.model.tp.ar(),
                cluster,
                &cores,
                &sm_partials,
                hidden,
                &streams,
            )
        }
        .context("MoE AR sum partial_shared_mlp_f32")?;
    }

    // Phase 5b: `rmsnorm_f32(partial_shared_mlp_f32, post_ffw_norm_1_f32,
    // cur_mlp_f32)` — direct F32-in / F32-out (no F16→F32 cast needed).
    for r in 0..n {
        let dev = driver.model.tp.cluster().device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &driver.model.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_stage = &driver.model.stages[r];
        let session_stage = &driver.session.stages[r];
        let tp_moe = model_stage.layer_weights[il]
            .tp_moe
            .as_ref()
            .ok_or_else(|| anyhow!("layer {il} rank {r}: tp_moe missing in Phase 5b"))?;
        let tp_moe_scratch = session_stage
            .tp_moe_scratch
            .as_ref()
            .ok_or_else(|| anyhow!("rank {r}: tp_moe_scratch missing in Phase 5b"))?;
        ops.rmsnorm_f32(
            tp_moe_scratch.partial_shared_mlp_f32,
            tp_moe.post_ffw_norm_1_f32,
            tp_moe_scratch.cur_mlp_f32,
            1,
            hidden,
            rms_eps,
        )
        .context("MoE post_ffw_norm_1 (F32, direct from F32 AR)")?;
    }

    // Phase 5c: AR-sum partial_moe_f32 (routed-MoE, F32 path).
    // F32 AR matches the F32 combine output from
    // `MoeExperts::forward_decode_tp_f32` — keeps the V-norm spike
    // intact instead of clipping it through F16. Pairs with the F32
    // attention output path (commit 6b85f29).
    {
        let cluster = driver.model.tp.cluster();
        let streams: Vec<&_> = (0..n).map(|r| cluster.device(r).default_stream()).collect();
        let cores: Vec<&TpRankCore> = driver.session.stages.iter().map(|s| &s.core).collect();
        let moe_partials: Vec<DevicePtr> = driver
            .session
            .stages
            .iter()
            .enumerate()
            .map(|(r, s)| -> Result<DevicePtr> {
                Ok(s.tp_moe_scratch
                    .as_ref()
                    .ok_or_else(|| anyhow!("rank {r}: tp_moe_scratch missing in Phase 5c"))?
                    .partial_moe_f32)
            })
            .collect::<Result<_>>()?;
        // SAFETY: partial_moe_f32 buffers own hidden*4 bytes per rank;
        // streams correspond to those ranks; cores carry the
        // producer_done events that the synced helper records.
        unsafe {
            tp_allreduce_sum_f32_synced(
                driver.model.tp.ar(),
                cluster,
                &cores,
                &moe_partials,
                hidden,
                &streams,
            )
        }
        .context("MoE AR sum partial_moe_f32 (routed)")?;
    }

    // Phase 5d: `rmsnorm_f32(partial_moe_f32, post_ffw_norm_2_f32,
    // cur_moe_f32)`. Direct F32-in / F32-out — no F16 cast needed,
    // since Phase 5c AR'd in F32.
    for r in 0..n {
        let dev = driver.model.tp.cluster().device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &driver.model.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_stage = &driver.model.stages[r];
        let session_stage = &driver.session.stages[r];
        let tp_moe = model_stage.layer_weights[il]
            .tp_moe
            .as_ref()
            .ok_or_else(|| anyhow!("layer {il} rank {r}: tp_moe missing in Phase 5d"))?;
        let tp_moe_scratch = session_stage
            .tp_moe_scratch
            .as_ref()
            .ok_or_else(|| anyhow!("rank {r}: tp_moe_scratch missing in Phase 5d"))?;
        ops.rmsnorm_f32(
            tp_moe_scratch.partial_moe_f32,
            tp_moe.post_ffw_norm_2_f32,
            tp_moe_scratch.cur_moe_f32,
            1,
            hidden,
            rms_eps,
        )
        .context("MoE post_ffw_norm_2 (F32, direct from F32 AR)")?;
    }

    // Phase 5e: `cur_combined_f32 = cur_mlp_f32 + cur_moe_f32` (F32
    // add — kept in F32 through the final norm in Phase 6).
    for r in 0..n {
        let dev = driver.model.tp.cluster().device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &driver.model.regs[r];
        let ops = HipOps::new(reg, stream);
        let session_stage = &driver.session.stages[r];
        let tp_moe_scratch = session_stage
            .tp_moe_scratch
            .as_ref()
            .ok_or_else(|| anyhow!("rank {r}: tp_moe_scratch missing in Phase 5e"))?;
        ops.add_f32(
            tp_moe_scratch.cur_mlp_f32,
            tp_moe_scratch.cur_moe_f32,
            tp_moe_scratch.cur_combined_f32,
            hidden,
        )
        .context("MoE combine cur_mlp + cur_moe (F32)")?;
    }

    // Phase 6: per-rank post_ffw_norm + residual add → next-layer hidden.
    for r in 0..n {
        let dev = driver.model.tp.cluster().device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &driver.model.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_stage = &driver.model.stages[r];
        let session_stage = &mut driver.session.stages[r];
        let layer = &model_stage.layer_weights[il];
        let scratch = &session_stage.scratch;
        let tp_moe = layer
            .tp_moe
            .as_ref()
            .ok_or_else(|| anyhow!("layer {il} rank {r}: tp_moe missing in Phase 6"))?;
        let tp_moe_scratch = session_stage
            .tp_moe_scratch
            .as_ref()
            .ok_or_else(|| anyhow!("rank {r}: tp_moe_scratch missing in Phase 6"))?;
        // 1. F32 rmsnorm of the combined F32 input under
        //    `post_ffw_norm_f32` — keeps the post-norm cascade in F32.
        ops.rmsnorm_f32(
            tp_moe_scratch.cur_combined_f32,
            tp_moe.post_ffw_norm_f32,
            tp_moe_scratch.tmp_f32,
            1,
            hidden,
            rms_eps,
        )
        .context("MoE post_ffw_norm (F32)")?;
        // 2. Cast F32 norm output → F16 staging (`attn_out_local` —
        //    free after Phase 1).
        ops.cast_f32_to_f16(tp_moe_scratch.tmp_f32, scratch.attn_out_local.0, hidden)
            .context("MoE Phase 6 cast normed F32→F16")?;
        // 3. Residual add into `session_stage.hidden`. Output is F16 —
        //    the residual stream stays F16 (bounded by `layer_output_scale`).
        ops.add_f16(
            scratch.attn_residual_f16.0,
            scratch.attn_out_local.0,
            session_stage.hidden,
            hidden,
        )
        .context("MoE Phase 6 residual add")?;
        flambeau_blocks::apply_layer_output_scale_f16(
            &ops,
            session_stage.hidden,
            hidden,
            layer.layer_output_scale,
        )
        .context("MoE layer_output_scale")?;
    }

    Ok(())
}

/// Per-rank attention with F32 output projection. Used by
/// [`forward_decode_layer_tp_moe`] on full-attention layers
/// (head_dim=512 on 26B-A4B) where the F16 output_proj saturates
/// due to V_norm's structural sqrt(head_dim) spike. Writes the F32
/// mmvq result directly into `stage.partial_attn_f32`; the caller
/// AR-sums F32 across ranks via `tp_allreduce_sum_f32_synced`.
fn forward_attn_f32_output(
    driver: &mut Gemma4TpDriver,
    r: usize,
    position: usize,
    il: usize,
) -> Result<()> {
    let spec = driver.model.layout.layers[il];
    let n_ranks = driver.model.stages.len();
    let hidden = driver.model.cfg.hidden_size;
    let head_dim = spec.head_dim;
    let n_heads_local = spec.n_heads / n_ranks;
    let n_kv_local = spec.n_kv_heads / n_ranks;
    let rms_eps = driver.model.cfg.rms_norm_eps;
    let dev = driver.model.tp.cluster().device(r);
    dev.bind()?;
    let stream = dev.default_stream();
    let reg = &driver.model.regs[r];
    let ops = HipOps::new(reg, stream);
    let model_stage = &driver.model.stages[r];
    let session_stage = &mut driver.session.stages[r];
    let weights = &model_stage.layer_weights[il];
    let x_in = session_stage.hidden;
    let block = weights
        .build_attn_block(
            &spec,
            hidden,
            n_heads_local,
            n_kv_local,
            head_dim,
            rms_eps,
            session_stage.scratch.v_ones_f16.0,
        )?
        .with_f32_output_proj(true);
    let kv = session_stage.kv_caches[il]
        .as_mut()
        .expect("per-layer KV required");
    let mut std_scratch = StandardAttentionDecodeScratch {
        x_q8_1: session_stage.scratch.x_q8_1.0,
        mmvq_f32: session_stage.scratch.mmvq_f32.0,
        q_fused_f16: DevicePtr(0),
        q_f16: session_stage.scratch.q_f16.0,
        gate_f16: DevicePtr(0),
        k_f16: session_stage.scratch.k_f16.0,
        v_f16: session_stage.scratch.v_f16.0,
        k_q8_0: DevicePtr(0),
        v_q8_0: DevicePtr(0),
        attn_out_f16: session_stage.scratch.attn_out_local.0,
        gated_out_f16: DevicePtr(0),
        positions: session_stage.scratch.positions.0,
        positions_host: &mut session_stage.positions_host,
        splitk_partials_m: session_stage.scratch.splitk_partials_m.0,
        splitk_partials_s: session_stage.scratch.splitk_partials_s.0,
        splitk_partials_o: session_stage.scratch.splitk_partials_o.0,
    };
    // Pass `partial_attn_f32` (F32 [hidden]) as the delta_out — the
    // F32-output-proj block writes the F32 mmvq result directly here,
    // skipping the saturating F16 cast.
    block
        .forward_decode(
            &ops,
            dev,
            stream,
            x_in,
            session_stage.partial_attn_f32,
            kv,
            &mut std_scratch,
            position,
            /* slots = */ None,
        )
        .context("StandardAttention::forward_decode F32-output (gemma4 MoE full-attn)")
}

/// Per-rank F32 post-attention-norm + residual add. F32 rmsnorm
/// (with F32 `post_attention_norm_f32`) → F32 staging → F16 cast
/// (safe: rmsnorm output is bounded) → F16 add to stage.hidden.
fn post_norm_residual_attn_f32(driver: &mut Gemma4TpDriver, r: usize, il: usize) -> Result<()> {
    let hidden = driver.model.cfg.hidden_size;
    let rms_eps = driver.model.cfg.rms_norm_eps;
    let dev = driver.model.tp.cluster().device(r);
    dev.bind()?;
    let stream = dev.default_stream();
    let reg = &driver.model.regs[r];
    let ops = HipOps::new(reg, stream);
    let model_stage = &driver.model.stages[r];
    let session_stage = &mut driver.session.stages[r];
    let layer = &model_stage.layer_weights[il];
    let norm_f32 = layer
        .post_attention_norm_f32
        .ok_or_else(|| anyhow!("layer {il} rank {r}: post_attention_norm_f32 missing"))?;
    // 1. F32 rmsnorm: partial_attn_f32 / RMS · post_attention_norm_f32 → attn_normed_f32_tmp.
    ops.rmsnorm_f32(
        session_stage.partial_attn_f32,
        norm_f32,
        session_stage.attn_normed_f32_tmp,
        1,
        hidden,
        rms_eps,
    )
    .context("F32 post_attention_norm (full-attn)")?;
    // 2. Cast F32 → F16 (safe; rmsnorm output is bounded). Reuse
    //    `attn_out_local` as the F16 staging.
    ops.cast_f32_to_f16(
        session_stage.attn_normed_f32_tmp,
        session_stage.scratch.attn_out_local.0,
        hidden,
    )
    .context("F32→F16 cast post-attn-norm (full-attn)")?;
    // 3. F16 residual add → attn_residual_f16.
    ops.add_f16(
        session_stage.hidden,
        session_stage.scratch.attn_out_local.0,
        session_stage.scratch.attn_residual_f16.0,
        hidden,
    )
    .context("F16 residual add (full-attn)")?;
    Ok(())
}

// `LayerComposerTp` impl — model-specific per-rank hooks. The
// composer drives the 6-phase order + the typed AR transitions.
impl LayerComposerTp for Gemma4TpDriver {
    fn n_ranks(&self) -> usize {
        self.model.stages.len()
    }

    fn hidden_size(&self) -> usize {
        self.model.cfg.hidden_size
    }

    fn ar(&self) -> &flambeau_backend_hip::BarP2pAllReduce {
        self.model.tp.ar()
    }

    fn cluster(&self) -> &HipCluster {
        self.model.tp.cluster()
    }

    fn core(&self, rank: usize) -> &TpRankCore {
        &self.session.stages[rank].core
    }

    fn partial_attn_ptr(&self, rank: usize) -> DevicePtr {
        self.session.stages[rank].partial_attn
    }

    fn partial_ffn_ptr(&self, rank: usize) -> DevicePtr {
        self.session.stages[rank].partial_ffn
    }

    fn forward_attn(&mut self, r: usize, position: usize, il: usize) -> Result<()> {
        let spec = self.model.layout.layers[il];
        let n_ranks = self.model.stages.len();
        let hidden = self.model.cfg.hidden_size;
        let head_dim = spec.head_dim;
        let n_heads_local = spec.n_heads / n_ranks;
        let n_kv_local = spec.n_kv_heads / n_ranks;
        let rms_eps = self.model.cfg.rms_norm_eps;

        let dev = self.model.tp.cluster().device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &self.model.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_stage = &self.model.stages[r];
        let session_stage = &mut self.session.stages[r];
        let weights = &model_stage.layer_weights[il];
        let x_in = session_stage.hidden;
        let block = weights.build_attn_block(
            &spec,
            hidden,
            n_heads_local,
            n_kv_local,
            head_dim,
            rms_eps,
            session_stage.scratch.v_ones_f16.0,
        )?;

        let kv = session_stage.kv_caches[il]
            .as_mut()
            .expect("per-layer KV required");
        let mut std_scratch = StandardAttentionDecodeScratch {
            x_q8_1: session_stage.scratch.x_q8_1.0,
            mmvq_f32: session_stage.scratch.mmvq_f32.0,
            q_fused_f16: DevicePtr(0),
            q_f16: session_stage.scratch.q_f16.0,
            gate_f16: DevicePtr(0),
            k_f16: session_stage.scratch.k_f16.0,
            v_f16: session_stage.scratch.v_f16.0,
            k_q8_0: DevicePtr(0),
            v_q8_0: DevicePtr(0),
            attn_out_f16: session_stage.scratch.attn_out_local.0,
            gated_out_f16: DevicePtr(0),
            positions: session_stage.scratch.positions.0,
            positions_host: &mut session_stage.positions_host,
            splitk_partials_m: session_stage.scratch.splitk_partials_m.0,
            splitk_partials_s: session_stage.scratch.splitk_partials_s.0,
            splitk_partials_o: session_stage.scratch.splitk_partials_o.0,
        };
        block.forward_decode(
            &ops,
            dev,
            stream,
            x_in,
            session_stage.partial_attn,
            kv,
            &mut std_scratch,
            position,
            /* slots = */ None,
        )
        .context("StandardAttention::forward_decode (gemma4 TP)")?;
        Ok(())
    }

    fn post_norm_residual_attn(&mut self, r: usize, il: usize) -> Result<()> {
        let hidden = self.model.cfg.hidden_size;
        let rms_eps = self.model.cfg.rms_norm_eps;
        let dev = self.model.tp.cluster().device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &self.model.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_stage = &self.model.stages[r];
        let session_stage = &mut self.session.stages[r];
        let weights = &model_stage.layer_weights[il];
        let scratch = &mut session_stage.scratch;
        post_norm_residual_f16(
            &ops,
            session_stage.partial_attn,
            weights.post_attention_norm,
            scratch.attn_out_local.0,
            session_stage.hidden,
            scratch.attn_residual_f16.0,
            1,
            hidden,
            rms_eps,
        )
        .context("post_attention_norm + residual (TP)")
    }

    fn forward_ffn(&mut self, r: usize, il: usize) -> Result<()> {
        let n_ranks = self.model.stages.len();
        let hidden = self.model.cfg.hidden_size;
        let ff_len_local = self.model.cfg.feed_forward_length / n_ranks;
        let rms_eps = self.model.cfg.rms_norm_eps;
        let dev = self.model.tp.cluster().device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &self.model.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_stage = &self.model.stages[r];
        let session_stage = &mut self.session.stages[r];
        let weights = &model_stage.layer_weights[il];
        let scratch = &mut session_stage.scratch;
        ops.rmsnorm_quant_q8_1(
            scratch.attn_residual_f16.0,
            weights.ffn_norm,
            scratch.x_q8_1.0,
            1,
            hidden,
            rms_eps,
        )
        .context("ffn_norm + quant (TP)")?;
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
        .context("DenseMlpTp::new (gemma4 TP)")?;
        let block_scratch = DenseMlpDecodeScratch {
            x_q8_1: scratch.x_q8_1.0,
            gate_f32: scratch.gate_f32.0,
            up_f32: scratch.up_f32.0,
            activated_f16: scratch.activated_f16.0,
            activated_q8_1: scratch.activated_q8_1.0,
            down_f32: scratch.mmvq_f32.0,
            down_f16: DevicePtr(0), // unused on the TP partial-out path
        };
        block.forward_decode(
            &ops,
            /* x_norm = */ DevicePtr(0),
            session_stage.partial_ffn,
            block_scratch,
            /* pre_quantized = */ true,
        )
        .context("DenseMlpTp::forward_decode (gemma4 TP)")
    }

    fn post_norm_residual_ffn(&mut self, r: usize, il: usize) -> Result<()> {
        let hidden = self.model.cfg.hidden_size;
        let rms_eps = self.model.cfg.rms_norm_eps;
        let dev = self.model.tp.cluster().device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &self.model.regs[r];
        let ops = HipOps::new(reg, stream);
        let model_stage = &self.model.stages[r];
        let session_stage = &mut self.session.stages[r];
        let weights = &model_stage.layer_weights[il];
        let scratch = &mut session_stage.scratch;
        post_norm_residual_f16(
            &ops,
            session_stage.partial_ffn,
            weights.post_ffw_norm,
            scratch.attn_out_local.0,
            scratch.attn_residual_f16.0,
            session_stage.hidden,
            1,
            hidden,
            rms_eps,
        )
        .context("post_ffw_norm + residual (TP)")?;
        // Per-layer scalar `layer_output_scale` (mirrors
        // `layer.rs::forward_layer_decode` step 14). Gemma4 31B uses
        // this to keep the residual stream's magnitude bounded across
        // 60 layers; without it values explode → Inf → NaN around
        // layer 5-10.
        flambeau_blocks::apply_layer_output_scale_f16(
            &ops,
            session_stage.hidden,
            hidden,
            weights.layer_output_scale,
        )
        .context("layer_output_scale (TP)")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TpDecodeDriver impl
// ---------------------------------------------------------------------------

impl TpDecodeDriver for Gemma4TpDriver {
    fn cluster(&self) -> &HipCluster {
        self.model.tp.cluster()
    }

    fn n_layers(&self) -> usize {
        self.model.cfg.num_layers
    }

    fn head_rank(&self) -> usize {
        self.model.head_rank
    }

    fn embed_token(&mut self, rank: usize, token_id: u32) -> Result<()> {
        let device = self.model.tp.cluster().device(rank);
        let stream = device.default_stream();
        let model_stage = &self.model.stages[rank];
        let session_stage = &mut self.session.stages[rank];
        embed_token_host(
            device,
            stream,
            model_stage.token_embd.ptr,
            model_stage.token_embd.dtype,
            model_stage.token_embd.bytes,
            self.model.cfg.vocab_size,
            self.model.cfg.hidden_size,
            token_id,
            session_stage.hidden,
        )?;
        // Gemma4 input scale: `inpL = scale(inpL, sqrt(n_embd))`
        // (`gemma4-iswa.cpp:20`). Same step as single-device and PP;
        // without it every TP decode produces a degenerate fixed
        // token (caught originally on PP by the parity test).
        let reg = &self.model.regs[rank];
        let ops = HipOps::new(reg, stream);
        ops.scale_f16(
            session_stage.hidden,
            session_stage.hidden,
            self.model.cfg.hidden_size,
            (self.model.cfg.hidden_size as f32).sqrt(),
        )
        .context("TP embed_token sqrt(n_embd) scale")?;
        if std::env::var_os("FLAMBEAU_TP_DEBUG_EMBED").is_some() {
            use flambeau_core::CopyDirection;
            let hidden = self.model.cfg.hidden_size;
            let mut host = vec![half::f16::from_f32(0.0); hidden];
            // SAFETY: session_stage.hidden owns hidden*2 bytes.
            unsafe {
                device.memcpy_async(
                    stream,
                    CopyDirection::DeviceToHost,
                    DevicePtr(host.as_mut_ptr() as usize),
                    session_stage.hidden,
                    hidden * 2,
                )?;
            }
            stream.synchronize()?;
            let max_abs = host.iter().map(|h| h.to_f32().abs()).fold(0.0f32, f32::max);
            let nan_count = host.iter().filter(|h| h.to_f32().is_nan()).count();
            eprintln!(
                "  [TP_DEBUG_EMBED] rank={rank} token={token_id} | max_abs={max_abs:.4} nans={nan_count} first8={:?}",
                &host[..8].iter().map(|h| h.to_f32()).collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    fn forward_layer_decode(&mut self, il: usize, position: usize) -> Result<()> {
        forward_layer_decode_tp(self, il, position)
    }

    fn output_head(&mut self) -> Result<()> {
        let rank = self.model.head_rank;
        let device = self.model.tp.cluster().device(rank);
        let stream = device.default_stream();
        let reg = &self.model.regs[rank];
        let ops = HipOps::new(reg, stream);
        let cfg = &self.model.cfg;
        let model_stage = &self.model.stages[rank];
        let session_stage = &mut self.session.stages[rank];
        let scratch = session_stage
            .output_head_scratch
            .as_mut()
            .ok_or_else(|| anyhow!("output_head: head rank missing scratch"))?;
        let lm_head_tensor = model_stage
            .lm_head
            .as_ref()
            .unwrap_or(&model_stage.token_embd);
        let lm_head: WeightHandle = lm_head_tensor.as_weight_handle(model_stage.token_embd_dims)?;
        let logits = forward_output_head(
            &ops,
            session_stage.hidden,
            model_stage.output_norm.ptr,
            lm_head,
            cfg.final_logit_softcap,
            scratch,
            cfg.hidden_size,
            cfg.vocab_size,
            cfg.rms_norm_eps,
        )?;
        // SAFETY: logits points at vocab*4 device bytes; host buffer matches.
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

// ---------------------------------------------------------------------------
// Real-GGUF TP upload helpers
// ---------------------------------------------------------------------------

/// Upload + assemble one rank's stage from a GGUF file. Slices
/// column- and row-parallel weights per rank via
/// [`blocks::sharding::upload_sharded_tensor`]; replicates globals.
/// Every device allocation is recorded in a single
/// [`RawAllocTracker`], which is then passed to
/// [`Gemma4TpStage::from_pieces`] so scratch + weights share one
/// dispose chain.
#[allow(clippy::too_many_arguments)]
fn upload_one_tp_stage(
    file: &flambeau_quant::GgufFile,
    cfg: &Gemma4Config,
    layout: &ModelLayout,
    rank: usize,
    n_ranks: usize,
    device: &HipDevice,
    max_tokens: usize,
) -> Result<Gemma4TpStage> {
    let stream = device.default_stream();
    let mut tracker = RawAllocTracker::new();

    // Replicated globals via typed roles (`world=1` so token_embd /
    // output_norm / lm_head are uploaded as-is on each rank).
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
    let token_embd = uploaded_to_device_tensor(up.upload_required::<TokenEmbd>(0)?);
    let output_norm = uploaded_to_device_tensor(up.upload_required::<OutputNorm>(0)?);
    let lm_head = if cfg.tied_lm_head {
        None
    } else {
        up.upload::<LmHead>(0)?.map(uploaded_to_device_tensor)
    };
    drop(up);

    // Per-layer sharded weights — each upload pushes into `tracker`.
    let mut layer_weights = Vec::with_capacity(cfg.num_layers);
    for spec in &layout.layers {
        let lw = upload_layer_tp(file, spec, cfg, rank, n_ranks, device, stream, &mut tracker)
            .with_context(|| format!("layer {}", spec.index))?;
        layer_weights.push(lw);
    }

    Gemma4TpStage::from_pieces(
        device,
        rank,
        cfg,
        layout,
        n_ranks,
        layer_weights,
        token_embd,
        token_embd_dims,
        output_norm,
        lm_head,
        rank == 0,
        max_tokens,
        tracker,
    )
}

/// Upload sharded layer weights for one (rank, layer). Q/K/V/gate/up
/// are column-parallel (output-dim slice, contiguous mmap range);
/// attn_output and ffn_down are row-parallel (input-dim slice,
/// host-gathered). All slicing flows through
/// [`flambeau_runtime::tp_slice::slice_for_tp`] via
/// [`blocks::sharding::upload_sharded_tensor`]; per-head + per-layer
/// norms go through `upload_replicated_norm_f32_to_f16`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upload_layer_tp(
    file: &flambeau_quant::GgufFile,
    spec: &crate::layout::LayerSpec,
    cfg: &Gemma4Config,
    rank: usize,
    n_ranks: usize,
    device: &HipDevice,
    stream: &flambeau_backend_hip::HipStream,
    tracker: &mut RawAllocTracker,
) -> Result<Gemma4LayerWeights> {
    let il = spec.index;
    let mut up = WeightUploader {
        device,
        stream,
        tracker,
        file,
        cfg,
        world: n_ranks as u32,
        rank: rank as u32,
    };

    let attn_norm = up.upload_norm_required::<AttnNorm>(il)?;
    let attn_q = up.upload_matmul_required::<AttnQ>(il)?;
    let attn_k = up.upload_matmul::<AttnK>(il)?;
    let attn_v = up.upload_matmul::<AttnV>(il)?;
    let attn_output = up.upload_matmul_required::<AttnOutput>(il)?;
    let attn_q_norm = up.upload_norm_required::<AttnQNorm>(il)?;
    let attn_k_norm = up.upload_norm::<AttnKNorm>(il)?;
    let post_attention_norm = up.upload_norm_required::<PostAttnNorm>(il)?;
    let ffn_norm = up.upload_norm_required::<FfnNorm>(il)?;
    let ffn_gate = up.upload_matmul_required::<FfnGate>(il)?;
    let ffn_up = up.upload_matmul_required::<FfnUp>(il)?;
    let ffn_down = up.upload_matmul_required::<FfnDown>(il)?;
    let post_ffw_norm = up.upload_norm_required::<PostFfwNorm>(il)?;
    drop(up);

    // Gemma4 quirks: shared-KV layer invariants + host-side F32 scalar.
    if spec.has_kv {
        if attn_k.is_none() {
            bail!("layer {il}: attn_k required for has_kv layer");
        }
        if attn_k_norm.is_none() {
            bail!("layer {il}: attn_k_norm required for has_kv layer");
        }
    }
    let layer_output_scale = {
        let name = crate::names::AttnNames::for_layer(il).layer_output_scale;
        if let Some(info) = file.tensors.get(&name) {
            let raw = file.tensor_raw(&info.name)?;
            if raw.len() < 4 {
                bail!("layer {il}: layer_output_scale < 4 bytes");
            }
            let v = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
            Some(v)
        } else {
            None
        }
    };

    // MoE branch (26B-A4B) — TP-sharded routed-experts upload.
    let tp_moe = if spec.ffn_kind == FfnKind::Moe {
        let moe_dims = cfg
            .moe
            .ok_or_else(|| anyhow!("layer {il}: ffn_kind=Moe but cfg.moe is None"))?;
        Some(
            crate::tp_moe_upload::upload_moe_layer_tp(
                file,
                il,
                cfg.hidden_size,
                moe_dims,
                n_ranks as u32,
                rank as u32,
                device,
                stream,
                tracker,
            )
            .with_context(|| format!("upload_moe_layer_tp layer {il} rank {rank}"))?,
        )
    } else {
        None
    };

    // F32 copy of `post_attention_norm` for the F32 attention output
    // path. Uploaded for MoE full-attention layers (head_dim=512 path
    // that overflows F16 in the row-parallel output_proj F32→F16
    // cast). SWA layers stay F16 (head_dim=256 doesn't trigger the
    // saturation).
    let post_attention_norm_f32 = if cfg.moe.is_some() && !spec.is_swa {
        let name = crate::names::AttnNames::for_layer(il).post_attention_norm;
        let info = file
            .tensors
            .get(&name)
            .ok_or_else(|| anyhow!("{name} missing for F32 upload"))?;
        Some(
            flambeau_blocks::upload_replicated_tensor(file, info, device, stream, tracker)
                .with_context(|| format!("upload F32 post_attention_norm layer {il}"))?
                .ptr,
        )
    } else {
        None
    };

    Ok(Gemma4LayerWeights {
        attn_norm,
        attn_q,
        attn_k,
        attn_v,
        attn_output,
        attn_q_norm,
        attn_k_norm,
        post_attention_norm,
        post_attention_norm_f32,
        layer_output_scale,
        ffn_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
        post_ffw_norm,
        per_layer_embed: None,
        moe: None,
        tp_moe,
    })
}

/// Convert a [`UploadedTensor`] (from `blocks::sharding`) into a
/// model-crate [`DeviceTensor`]. The fields match by definition; this
/// is here so call sites that still take `DeviceTensor` (synthetic
/// tests, `from_pieces` API) stay unchanged.
pub(crate) fn uploaded_to_device_tensor(u: UploadedTensor) -> DeviceTensor {
    DeviceTensor {
        ptr: u.ptr,
        dtype: u.dtype,
        bytes: u.bytes,
    }
}

