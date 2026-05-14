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
    embed_token_host, forward_one_token_tp, post_norm_residual_f16, tp_allreduce_sum,
    upload_f16_ones, AttnK, AttnKNorm, AttnNorm, AttnOutput, AttnQ, AttnQNorm, AttnV, Buffer,
    FfnDown, FfnGate, FfnNorm, FfnUp, LmHead, OutputNorm, PostAttnNorm, PostFfwNorm, RowParallel,
    TokenEmbd, Activation, DenseMlpDecodeScratch, DenseMlpTp, RawAllocTracker, F16,
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

/// Per-rank state for a TP stage. `layer_weights` carries the
/// **sharded** weights for every layer (this rank's slice). KV caches
/// are sized for the rank's local KV-head count.
pub struct Gemma4TpStage {
    pub rank: usize,
    /// Per-layer sharded weights; ALL ranks carry weights for ALL layers.
    pub layer_weights: Vec<Gemma4LayerWeights>,
    /// Per-layer KV cache (each holds this rank's local KV head shard).
    pub kv_caches: Vec<Option<KvCache<F16Contig, HipDevice>>>,
    /// Replicated token_embd (each rank holds a full copy).
    pub token_embd: DeviceTensor,
    pub token_embd_dims: [usize; 2],
    /// Replicated output_norm.
    pub output_norm: DeviceTensor,
    /// Replicated LM head (gemma4 ties to `token_embd`).
    pub lm_head: Option<DeviceTensor>,
    /// F16 [hidden] hidden buffer holding the current residual stream.
    pub hidden: DevicePtr,
    /// F16 [hidden] buffer holding this rank's partial attn-out
    /// contribution after the row-parallel output proj.
    pub partial_attn: DevicePtr,
    /// F16 [hidden] buffer holding this rank's partial FFN-out
    /// contribution after the row-parallel down proj.
    pub partial_ffn: DevicePtr,
    /// Layer scratch — sized to the local head shard.
    scratch: TpScratchPtrs,
    /// Optional output-head scratch (head rank only).
    pub output_head_scratch: Option<OutputHeadScratch>,
    positions_host: Vec<i32>,
    raw_alloc: RawAllocTracker,
    disposed: bool,
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

/// TP driver. Single-token decode across N ranks with column / row
/// parallel attention + FFN and BAR1 P2P AllReduce after the two
/// row-parallel projections (attn_output + ffn_down).
pub struct Gemma4TpDriver {
    pub tp: TpCluster,
    pub cfg: Gemma4Config,
    pub layout: ModelLayout,
    pub stages: Vec<Gemma4TpStage>,
    /// Rank that runs the LM head; head_rank == 0 in V1.
    pub head_rank: usize,
    regs: Vec<OpsRegistry>,
    logits_host: Vec<f32>,
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
    /// upload the weight tensors — scratch + v_ones allocations are
    /// appended to the same tracker so dispose frees everything in
    /// one pass. Pass `RawAllocTracker::new()` when no weights are
    /// pre-tracked (synthetic-weight tests).
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
        device.bind()?;
        if layer_weights.len() != cfg.num_layers {
            bail!(
                "Gemma4TpStage: expected {} layer weights, got {}",
                cfg.num_layers,
                layer_weights.len()
            );
        }

        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim.max(cfg.head_dim_swa);
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
                    "Gemma4TpStage: shared-KV tail layer {} not supported in S9-A",
                    spec.index
                );
            }
            if spec.ffn_kind != FfnKind::Dense {
                bail!(
                    "Gemma4TpStage: MoE FFN layer {} not supported in S9-A",
                    spec.index
                );
            }
            let n_kv_local = spec.n_kv_heads / n_ranks;
            let kv =
                KvCache::<F16Contig, HipDevice>::new(device, n_kv_local, spec.head_dim, max_tokens)
                    .map_err(|e| anyhow!("kv alloc layer {} rank {rank}: {e}", spec.index))?;
            kv_caches.push(Some(kv));
        }

        let mut raw_alloc = weight_alloc;

        // Sized for the widest layer.
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

        let output_head_scratch = if is_head_rank {
            Some(OutputHeadScratch {
                x_norm_f16: raw_alloc.alloc_f16(device, hidden)?.0,
                x_q8_1: raw_alloc.alloc_q8_1(device, q8_1_n)?.0,
                logits_f32: raw_alloc.alloc_f32(device, cfg.vocab_size)?.0,
            })
        } else {
            None
        };

        Ok(Self {
            rank,
            layer_weights,
            kv_caches,
            token_embd,
            token_embd_dims,
            output_norm,
            lm_head,
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

    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
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

impl Drop for Gemma4TpStage {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                "Gemma4TpStage rank {} dropped without dispose(); resources leaked",
                self.rank
            );
        }
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
        Ok(Self {
            tp,
            cfg,
            layout,
            stages,
            head_rank,
            regs,
            logits_host,
        })
    }

    pub fn dispose(&mut self) -> Result<()> {
        for (r, stage) in self.stages.iter_mut().enumerate() {
            let dev = self.tp.cluster().device(r);
            stage.dispose(dev)?;
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
        if cfg.moe.is_some() {
            bail!("Gemma4TpDriver::upload: MoE TP path is followup work");
        }
        if cfg.per_layer_embed.is_some() {
            bail!("Gemma4TpDriver::upload: per-layer-embd TP path is followup work");
        }
        for spec in &layout.layers {
            if !spec.has_kv {
                bail!(
                    "Gemma4TpDriver::upload: shared-KV tail layer {} unsupported (S9-B)",
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
        for (i, &v) in self.logits_host.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i as u32;
            }
        }
        Ok(best_i)
    }
}

impl Drop for Gemma4TpDriver {
    fn drop(&mut self) {
        if self.stages.iter().any(|s| !s.disposed) {
            tracing::warn!("Gemma4TpDriver dropped without dispose()");
        }
    }
}

// ---------------------------------------------------------------------------
// Per-layer TP composition
// ---------------------------------------------------------------------------

/// One layer's TP decode. Phases:
/// 1. Per-rank: attn-norm + Q/K/V proj + per-head norms + RoPE + KV
///    append + local attention + row-parallel output proj (partial).
/// 2. AR-sum partial_attn across ranks.
/// 3. Per-rank: post_attention_norm + residual add → attn_residual.
/// 4. Per-rank: ffn_norm + gate/up + GELU + row-parallel down (partial).
/// 5. AR-sum partial_ffn across ranks.
/// 6. Per-rank: post_ffw_norm + residual add → hidden (next layer input).
#[allow(clippy::too_many_arguments)]
fn forward_layer_decode_tp(
    driver: &mut Gemma4TpDriver,
    il: usize,
    position: usize,
) -> Result<()> {
    let cfg = driver.cfg.clone();
    let spec = driver.layout.layers[il];
    let n_ranks = driver.stages.len();
    if n_ranks != 2 {
        // S9-A: only TP2 supported. TP4 is just adding AR-residual_tp4
        // hookups; left for S9-B.
        bail!("forward_layer_decode_tp: TP{n_ranks} not supported in S9-A; only TP2");
    }

    let hidden = cfg.hidden_size;
    let head_dim = spec.head_dim;
    let n_heads_local = spec.n_heads / n_ranks;
    let n_kv_local = spec.n_kv_heads / n_ranks;
    let ff_len = cfg.feed_forward_length;
    let ff_len_local = ff_len / n_ranks;
    let rms_eps = cfg.rms_norm_eps;

    // Phase 1: per-rank → partial_attn (row-parallel output proj).
    // Delegates the full kernel sequence to
    // `flambeau_blocks::StandardAttention::forward_decode` with
    // per-rank sliced shapes, alt-V via `attn_v: None`, V-norm via the
    // unit-weight buffer, and softmax_scale=1.0 + window_size for SWA.
    // Splitk dispatches automatically at n_tokens_kv > 256.
    for r in 0..n_ranks {
        let dev = driver.tp.cluster().device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &driver.regs[r];
        let ops = HipOps::new(reg, stream);
        let stage = &mut driver.stages[r];
        let weights = &stage.layer_weights[il];
        let x_in = stage.hidden;
        let block = weights.build_attn_block(
            &spec,
            hidden,
            n_heads_local,
            n_kv_local,
            head_dim,
            rms_eps,
            stage.scratch.v_ones_f16.0,
        )?;

        let kv = stage.kv_caches[il]
            .as_mut()
            .expect("S9-A requires per-layer KV");
        let mut std_scratch = StandardAttentionDecodeScratch {
            x_q8_1: stage.scratch.x_q8_1.0,
            mmvq_f32: stage.scratch.mmvq_f32.0,
            q_fused_f16: DevicePtr(0),
            q_f16: stage.scratch.q_f16.0,
            gate_f16: DevicePtr(0),
            k_f16: stage.scratch.k_f16.0,
            v_f16: stage.scratch.v_f16.0,
            k_q8_0: DevicePtr(0),
            v_q8_0: DevicePtr(0),
            attn_out_f16: stage.scratch.attn_out_local.0,
            gated_out_f16: DevicePtr(0),
            positions: stage.scratch.positions.0,
            positions_host: &mut stage.positions_host,
            splitk_partials_m: stage.scratch.splitk_partials_m.0,
            splitk_partials_s: stage.scratch.splitk_partials_s.0,
            splitk_partials_o: stage.scratch.splitk_partials_o.0,
        };
        block.forward_decode(
            &ops,
            dev,
            stream,
            x_in,
            stage.partial_attn,
            kv,
            &mut std_scratch,
            position,
            /* slots = */ None,
        )
        .context("StandardAttention::forward_decode (gemma4 TP)")?;
    }

    if std::env::var_os("FLAMBEAU_TP_DEBUG_PHASES").is_some() && il < 2 {
        for r in 0..n_ranks {
            tp_dump_buffer(driver, r, driver.stages[r].partial_attn, hidden,
                &format!("L{il} P1 partial_attn rank{r}"));
        }
    }
    // Phase 2: AR-sum partial_attn across ranks via the typed
    // transition. After this every rank's `partial_attn` buffer holds
    // the full hidden-dim attn-out — typestate-tagged `Replicated`.
    //
    // Field-type migration to `Buffer<F16, RowParallel<0>>` storage is
    // deferred (Phase 8d) since it cascades through ~12 read sites
    // each for `hidden` / `partial_attn` / `partial_ffn`. The typed
    // wrapper at the AR boundary is the proof-of-concept (Phase 8b)
    // — it demonstrates the compile-time transition flow without
    // forcing the larger field-type churn.
    {
        let partials: [Buffer<F16, RowParallel<0>>; 2] = [
            Buffer::from_raw_unchecked(driver.stages[0].partial_attn, hidden),
            Buffer::from_raw_unchecked(driver.stages[1].partial_attn, hidden),
        ];
        let streams: [&_; 2] = [
            driver.tp.cluster().device(0).default_stream(),
            driver.tp.cluster().device(1).default_stream(),
        ];
        // SAFETY: partial_attn is hidden F16 elems per rank; streams outlive
        // this call; subsequent Phase-3 reads on each rank are serialised on
        // that rank's stream.
        let _replicated = unsafe {
            tp_allreduce_sum::<0>(driver.tp.ar(), &partials, &streams)
        }
        .context("AR sum partial_attn (typed)")?;
        // `_replicated` is `Vec<Buffer<F16, Replicated>>` tagging the
        // same allocations as Replicated. Phase-3 reads consume them
        // via the raw `stage.partial_attn` pointer (the typed flow
        // ends at the AR transition for this proof-of-concept).
    }

    if std::env::var_os("FLAMBEAU_TP_DEBUG_PHASES").is_some() && il < 2 {
        tp_dump_buffer(driver, 0, driver.stages[0].partial_attn, hidden,
            &format!("L{il} P2 partial_attn rank0 (post-AR)"));
    }
    // Phase 3: per-rank post_attention_norm + residual add → attn_residual.
    for r in 0..n_ranks {
        let dev = driver.tp.cluster().device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &driver.regs[r];
        let ops = HipOps::new(reg, stream);
        let stage = &mut driver.stages[r];
        let weights = &stage.layer_weights[il];
        let scratch = &mut stage.scratch;
        post_norm_residual_f16(
            &ops,
            stage.partial_attn,
            weights.post_attention_norm,
            scratch.attn_out_local.0,
            stage.hidden,
            scratch.attn_residual_f16.0,
            1,
            hidden,
            rms_eps,
        )
        .context("post_attention_norm + residual (TP)")?;
    }

    if std::env::var_os("FLAMBEAU_TP_DEBUG_PHASES").is_some() && il < 2 {
        tp_dump_buffer(driver, 0, driver.stages[0].scratch.attn_residual_f16.0, hidden,
            &format!("L{il} P3 attn_residual rank0"));
    }
    // Phase 4: per-rank ffn_norm + gate/up + GELU + down (row-parallel) → partial_ffn.
    // Delegates the FFN tail (gate/up/activation/down/cast) to
    // `DenseMlpTp` so it shares the kernel sequence with qwen3-moe's
    // TP path. The ffn_norm + Q8_1 quantise stays inline here because
    // the block API only owns gate/up/down.
    for r in 0..n_ranks {
        let dev = driver.tp.cluster().device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &driver.regs[r];
        let ops = HipOps::new(reg, stream);
        let stage = &mut driver.stages[r];
        let weights = &stage.layer_weights[il];
        let scratch = &mut stage.scratch;
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
            stage.partial_ffn,
            block_scratch,
            /* pre_quantized = */ true,
        )
        .context("DenseMlpTp::forward_decode (gemma4 TP)")?;
    }

    if std::env::var_os("FLAMBEAU_TP_DEBUG_PHASES").is_some() && il < 2 {
        for r in 0..n_ranks {
            tp_dump_buffer(driver, r, driver.stages[r].partial_ffn, hidden,
                &format!("L{il} P4 partial_ffn rank{r}"));
        }
    }
    // Phase 5: AR-sum partial_ffn via the typed transition (same
    // pattern as Phase 2 above).
    {
        let partials_ffn: [Buffer<F16, RowParallel<0>>; 2] = [
            Buffer::from_raw_unchecked(driver.stages[0].partial_ffn, hidden),
            Buffer::from_raw_unchecked(driver.stages[1].partial_ffn, hidden),
        ];
        let streams: [&_; 2] = [
            driver.tp.cluster().device(0).default_stream(),
            driver.tp.cluster().device(1).default_stream(),
        ];
        // SAFETY: same as Phase 2.
        let _replicated = unsafe {
            tp_allreduce_sum::<0>(driver.tp.ar(), &partials_ffn, &streams)
        }
        .context("AR sum partial_ffn (typed)")?;
    }

    // Phase 6: post_ffw_norm + residual add (with attn_residual) → next-layer hidden.
    for r in 0..n_ranks {
        let dev = driver.tp.cluster().device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &driver.regs[r];
        let ops = HipOps::new(reg, stream);
        let stage = &mut driver.stages[r];
        let weights = &stage.layer_weights[il];
        let scratch = &mut stage.scratch;
        post_norm_residual_f16(
            &ops,
            stage.partial_ffn,
            weights.post_ffw_norm,
            scratch.attn_out_local.0,
            scratch.attn_residual_f16.0,
            stage.hidden,
            1,
            hidden,
            rms_eps,
        )
        .context("post_ffw_norm + residual (TP)")?;
        // Per-layer scalar `layer_output_scale` (mirrors
        // `layer.rs::forward_layer_decode` step 14). Gemma4 31B uses
        // this to keep the residual stream's magnitude bounded across
        // 60 layers; without it values explode → Inf → NaN around
        // layer 5-10 (caught by tp_hidden_cross_rank_match diagnostic).
        if let Some(scale_v) = weights.layer_output_scale {
            if scale_v != 1.0 {
                ops.scale_f16(stage.hidden, stage.hidden, hidden, scale_v)
                    .context("layer_output_scale (TP)")?;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// TpDecodeDriver impl
// ---------------------------------------------------------------------------

impl TpDecodeDriver for Gemma4TpDriver {
    fn cluster(&self) -> &HipCluster {
        self.tp.cluster()
    }

    fn n_layers(&self) -> usize {
        self.cfg.num_layers
    }

    fn head_rank(&self) -> usize {
        self.head_rank
    }

    fn embed_token(&mut self, rank: usize, token_id: u32) -> Result<()> {
        let device = self.tp.cluster().device(rank);
        let stream = device.default_stream();
        let stage = &mut self.stages[rank];
        embed_token_host(
            device,
            stream,
            stage.token_embd.ptr,
            stage.token_embd.dtype,
            stage.token_embd.bytes,
            self.cfg.vocab_size,
            self.cfg.hidden_size,
            token_id,
            stage.hidden,
        )?;
        // Gemma4 input scale: `inpL = scale(inpL, sqrt(n_embd))`
        // (`gemma4-iswa.cpp:20`). Same step as single-device and PP;
        // without it every TP decode produces a degenerate fixed
        // token (caught originally on PP by the parity test).
        let reg = &self.regs[rank];
        let ops = HipOps::new(reg, stream);
        ops.scale_f16(
            stage.hidden,
            stage.hidden,
            self.cfg.hidden_size,
            (self.cfg.hidden_size as f32).sqrt(),
        )
        .context("TP embed_token sqrt(n_embd) scale")?;
        if std::env::var_os("FLAMBEAU_TP_DEBUG_EMBED").is_some() {
            use flambeau_core::CopyDirection;
            let hidden = self.cfg.hidden_size;
            let mut host = vec![half::f16::from_f32(0.0); hidden];
            // SAFETY: stage.hidden owns hidden*2 bytes.
            unsafe {
                device.memcpy_async(
                    stream,
                    CopyDirection::DeviceToHost,
                    DevicePtr(host.as_mut_ptr() as usize),
                    stage.hidden,
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
        let rank = self.head_rank;
        let device = self.tp.cluster().device(rank);
        let stream = device.default_stream();
        let reg = &self.regs[rank];
        let ops = HipOps::new(reg, stream);
        let cfg = &self.cfg;
        let stage = &mut self.stages[rank];
        let scratch = stage
            .output_head_scratch
            .as_mut()
            .ok_or_else(|| anyhow!("output_head: head rank missing scratch"))?;
        let lm_head_tensor = stage
            .lm_head
            .as_ref()
            .unwrap_or(&stage.token_embd);
        let lm_head: WeightHandle = lm_head_tensor.as_weight_handle(stage.token_embd_dims)?;
        let logits = forward_output_head(
            &ops,
            stage.hidden,
            stage.output_norm.ptr,
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
                DevicePtr(self.logits_host.as_mut_ptr() as usize),
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
fn upload_layer_tp(
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
            Some(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
        } else {
            None
        }
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
        layer_output_scale,
        ffn_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
        post_ffw_norm,
        per_layer_embed: None,
        moe: None,
    })
}

/// Convert a [`UploadedTensor`] (from `blocks::sharding`) into a
/// model-crate [`DeviceTensor`]. The fields match by definition; this
/// is here so call sites that still take `DeviceTensor` (synthetic
/// tests, `from_pieces` API) stay unchanged.
fn uploaded_to_device_tensor(u: UploadedTensor) -> DeviceTensor {
    DeviceTensor {
        ptr: u.ptr,
        dtype: u.dtype,
        bytes: u.bytes,
    }
}

fn tp_dump_buffer(driver: &Gemma4TpDriver, rank: usize, ptr: DevicePtr, n: usize, label: &str) {
    use flambeau_core::CopyDirection;
    let device = driver.tp.cluster().device(rank);
    let _ = device.bind();
    let mut host = vec![half::f16::from_f32(0.0); n];
    // SAFETY: caller guarantees ptr owns n*2 bytes.
    unsafe {
        let _ = device.memcpy_async(
            device.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            ptr,
            n * 2,
        );
    }
    let _ = device.default_stream().synchronize();
    let max_abs = host.iter().map(|h| h.to_f32().abs()).fold(0.0f32, f32::max);
    let nans = host.iter().filter(|h| h.to_f32().is_nan()).count();
    let infs = host.iter().filter(|h| h.to_f32().is_infinite()).count();
    let zeros = host.iter().filter(|h| h.to_f32() == 0.0).count();
    eprintln!(
        "  [TP_PHASE] {label} | max_abs={max_abs:.4} nans={nans} infs={infs} zeros={zeros}/{n} first4={:?}",
        &host[..4].iter().map(|h| h.to_f32()).collect::<Vec<_>>()
    );
}

