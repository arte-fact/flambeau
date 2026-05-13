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
use flambeau_backend_hip::{BarP2pAllReduce, HipCluster, HipDevice, HipStream};
use flambeau_blocks::{
    forward_one_token_tp, Activation, DenseMlpDecodeScratch, DenseMlpTp,
    StandardAttention, StandardAttentionDecodeScratch, TpDecodeDriver, WeightHandle,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{HipOps, OpsRegistry};
use flambeau_ops::Ops;
use flambeau_quant::GgmlDType;
use flambeau_runtime::{F16Contig, KvCache};
use half::f16;

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
    raw_alloc_bytes: Vec<(DevicePtr, usize)>,
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
    pub cluster: Arc<HipCluster>,
    pub cfg: Gemma4Config,
    pub layout: ModelLayout,
    pub stages: Vec<Gemma4TpStage>,
    /// Rank that runs the LM head; head_rank == 0 in V1.
    pub head_rank: usize,
    regs: Vec<OpsRegistry>,
    ar: BarP2pAllReduce,
    logits_host: Vec<f32>,
}

fn alloc_zeroed(dev: &HipDevice, bytes: usize) -> Result<DevicePtr> {
    let p = dev.alloc(bytes).map_err(|e| anyhow!("alloc {bytes}: {e}"))?;
    let zero = vec![0u8; bytes];
    // SAFETY: dst has `bytes` allocation; src is a host vec of the same length.
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            p,
            DevicePtr(zero.as_ptr() as usize),
            bytes,
        )?;
    }
    dev.default_stream().synchronize()?;
    Ok(p)
}

fn upload_f16_ones(dev: &HipDevice, n: usize) -> Result<DevicePtr> {
    let ones: Vec<f16> = vec![f16::from_f32(1.0); n];
    let bytes = ones.len() * 2;
    let p = dev.alloc(bytes).map_err(|e| anyhow!("alloc {bytes}: {e}"))?;
    // SAFETY: see above.
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            p,
            DevicePtr(ones.as_ptr() as usize),
            bytes,
        )?;
    }
    dev.default_stream().synchronize()?;
    Ok(p)
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

        let mut raw_alloc_bytes: Vec<(DevicePtr, usize)> = Vec::new();
        macro_rules! ta {
            ($bytes:expr) => {{
                let bytes = $bytes;
                let p = alloc_zeroed(device, bytes)?;
                raw_alloc_bytes.push((p, bytes));
                (p, bytes)
            }};
        }

        // Sized for the widest layer.
        let mmvq_max = q_width_local_max
            .max(kv_width_local_max)
            .max(hidden)
            .max(ff_len_local);
        let q8_1_blocks = hidden.max(ff_len_local).div_ceil(32);
        let q8_1_bytes_per_block = 36;
        let x_q8_1_bytes = q8_1_blocks * q8_1_bytes_per_block;
        let activated_q8_1_bytes = ff_len_local.div_ceil(32) * q8_1_bytes_per_block;

        let v_ones_ptr = upload_f16_ones(device, head_dim)?;
        raw_alloc_bytes.push((v_ones_ptr, head_dim * 2));

        // Splitk partials sized for n_heads_local_max × MAX_SPLITK_CHUNKS.
        // F32 m/s buffers per (head, chunk); F32 o buffer per (head, chunk, head_dim).
        let splitk_ms_bytes = n_heads_local_max * flambeau_blocks::MAX_SPLITK_CHUNKS * 4;
        let splitk_o_bytes =
            n_heads_local_max * flambeau_blocks::MAX_SPLITK_CHUNKS * head_dim * 4;

        let scratch = TpScratchPtrs {
            x_q8_1: ta!(x_q8_1_bytes),
            mmvq_f32: ta!(mmvq_max * 4),
            q_f16: ta!(q_width_local_max * 2),
            k_f16: ta!(kv_width_local_max * 2),
            v_f16: ta!(kv_width_local_max * 2),
            attn_out_local: ta!(q_width_local_max.max(hidden) * 2),
            attn_residual_f16: ta!(hidden * 2),
            gate_f32: ta!(ff_len_local * 4),
            up_f32: ta!(ff_len_local * 4),
            activated_f16: ta!(ff_len_local * 2),
            activated_q8_1: ta!(activated_q8_1_bytes),
            positions: ta!(4),
            v_ones_f16: (v_ones_ptr, head_dim * 2),
            splitk_partials_m: ta!(splitk_ms_bytes),
            splitk_partials_s: ta!(splitk_ms_bytes),
            splitk_partials_o: ta!(splitk_o_bytes),
        };

        let hidden_ptr = ta!(hidden * 2).0;
        let partial_attn = ta!(hidden * 2).0;
        let partial_ffn = ta!(hidden * 2).0;

        let output_head_scratch = if is_head_rank {
            Some(OutputHeadScratch {
                x_norm_f16: ta!(hidden * 2).0,
                x_q8_1: ta!(x_q8_1_bytes).0,
                logits_f32: ta!(cfg.vocab_size * 4).0,
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
            raw_alloc_bytes,
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
        for (p, b) in self.raw_alloc_bytes.drain(..) {
            // SAFETY: every ptr came from device.alloc(bytes).
            unsafe {
                let _ = device.dealloc(p, b);
            }
        }
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

        let ar = BarP2pAllReduce::new(cluster.clone())
            .map_err(|e| anyhow!("BarP2pAllReduce: {e}"))?;
        let mut regs = Vec::with_capacity(n_ranks);
        for r in 0..n_ranks {
            let dev = cluster.device(r);
            dev.bind()?;
            regs.push(OpsRegistry::new(dev).map_err(|e| anyhow!("registry rank {r}: {e}"))?);
        }
        let logits_host = vec![0.0f32; cfg.vocab_size];
        Ok(Self {
            cluster,
            cfg,
            layout,
            stages,
            head_rank,
            regs,
            ar,
            logits_host,
        })
    }

    pub fn dispose(&mut self) -> Result<()> {
        for (r, stage) in self.stages.iter_mut().enumerate() {
            let dev = self.cluster.device(r);
            stage.dispose(dev)?;
        }
        Ok(())
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

fn embed_decode_host(
    device: &HipDevice,
    stream: &HipStream,
    tok_embd_ptr: DevicePtr,
    tok_embd_bytes: usize,
    tok_embd_dtype: GgmlDType,
    vocab: usize,
    hidden: usize,
    token_id: u32,
    out_f16_dev: DevicePtr,
) -> Result<()> {
    if (token_id as usize) >= vocab {
        bail!("token_id {token_id} >= vocab {vocab}");
    }
    let bs = tok_embd_dtype.block_size() as usize;
    let ts = tok_embd_dtype.type_size() as usize;
    if hidden % bs != 0 {
        bail!("hidden {hidden} % block_size {bs} != 0 for {tok_embd_dtype:?}");
    }
    let row_bytes = (hidden / bs) * ts;
    let offset = token_id as usize * row_bytes;
    if offset + row_bytes > tok_embd_bytes {
        bail!("tok_embd row OOB at token {token_id}");
    }
    let src = tok_embd_ptr.offset_bytes(offset);

    let mut row_raw = vec![0u8; row_bytes];
    // SAFETY: src points at ≥ row_bytes valid device bytes.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(row_raw.as_mut_ptr() as usize),
            src,
            row_bytes,
        )?;
    }
    stream.synchronize()?;
    let row_f16: Vec<f16> = if tok_embd_dtype == GgmlDType::F16 {
        bytemuck::cast_slice::<u8, f16>(&row_raw).to_vec()
    } else {
        let f32 = flambeau_quant::dequantize_to_vec(tok_embd_dtype, &row_raw, hidden)
            .map_err(|e| anyhow!("dequant tok_embd row {token_id}: {e}"))?;
        f32.into_iter().map(f16::from_f32).collect()
    };
    let upload_bytes = hidden * 2;
    // SAFETY: out_f16_dev has hidden*2 valid bytes.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            out_f16_dev,
            DevicePtr(row_f16.as_ptr() as usize),
            upload_bytes,
        )?;
    }
    stream.synchronize()?;
    Ok(())
}

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
    let q_width_local = n_heads_local * head_dim;
    let kv_width_local = n_kv_local * head_dim;
    let ff_len = cfg.feed_forward_length;
    let ff_len_local = ff_len / n_ranks;
    let window: i32 = spec.window as i32;
    let softmax_scale: f32 = 1.0;
    let rms_eps = cfg.rms_norm_eps;

    // Phase 1: per-rank → partial_attn (row-parallel output proj).
    // Delegates the full kernel sequence to
    // `flambeau_blocks::StandardAttention::forward_decode` with
    // per-rank sliced shapes, alt-V via `attn_v: None`, V-norm via the
    // unit-weight buffer, and softmax_scale=1.0 + window_size for SWA.
    // Splitk dispatches automatically at n_tokens_kv > 256.
    for r in 0..n_ranks {
        let dev = driver.cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &driver.regs[r];
        let ops = HipOps::new(reg, stream);
        let stage = &mut driver.stages[r];
        let weights = &stage.layer_weights[il];
        let x_in = stage.hidden;
        let attn_k = weights
            .attn_k
            .as_ref()
            .ok_or_else(|| anyhow!("layer {il}: attn_k missing"))?;
        let attn_k_norm_w = weights
            .attn_k_norm
            .ok_or_else(|| anyhow!("layer {il}: attn_k_norm missing"))?;

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
        .context("StandardAttention::new (gemma4 TP)")?
        .with_softmax_scale(softmax_scale)
        .with_v_norm_w(stage.scratch.v_ones_f16.0);
        let block = if window > 0 {
            block.with_window_size(window as u32)
        } else {
            block
        };

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

    // Phase 2: AR-sum partial_attn across ranks. After this every rank's
    // `partial_attn` holds the full hidden-dim attn-out.
    let partials: [DevicePtr; 2] = [driver.stages[0].partial_attn, driver.stages[1].partial_attn];
    let streams = [
        driver.cluster.device(0).default_stream(),
        driver.cluster.device(1).default_stream(),
    ];
    // SAFETY: partial_attn is hidden F16 elems per rank.
    unsafe {
        driver
            .ar
            .sum_tp2(&partials, hidden as u32, &streams)
            .map_err(|e| anyhow!("AR sum_tp2 attn: {e}"))?;
    }

    // Phase 3: per-rank post_attention_norm + residual add → attn_residual.
    for r in 0..n_ranks {
        let dev = driver.cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &driver.regs[r];
        let ops = HipOps::new(reg, stream);
        let stage = &mut driver.stages[r];
        let weights = &stage.layer_weights[il];
        let scratch = &mut stage.scratch;
        ops.rmsnorm_f16(
            stage.partial_attn,
            weights.post_attention_norm,
            scratch.attn_out_local.0,
            1,
            hidden,
            rms_eps,
        )
        .context("post_attention_norm (TP)")?;
        ops.add_f16(
            stage.hidden,
            scratch.attn_out_local.0,
            scratch.attn_residual_f16.0,
            hidden,
        )
        .context("attn residual add (TP)")?;
    }

    // Phase 4: per-rank ffn_norm + gate/up + GELU + down (row-parallel) → partial_ffn.
    // Delegates the FFN tail (gate/up/activation/down/cast) to
    // `DenseMlpTp` so it shares the kernel sequence with qwen3-moe's
    // TP path. The ffn_norm + Q8_1 quantise stays inline here because
    // the block API only owns gate/up/down.
    for r in 0..n_ranks {
        let dev = driver.cluster.device(r);
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

    // Phase 5: AR-sum partial_ffn.
    let partials_ffn: [DevicePtr; 2] = [driver.stages[0].partial_ffn, driver.stages[1].partial_ffn];
    let streams = [
        driver.cluster.device(0).default_stream(),
        driver.cluster.device(1).default_stream(),
    ];
    // SAFETY: same as Phase 2.
    unsafe {
        driver
            .ar
            .sum_tp2(&partials_ffn, hidden as u32, &streams)
            .map_err(|e| anyhow!("AR sum_tp2 ffn: {e}"))?;
    }

    // Phase 6: post_ffw_norm + residual add (with attn_residual) → next-layer hidden.
    for r in 0..n_ranks {
        let dev = driver.cluster.device(r);
        dev.bind()?;
        let stream = dev.default_stream();
        let reg = &driver.regs[r];
        let ops = HipOps::new(reg, stream);
        let stage = &mut driver.stages[r];
        let weights = &stage.layer_weights[il];
        let scratch = &mut stage.scratch;
        ops.rmsnorm_f16(
            stage.partial_ffn,
            weights.post_ffw_norm,
            scratch.attn_out_local.0,
            1,
            hidden,
            rms_eps,
        )
        .context("post_ffw_norm (TP)")?;
        ops.add_f16(
            scratch.attn_residual_f16.0,
            scratch.attn_out_local.0,
            stage.hidden,
            hidden,
        )
        .context("ffn residual add (TP)")?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// TpDecodeDriver impl
// ---------------------------------------------------------------------------

impl TpDecodeDriver for Gemma4TpDriver {
    fn cluster(&self) -> &HipCluster {
        &self.cluster
    }

    fn n_layers(&self) -> usize {
        self.cfg.num_layers
    }

    fn head_rank(&self) -> usize {
        self.head_rank
    }

    fn embed_token(&mut self, rank: usize, token_id: u32) -> Result<()> {
        let device = self.cluster.device(rank);
        let stream = device.default_stream();
        let stage = &mut self.stages[rank];
        embed_decode_host(
            device,
            stream,
            stage.token_embd.ptr,
            stage.token_embd.bytes,
            stage.token_embd.dtype,
            self.cfg.vocab_size,
            self.cfg.hidden_size,
            token_id,
            stage.hidden,
        )
    }

    fn forward_layer_decode(&mut self, il: usize, position: usize) -> Result<()> {
        forward_layer_decode_tp(self, il, position)
    }

    fn output_head(&mut self) -> Result<()> {
        let rank = self.head_rank;
        let device = self.cluster.device(rank);
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
