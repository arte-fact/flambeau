//! Pipeline-parallel decode driver for Gemma 4.
//!
//! Each rank owns:
//! - A subset of layers (assigned by [`partition_layers`]).
//! - The KV caches for those layers (shared-KV tail layers reference
//!   another layer in the same rank).
//! - Per-call layer scratch + hidden ping-pong (`hidden_a`/`hidden_b`).
//! - Rank 0 also owns `token_embd`.
//! - Last rank also owns `output_norm` (+ optional `output`) and the
//!   output-head scratch.
//!
//! Stage-boundary invariant: a tail layer (`has_kv == false`) MUST be
//! on the same rank as its `kv_share_src`. The partition function
//! enforces this — splits that would violate it return an error.
//!
//! Limitations of S8-A:
//! - No per-layer side-channel embedding (S5-B-2).

#![cfg(feature = "hip")]

use anyhow::{anyhow, bail, Context, Result};
use flambeau_backend_hip::{HipCluster, HipDevice, HipStream};
use flambeau_blocks::{
    embed_token_host, forward_one_token_pp, forward_prefill_pp, row_bytes_for_dtype,
    upload_f16_ones, AttnK, AttnKNorm, AttnNorm, AttnOutput, AttnQ, AttnQNorm, AttnV, FfnDown,
    FfnGate, FfnNorm, FfnUp, PostAttnNorm, PostFfwNorm, PpDecodeDriver, PpPrefillDriver,
    RawAllocTracker, StageCommon, WeightUploader,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{HipOps, OpsRegistry};
use flambeau_quant::GgmlDType;
use flambeau_runtime::{F16Contig, KvCache};
use half::f16;

use flambeau_quant::{GgufFile, TensorInfo};

use crate::config::Gemma4Config;
use crate::layer::{forward_layer_decode, forward_layer_prefill, Gemma4LayerWeights};
use crate::layout::{FfnKind, LayerSpec, ModelLayout};
use crate::names::{AttnNames, GlobalNames};
use crate::output_head::{forward_output_head, OutputHeadScratch};
use crate::scratch::{LayerDecodeScratch, LayerPrefillScratch};
use crate::softcap::apply_logit_softcap;
use crate::weights_hip::DeviceTensor;

/// Even-split layer-to-rank assignment with a stage-boundary validity
/// check. Returns one rank index per layer (`out.len() == cfg.num_layers`).
/// Errors when a shared-KV tail layer would land on a different rank
/// than its `kv_share_src`.
pub fn partition_layers(n_ranks: usize, layout: &ModelLayout) -> Result<Vec<usize>> {
    if n_ranks == 0 {
        bail!("partition_layers: n_ranks=0");
    }
    let n = layout.layers.len();
    if n == 0 {
        bail!("partition_layers: layout has 0 layers");
    }
    let per_rank = n.div_ceil(n_ranks);
    let mut layer_to_rank = vec![0usize; n];
    for (i, spec) in layout.layers.iter().enumerate() {
        layer_to_rank[i] = (spec.index / per_rank).min(n_ranks - 1);
    }

    // Stage-boundary check: tail layer's source must be on same rank.
    for spec in &layout.layers {
        if !spec.has_kv {
            let src = spec.kv_share_src.ok_or_else(|| {
                anyhow!(
                    "partition_layers: tail layer {} has no kv_share_src \
                     (run `ModelLayout::resolve_kv_sharing` first)",
                    spec.index
                )
            })?;
            if layer_to_rank[spec.index] != layer_to_rank[src] {
                bail!(
                    "partition_layers: shared-KV tail layer {} on rank {} but \
                     its kv_share_src layer {} is on rank {}; split would break \
                     the tail's KV reference. Adjust per_rank or merge stages.",
                    spec.index,
                    layer_to_rank[spec.index],
                    src,
                    layer_to_rank[src]
                );
            }
        }
    }
    Ok(layer_to_rank)
}

struct LayerScratchPtrs {
    x_q8_1: (DevicePtr, usize),
    mmvq_f32: (DevicePtr, usize),
    q_f16: (DevicePtr, usize),
    k_f16: (DevicePtr, usize),
    v_f16: (DevicePtr, usize),
    attn_out_f16: (DevicePtr, usize),
    post_attn_norm_f16: (DevicePtr, usize),
    attn_residual_f16: (DevicePtr, usize),
    ffn_norm_f16: (DevicePtr, usize),
    gate_f32: (DevicePtr, usize),
    up_f32: (DevicePtr, usize),
    activated_f16: (DevicePtr, usize),
    activated_q8_1: (DevicePtr, usize),
    down_f32: (DevicePtr, usize),
    post_ffw_norm_f16: (DevicePtr, usize),
    positions: (DevicePtr, usize),
    v_ones_f16: (DevicePtr, usize),
    splitk_partials_m: (DevicePtr, usize),
    splitk_partials_s: (DevicePtr, usize),
    splitk_partials_o: (DevicePtr, usize),
}

/// Per-stage prefill scratch, sized for `max_tokens` rows.
struct PrefillScratchPtrs {
    x_norm_f16: (DevicePtr, usize),
    x_q8_1: (DevicePtr, usize),
    x_q8_1_mmq: (DevicePtr, usize),
    mmvq_f32: (DevicePtr, usize),
    q_f16: (DevicePtr, usize),
    k_f16: (DevicePtr, usize),
    v_f16: (DevicePtr, usize),
    attn_out_f16: (DevicePtr, usize),
    post_attn_norm_f16: (DevicePtr, usize),
    attn_residual_f16: (DevicePtr, usize),
    gate_f32: (DevicePtr, usize),
    up_f32: (DevicePtr, usize),
    activated_f16: (DevicePtr, usize),
    activated_q8_1: (DevicePtr, usize),
    activated_q8_1_mmq: (DevicePtr, usize),
    down_f32: (DevicePtr, usize),
    post_ffw_norm_f16: (DevicePtr, usize),
    positions: (DevicePtr, usize),
    v_ones_f16: (DevicePtr, usize),
    gated_q8_1: (DevicePtr, usize),
    gated_q8_1_mmq: (DevicePtr, usize),
    positions_host: Vec<i32>,
}

/// Per-rank model state (weights only — Arc-shareable across multiple
/// concurrent `Gemma4PpSession` slots).
pub struct Gemma4PpModelStage {
    /// Shared bookkeeping for **weight** allocations on this rank.
    /// `common.raw_alloc` holds every device alloc that backs a
    /// `Gemma4LayerWeights` / `token_embd` / `output_norm` / `output`
    /// tensor; `dispose(device)` frees them all in one pass.
    pub common: StageCommon,
    /// Indices into [`ModelLayout::layers`] for this rank's layers,
    /// in ascending order.
    pub global_layer_indices: Vec<usize>,
    /// One [`Gemma4LayerWeights`] per `global_layer_indices` entry.
    pub layer_weights: Vec<Gemma4LayerWeights>,
    /// Maps each local layer index to the LOCAL index of its
    /// `kv_share_src` (or itself when `has_kv == true`). A model-side
    /// invariant of the partition; sessions allocate KV slots accordingly.
    pub local_kv_share_src: Vec<usize>,
    /// Rank-0 only: device-resident `token_embd`.
    pub token_embd: Option<DeviceTensor>,
    pub token_embd_dims: Option<[usize; 2]>,
    /// Last-rank only: device-resident `output_norm`.
    pub output_norm: Option<DeviceTensor>,
    /// Last-rank only: optional separate `output` (gemma4 ties).
    pub output: Option<DeviceTensor>,
}

impl Gemma4PpModelStage {
    pub fn rank(&self) -> usize {
        self.common.rank as usize
    }

    /// Free this rank's weight allocations.
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if self.common.is_disposed() {
            return Ok(());
        }
        self.common
            .dispose(device)
            .map_err(|e| anyhow!("model raw_alloc dispose: {e}"))?;
        // Globals were uploaded into their own DeviceTensors; free them.
        for t in self.token_embd.take().into_iter()
            .chain(self.output_norm.take().into_iter())
            .chain(self.output.take().into_iter())
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

impl Drop for Gemma4PpModelStage {
    fn drop(&mut self) {
        self.common
            .warn_on_leak("flambeau_gemma4::pp::Gemma4PpModelStage");
    }
}

/// Per-rank session state (KV caches + scratch — per request).
pub struct Gemma4PpSessionStage {
    /// Shared bookkeeping for **scratch + KV-scratch** allocations on
    /// this rank. `common.raw_alloc` holds the device buffers backing
    /// `scratch`, `prefill`, `hidden_a`, `hidden_b`, `output_head_scratch`,
    /// `moe_scratch`. KV caches live in their own `KvCache` objects.
    pub common: StageCommon,
    /// One KV cache per local layer; `None` for shared-KV tail layers
    /// (which read from another local entry resolved via the model
    /// stage's `local_kv_share_src`).
    pub kv_caches: Vec<Option<KvCache<F16Contig, HipDevice>>>,
    /// F16 [hidden] hidden ping-pong slot A.
    pub hidden_a: DevicePtr,
    /// F16 [hidden] hidden ping-pong slot B.
    pub hidden_b: DevicePtr,
    /// Last-rank only.
    pub output_head_scratch: Option<OutputHeadScratch>,
    scratch: LayerScratchPtrs,
    /// MoE composer scratch — `Some` when this stage owns at least one
    /// MoE layer (26B-A4B). Shared across layers since gemma4 26B-A4B
    /// has uniform MoE shape (`n_ff_exp`, `n_experts`, `top_k`).
    moe_scratch: Option<crate::moe::Gemma4MoeScratch>,
    /// Per-stage prefill scratch, sized for `max_tokens` rows.
    prefill: PrefillScratchPtrs,
    /// One 1-element host position slot per rank (HtoD'd into
    /// `scratch.positions` per layer).
    positions_host: Vec<i32>,
    /// Maximum prefill chunk size (number of tokens) this stage can
    /// handle in a single `forward_layers_prefill_in_stage` call.
    pub max_tokens: usize,
}

impl Gemma4PpSessionStage {
    pub fn rank(&self) -> usize {
        self.common.rank as usize
    }

    fn layer_scratch_view(&mut self) -> LayerDecodeScratch<'_> {
        LayerDecodeScratch {
            x_q8_1: self.scratch.x_q8_1.0,
            mmvq_f32: self.scratch.mmvq_f32.0,
            q_f16: self.scratch.q_f16.0,
            k_f16: self.scratch.k_f16.0,
            v_f16: self.scratch.v_f16.0,
            attn_out_f16: self.scratch.attn_out_f16.0,
            post_attn_norm_f16: self.scratch.post_attn_norm_f16.0,
            attn_residual_f16: self.scratch.attn_residual_f16.0,
            ffn_norm_f16: self.scratch.ffn_norm_f16.0,
            gate_f32: self.scratch.gate_f32.0,
            up_f32: self.scratch.up_f32.0,
            activated_f16: self.scratch.activated_f16.0,
            activated_q8_1: self.scratch.activated_q8_1.0,
            down_f32: self.scratch.down_f32.0,
            post_ffw_norm_f16: self.scratch.post_ffw_norm_f16.0,
            positions: self.scratch.positions.0,
            positions_host: &mut self.positions_host,
            v_ones_f16: self.scratch.v_ones_f16.0,
            splitk_partials_m: self.scratch.splitk_partials_m.0,
            splitk_partials_s: self.scratch.splitk_partials_s.0,
            splitk_partials_o: self.scratch.splitk_partials_o.0,
        }
    }

    fn prefill_scratch_view(&mut self) -> LayerPrefillScratch<'_> {
        LayerPrefillScratch {
            max_tokens: self.max_tokens,
            x_norm_f16: self.prefill.x_norm_f16.0,
            x_q8_1: self.prefill.x_q8_1.0,
            x_q8_1_mmq: self.prefill.x_q8_1_mmq.0,
            mmvq_f32: self.prefill.mmvq_f32.0,
            q_f16: self.prefill.q_f16.0,
            k_f16: self.prefill.k_f16.0,
            v_f16: self.prefill.v_f16.0,
            attn_out_f16: self.prefill.attn_out_f16.0,
            post_attn_norm_f16: self.prefill.post_attn_norm_f16.0,
            attn_residual_f16: self.prefill.attn_residual_f16.0,
            gate_f32: self.prefill.gate_f32.0,
            up_f32: self.prefill.up_f32.0,
            activated_f16: self.prefill.activated_f16.0,
            activated_q8_1: self.prefill.activated_q8_1.0,
            activated_q8_1_mmq: self.prefill.activated_q8_1_mmq.0,
            down_f32: self.prefill.down_f32.0,
            post_ffw_norm_f16: self.prefill.post_ffw_norm_f16.0,
            positions: self.prefill.positions.0,
            positions_host: &mut self.prefill.positions_host,
            v_ones_f16: self.prefill.v_ones_f16.0,
            gated_q8_1: self.prefill.gated_q8_1.0,
            gated_q8_1_mmq: self.prefill.gated_q8_1_mmq.0,
        }
    }

    /// Free this rank's session allocations (KV caches + scratch).
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

impl Drop for Gemma4PpSessionStage {
    fn drop(&mut self) {
        self.common
            .warn_on_leak("flambeau_gemma4::pp::Gemma4PpSessionStage");
    }
}

/// Per-rank pipeline stage. Bundles the weight half (`model`) and the
/// per-request half (`session`). Test fixtures + `Gemma4PpDriver`
/// construct this; production callers that want `Arc<Gemma4PpModel>`
/// sharing across multiple sessions split via
/// `Gemma4PpStage::into_halves`.
pub struct Gemma4PpStage {
    pub model: Gemma4PpModelStage,
    pub session: Gemma4PpSessionStage,
}

impl Gemma4PpStage {
    pub fn rank(&self) -> usize {
        self.model.rank()
    }

    /// Split the bundle into its (model, session) halves. Used by
    /// `Gemma4PpDriver::upload` to hand the Model half to `Arc::new`
    /// and keep the Session half as per-request state.
    pub fn into_halves(self) -> (Gemma4PpModelStage, Gemma4PpSessionStage) {
        (self.model, self.session)
    }
}

/// Pipeline-parallel model. Owns the cluster + per-rank weights +
/// per-rank `OpsRegistry`. Arc-shareable across multiple concurrent
/// `Gemma4PpSession`s.
pub struct Gemma4PpModel {
    pub cluster: HipCluster,
    pub cfg: Gemma4Config,
    pub layout: ModelLayout,
    pub layer_to_rank: Vec<usize>,
    pub stages: Vec<Gemma4PpModelStage>,
    /// One `OpsRegistry` per rank. Built once at driver init; bound
    /// to the device, reusable across requests.
    pub regs: Vec<OpsRegistry>,
}

impl Gemma4PpModel {
    /// Free every rank's weight allocations. Idempotent.
    pub fn dispose(&mut self) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for (rank, stage) in self.stages.iter_mut().enumerate() {
            let dev = self.cluster.device(rank);
            if let Err(e) = stage.dispose(dev) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

impl Drop for Gemma4PpModel {
    fn drop(&mut self) {
        if self
            .stages
            .iter()
            .any(|s| !s.common.is_disposed())
        {
            tracing::warn!("Gemma4PpModel dropped without dispose()");
        }
    }
}

/// Per-request pipeline-parallel session. Owns the KV caches + scratch
/// for every rank; borrows weights through `&Gemma4PpModel`.
pub struct Gemma4PpSession {
    pub stages: Vec<Gemma4PpSessionStage>,
    /// Last-rank logits scratch host buffer (vocab F32) for argmax.
    pub logits_host: Vec<f32>,
}

impl Gemma4PpSession {
    /// Allocate per-rank KV caches + scratch sized for `max_tokens`,
    /// using the model's layout/cluster.
    pub fn new(model: &Gemma4PpModel, max_tokens: usize) -> Result<Self> {
        let mut stages = Vec::with_capacity(model.stages.len());
        for (rank, model_stage) in model.stages.iter().enumerate() {
            let device = model.cluster.device(rank);
            stages.push(Gemma4PpSessionStage::from_pieces(
                device,
                &model.cfg,
                &model.layout,
                model_stage,
                max_tokens,
            )?);
        }
        let logits_host = vec![0.0f32; model.cfg.vocab_size];
        Ok(Self {
            stages,
            logits_host,
        })
    }

    /// Free every rank's session allocations against `model.cluster`.
    pub fn dispose(&mut self, model: &Gemma4PpModel) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for (rank, stage) in self.stages.iter_mut().enumerate() {
            let dev = model.cluster.device(rank);
            if let Err(e) = stage.dispose(dev) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

/// Borrowed bundle that `PpDecodeDriver` / `PpPrefillDriver` is
/// implemented on. Constructed per forward call from a
/// `(&Gemma4PpModel, &mut Gemma4PpSession)` pair so the topology
/// orchestrators in `flambeau-blocks` can drive a session over shared
/// weights.
pub struct Gemma4PpHandle<'a> {
    pub model: &'a Gemma4PpModel,
    pub session: &'a mut Gemma4PpSession,
}

/// Pipeline-parallel driver. Bundles `Arc<Gemma4PpModel>` (shared
/// weights) and `Gemma4PpSession` (per-request KV+scratch). Existing
/// single-request callers construct one Driver; multi-slot callers
/// construct one `Arc<Gemma4PpModel>` + N `Gemma4PpSession`s and use
/// `Gemma4PpHandle` directly.
pub struct Gemma4PpDriver {
    pub model: std::sync::Arc<Gemma4PpModel>,
    pub session: Gemma4PpSession,
}

impl Gemma4PpModelStage {
    /// Build the model-side stage for `rank`: resolves
    /// `global_layer_indices` + `local_kv_share_src` from the
    /// partition, wraps the passed-in pre-uploaded weights, and
    /// initialises an empty weight tracker (caller appends weight
    /// allocs via `common.raw_alloc.track`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_pieces(
        device: &HipDevice,
        rank: usize,
        layout: &ModelLayout,
        layer_to_rank: &[usize],
        layer_weights: Vec<Gemma4LayerWeights>,
        token_embd: Option<DeviceTensor>,
        token_embd_dims: Option<[usize; 2]>,
        output_norm: Option<DeviceTensor>,
        output: Option<DeviceTensor>,
    ) -> Result<Self> {
        let global_layer_indices: Vec<usize> = (0..layout.layers.len())
            .filter(|&i| layer_to_rank[i] == rank)
            .collect();
        if global_layer_indices.len() != layer_weights.len() {
            bail!(
                "Gemma4PpModelStage::from_pieces: rank {rank} owns {} layers but got {} weights",
                global_layer_indices.len(),
                layer_weights.len()
            );
        }
        let mut global_to_local = vec![usize::MAX; layout.layers.len()];
        for (li, &gi) in global_layer_indices.iter().enumerate() {
            global_to_local[gi] = li;
        }
        let mut local_kv_share_src = vec![0usize; global_layer_indices.len()];
        for (li, &gi) in global_layer_indices.iter().enumerate() {
            let spec = &layout.layers[gi];
            if spec.has_kv {
                local_kv_share_src[li] = li;
            } else {
                let src_g = spec.kv_share_src.expect("partition validated this");
                let src_l = global_to_local[src_g];
                if src_l == usize::MAX {
                    bail!(
                        "Gemma4PpModelStage::from_pieces: rank {rank} layer {gi} \
                         kv_share_src={src_g} not on same rank",
                    );
                }
                local_kv_share_src[li] = src_l;
            }
        }
        Ok(Self {
            common: StageCommon::new(rank as u32, device.id()),
            global_layer_indices,
            layer_weights,
            local_kv_share_src,
            token_embd,
            token_embd_dims,
            output_norm,
            output,
        })
    }
}

impl Gemma4PpSessionStage {
    /// Build the per-request session stage for `rank`. Reads layer
    /// shapes through `model` to size scratch / KV slots; allocates
    /// scratch into `common.raw_alloc`. `max_tokens` caps prefill
    /// chunk size.
    pub fn from_pieces(
        device: &HipDevice,
        cfg: &Gemma4Config,
        layout: &ModelLayout,
        model: &Gemma4PpModelStage,
        max_tokens: usize,
    ) -> Result<Self> {
        device.bind()?;

        // KV caches: one per owning local layer; None for tail.
        let mut kv_caches = Vec::with_capacity(model.global_layer_indices.len());
        for &gi in &model.global_layer_indices {
            let spec = &layout.layers[gi];
            if spec.has_kv {
                let kv =
                    KvCache::<F16Contig, HipDevice>::new(device, spec.n_kv_heads, spec.head_dim, max_tokens)
                        .map_err(|e| anyhow!("kv alloc layer {gi}: {e}"))?;
                kv_caches.push(Some(kv));
            } else {
                kv_caches.push(None);
            }
        }

        // Per-stage layer scratch. Sizes are widest over THIS stage's
        // layers — different stages may carry layers with different
        // shapes (per-layer head_dim/n_kv_heads).
        let hidden = cfg.hidden_size;
        let ff_len = cfg.feed_forward_length;
        let q_width_max = model
            .global_layer_indices
            .iter()
            .map(|&gi| layout.layers[gi].n_heads * layout.layers[gi].head_dim)
            .max()
            .unwrap_or(hidden);
        let kv_width_max_stage = model
            .global_layer_indices
            .iter()
            .map(|&gi| layout.layers[gi].n_kv_heads * layout.layers[gi].head_dim)
            .max()
            .unwrap_or(hidden);
        let head_dim_max = model
            .global_layer_indices
            .iter()
            .map(|&gi| layout.layers[gi].head_dim)
            .max()
            .unwrap_or(64);
        let mmvq_max = q_width_max.max(kv_width_max_stage).max(hidden).max(ff_len);
        let x_q8_1_n = hidden.max(ff_len).max(q_width_max).div_ceil(32) * 32;
        let activated_q8_1_n = ff_len.div_ceil(32) * 32;

        let mut common = StageCommon::new(model.common.rank, device.id());
        let raw_alloc = &mut common.raw_alloc;

        let v_ones_ptr = upload_f16_ones(device, head_dim_max)?;
        raw_alloc.track(v_ones_ptr, head_dim_max * 2);
        let n_heads_max_stage = model
            .global_layer_indices
            .iter()
            .map(|&gi| layout.layers[gi].n_heads)
            .max()
            .unwrap_or(1);
        let splitk_chunks = flambeau_blocks::MAX_SPLITK_CHUNKS;
        let scratch = LayerScratchPtrs {
            x_q8_1: raw_alloc.alloc_q8_1(device, x_q8_1_n)?,
            mmvq_f32: raw_alloc.alloc_f32(device, mmvq_max)?,
            q_f16: raw_alloc.alloc_f16(device, q_width_max)?,
            k_f16: raw_alloc.alloc_f16(device, kv_width_max_stage)?,
            v_f16: raw_alloc.alloc_f16(device, kv_width_max_stage)?,
            attn_out_f16: raw_alloc.alloc_f16(device, q_width_max.max(hidden))?,
            post_attn_norm_f16: raw_alloc.alloc_f16(device, hidden)?,
            attn_residual_f16: raw_alloc.alloc_f16(device, hidden)?,
            ffn_norm_f16: raw_alloc.alloc_f16(device, hidden)?,
            gate_f32: raw_alloc.alloc_f32(device, ff_len)?,
            up_f32: raw_alloc.alloc_f32(device, ff_len)?,
            activated_f16: raw_alloc.alloc_f16(device, ff_len)?,
            activated_q8_1: raw_alloc.alloc_q8_1(device, activated_q8_1_n)?,
            down_f32: raw_alloc.alloc_f32(device, hidden)?,
            post_ffw_norm_f16: raw_alloc.alloc_f16(device, hidden)?,
            positions: raw_alloc.alloc_i32(device, 1)?,
            v_ones_f16: (v_ones_ptr, head_dim_max * 2),
            splitk_partials_m: raw_alloc.alloc_f32(device, n_heads_max_stage * splitk_chunks)?,
            splitk_partials_s: raw_alloc.alloc_f32(device, n_heads_max_stage * splitk_chunks)?,
            splitk_partials_o: raw_alloc
                .alloc_f32(device, n_heads_max_stage * splitk_chunks * head_dim_max)?,
        };

        let hidden_a = raw_alloc.alloc_f16(device, max_tokens * hidden)?.0;
        let hidden_b = raw_alloc.alloc_f16(device, max_tokens * hidden)?.0;

        let n_heads_max = layout
            .layers
            .iter()
            .map(|l| l.n_heads * l.head_dim)
            .max()
            .unwrap_or(hidden);
        let kv_width_max_all = layout
            .layers
            .iter()
            .map(|l| l.n_kv_heads * l.head_dim)
            .max()
            .unwrap_or(hidden);
        let mmvq_max_p = n_heads_max.max(kv_width_max_all).max(hidden).max(ff_len);
        let pf_x_q8_1_n = max_tokens * x_q8_1_n;
        let pf_activated_q8_1_n = max_tokens * activated_q8_1_n;
        let prefill = PrefillScratchPtrs {
            x_norm_f16: raw_alloc.alloc_f16(device, max_tokens * hidden)?,
            x_q8_1: raw_alloc.alloc_q8_1(device, pf_x_q8_1_n)?,
            x_q8_1_mmq: raw_alloc.alloc_q8_1(device, pf_x_q8_1_n)?,
            mmvq_f32: raw_alloc.alloc_f32(device, max_tokens * mmvq_max_p)?,
            q_f16: raw_alloc.alloc_f16(device, max_tokens * n_heads_max)?,
            k_f16: raw_alloc.alloc_f16(device, max_tokens * kv_width_max_all)?,
            v_f16: raw_alloc.alloc_f16(device, max_tokens * kv_width_max_all)?,
            attn_out_f16: raw_alloc.alloc_f16(device, max_tokens * n_heads_max.max(hidden))?,
            post_attn_norm_f16: raw_alloc.alloc_f16(device, max_tokens * hidden)?,
            attn_residual_f16: raw_alloc.alloc_f16(device, max_tokens * hidden)?,
            gate_f32: raw_alloc.alloc_f32(device, max_tokens * ff_len)?,
            up_f32: raw_alloc.alloc_f32(device, max_tokens * ff_len)?,
            activated_f16: raw_alloc.alloc_f16(device, max_tokens * ff_len)?,
            activated_q8_1: raw_alloc.alloc_q8_1(device, pf_activated_q8_1_n)?,
            activated_q8_1_mmq: raw_alloc.alloc_q8_1(device, pf_activated_q8_1_n)?,
            down_f32: raw_alloc.alloc_f32(device, max_tokens * hidden)?,
            post_ffw_norm_f16: raw_alloc.alloc_f16(device, max_tokens * hidden)?,
            positions: raw_alloc.alloc_i32(device, max_tokens)?,
            v_ones_f16: (v_ones_ptr, head_dim_max * 2),
            gated_q8_1: raw_alloc.alloc_q8_1(device, max_tokens * n_heads_max)?,
            gated_q8_1_mmq: raw_alloc.alloc_q8_1_mmq(device, max_tokens * n_heads_max)?,
            positions_host: vec![0i32; max_tokens],
        };

        let output_head_scratch = if model.output_norm.is_some() {
            Some(OutputHeadScratch {
                x_norm_f16: raw_alloc.alloc_f16(device, hidden)?.0,
                x_q8_1: raw_alloc.alloc_q8_1(device, x_q8_1_n)?.0,
                logits_f32: raw_alloc.alloc_f32(device, cfg.vocab_size)?.0,
            })
        } else {
            None
        };

        let any_moe = model
            .global_layer_indices
            .iter()
            .any(|&gi| layout.layers[gi].ffn_kind == crate::layout::FfnKind::Moe);
        let moe_scratch = if any_moe {
            let dims = cfg
                .moe
                .ok_or_else(|| anyhow!("rank {}: MoE layer present but cfg.moe is None", model.common.rank))?;
            Some(crate::moe::Gemma4MoeScratch::alloc(
                device,
                hidden,
                dims.moe_intermediate_size,
                dims.num_experts,
                dims.num_experts_per_tok,
                raw_alloc,
            )?)
        } else {
            None
        };

        Ok(Self {
            common,
            kv_caches,
            hidden_a,
            hidden_b,
            output_head_scratch,
            scratch,
            moe_scratch,
            prefill,
            positions_host: vec![0i32; 1],
            max_tokens,
        })
    }
}

impl Gemma4PpStage {
    /// Build the bundled stage. Test-friendly constructor that wraps
    /// model + session halves into one struct (the real-GGUF
    /// `Gemma4PpDriver::upload` path lands in S8-B and now splits the
    /// halves into `Arc<Gemma4PpModel>` + `Gemma4PpSession`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_pieces(
        device: &HipDevice,
        rank: usize,
        cfg: &Gemma4Config,
        layout: &ModelLayout,
        layer_to_rank: &[usize],
        layer_weights: Vec<Gemma4LayerWeights>,
        token_embd: Option<DeviceTensor>,
        token_embd_dims: Option<[usize; 2]>,
        output_norm: Option<DeviceTensor>,
        output: Option<DeviceTensor>,
        max_tokens: usize,
    ) -> Result<Self> {
        device.bind()?;
        let model = Gemma4PpModelStage::from_pieces(
            device,
            rank,
            layout,
            layer_to_rank,
            layer_weights,
            token_embd,
            token_embd_dims,
            output_norm,
            output,
        )?;
        let session = Gemma4PpSessionStage::from_pieces(device, cfg, layout, &model, max_tokens)?;
        Ok(Self { model, session })
    }

    /// Dispose both halves. Test-fixture convenience — production code
    /// goes through `Gemma4PpModel::dispose` + `Gemma4PpSession::dispose`
    /// separately.
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        // Session first (KV caches frees first since they reference
        // the same device as the stage's allocations).
        self.session.dispose(device)?;
        self.model.dispose(device)?;
        Ok(())
    }
}

impl Gemma4PpDriver {
    /// Build the driver from per-rank stages. The caller is
    /// responsible for setting up `cluster.device(rank)` correctly
    /// (e.g. via `HipCluster::new(&device_ids)`) and for uploading
    /// the per-stage weights / globals matching the partition. The
    /// stages must have been built on the matching device. Splits
    /// each bundled `Gemma4PpStage` into its model + session halves;
    /// the model halves go into `Arc<Gemma4PpModel>`, the session
    /// halves into `Gemma4PpSession`.
    pub fn from_pieces(
        cluster: HipCluster,
        cfg: Gemma4Config,
        layout: ModelLayout,
        layer_to_rank: Vec<usize>,
        stages: Vec<Gemma4PpStage>,
    ) -> Result<Self> {
        if stages.is_empty() {
            bail!("Gemma4PpDriver::from_pieces: 0 stages");
        }
        let n_ranks = stages.len();
        if cluster.ranks() != n_ranks {
            bail!(
                "Gemma4PpDriver::from_pieces: cluster has {} ranks but got {} stages",
                cluster.ranks(),
                n_ranks
            );
        }
        if stages[0].model.token_embd.is_none() {
            bail!("Gemma4PpDriver::from_pieces: rank 0 must own token_embd");
        }
        if stages[n_ranks - 1].model.output_norm.is_none() {
            bail!("Gemma4PpDriver::from_pieces: last rank must own output_norm");
        }

        let mut regs = Vec::with_capacity(n_ranks);
        for rank in 0..n_ranks {
            let dev = cluster.device(rank);
            dev.bind()?;
            let reg = OpsRegistry::new(dev).map_err(|e| anyhow!("registry rank {rank}: {e}"))?;
            regs.push(reg);
        }
        let logits_host = vec![0.0f32; cfg.vocab_size];

        let mut model_stages = Vec::with_capacity(n_ranks);
        let mut session_stages = Vec::with_capacity(n_ranks);
        for stage in stages.into_iter() {
            let (m, s) = stage.into_halves();
            model_stages.push(m);
            session_stages.push(s);
        }
        let model = Gemma4PpModel {
            cluster,
            cfg,
            layout,
            layer_to_rank,
            stages: model_stages,
            regs,
        };
        let session = Gemma4PpSession {
            stages: session_stages,
            logits_host,
        };
        Ok(Self {
            model: std::sync::Arc::new(model),
            session,
        })
    }

    /// Real-GGUF upload: build the driver by streaming each rank's
    /// assigned layers from the GGUF mmap to that rank's device, with
    /// F32→F16 norm casts and tied LM-head replication on the last
    /// rank. Bails on MoE / per-layer-embd variants (those have their
    /// own follow-ups).
    pub fn upload(
        file: &GgufFile,
        cfg: Gemma4Config,
        layout: ModelLayout,
        layer_to_rank: Vec<usize>,
        cluster: HipCluster,
        max_tokens: usize,
    ) -> Result<Self> {
        if cluster.ranks() == 0 {
            bail!("Gemma4PpDriver::upload: cluster has 0 ranks");
        }
        if cfg.per_layer_embed.is_some() {
            bail!(
                "Gemma4PpDriver::upload: per-layer side-channel embedding \
                 (E2B/E4B) needs the per-layer-embd upload path"
            );
        }
        for spec in &layout.layers {
            if spec.ffn_kind == FfnKind::Moe && cfg.moe.is_none() {
                bail!(
                    "Gemma4PpDriver::upload: layer {} ffn_kind=Moe but cfg.moe is None",
                    spec.index
                );
            }
            if !spec.has_kv && spec.kv_share_src.is_none() {
                bail!(
                    "Gemma4PpDriver::upload: tail layer {} missing kv_share_src; \
                     run `ModelLayout::resolve_kv_sharing()` first",
                    spec.index
                );
            }
        }
        if layer_to_rank.len() != layout.layers.len() {
            bail!(
                "Gemma4PpDriver::upload: layer_to_rank len {} != num_layers {}",
                layer_to_rank.len(),
                layout.layers.len()
            );
        }

        let n_ranks = cluster.ranks();
        let g_names = GlobalNames::default_names();
        let token_embd_info = file
            .tensors
            .get(&g_names.token_embd)
            .ok_or_else(|| anyhow!("token_embd missing"))?;
        let token_embd_dims = [
            token_embd_info.dims[0] as usize,
            token_embd_info.dims[1] as usize,
        ];

        let mut stages: Vec<Gemma4PpStage> = Vec::with_capacity(n_ranks);
        for rank in 0..n_ranks {
            let device = cluster.device(rank);
            device.bind()?;
            let stream = device.default_stream();
            let mut raw: Vec<(DevicePtr, usize)> = Vec::new();

            // Per-layer uploads for this rank.
            let mut layer_weights: Vec<Gemma4LayerWeights> = Vec::new();
            for (i, spec) in layout.layers.iter().enumerate() {
                if layer_to_rank[i] != rank {
                    continue;
                }
                let lw = upload_layer_pp(file, spec, &cfg, device, stream, &mut raw)
                    .with_context(|| format!("rank {rank} layer {}", spec.index))?;
                layer_weights.push(lw);
            }

            // Globals: rank 0 owns token_embd; last rank owns output_norm
            // (F32→F16 cast) and a replica of the LM-head weight (separate
            // device alloc — gemma4 ties, so this re-uploads token_embd
            // from the GGUF mmap).
            let (token_embd, td_dims): (Option<DeviceTensor>, Option<[usize; 2]>) =
                if rank == 0 {
                    let t = upload_tensor_raw(file, token_embd_info, device, stream)
                        .context("rank 0 token_embd")?;
                    (Some(t), Some(token_embd_dims))
                } else {
                    (None, None)
                };

            let (output_norm, output) = if rank == n_ranks - 1 {
                let on_info = file
                    .tensors
                    .get(&g_names.output_norm)
                    .ok_or_else(|| anyhow!("output_norm missing"))?;
                let on_ptr = upload_norm_f32_as_f16(file, on_info, device, stream)
                    .context("output_norm cast")?;
                let on_bytes = (on_info.dims.iter().product::<u64>() as usize) * 2;
                let on = DeviceTensor {
                    ptr: on_ptr,
                    dtype: GgmlDType::F16,
                    bytes: on_bytes,
                };
                // LM head: either explicit `output.weight` (untied) or
                // re-upload `token_embd.weight` (tied; gemma4 default).
                let lm = if let Some(info) = file.tensors.get(&g_names.output) {
                    upload_tensor_raw(file, info, device, stream).context("output (untied)")?
                } else {
                    upload_tensor_raw(file, token_embd_info, device, stream)
                        .context("tied LM head replica (token_embd)")?
                };
                (Some(on), Some(lm))
            } else {
                (None, None)
            };

            // Build the bundled stage; its model half tracks weight
            // allocs (via `raw` appended below), its session half
            // tracks scratch allocs.
            let mut stage = Gemma4PpStage::from_pieces(
                device,
                rank,
                &cfg,
                &layout,
                &layer_to_rank,
                layer_weights,
                token_embd,
                td_dims,
                output_norm,
                output,
                max_tokens,
            )?;
            for (ptr, bytes) in raw {
                stage.model.common.raw_alloc.track(ptr, bytes);
            }
            // Last rank also needs token_embd_dims for the LM-head GEMM
            // shape (we keep it on every rank so callers can introspect,
            // but only the last rank consumes it in `output_head`).
            if rank == n_ranks - 1 && stage.model.token_embd_dims.is_none() {
                stage.model.token_embd_dims = Some(token_embd_dims);
            }
            stages.push(stage);
        }

        Self::from_pieces(cluster, cfg, layout, layer_to_rank, stages)
    }

    /// Free every device allocation. Idempotent. Requires unique
    /// ownership of the Model `Arc` (i.e. no other Sessions reference
    /// the same weights) to dispose the model half.
    pub fn dispose(&mut self) -> Result<()> {
        // Session first (KV caches before scratch tracker).
        self.session.dispose(&self.model)?;
        // Then the model — requires sole ownership of the Arc.
        match std::sync::Arc::get_mut(&mut self.model) {
            Some(m) => m.dispose()?,
            None => {
                tracing::warn!(
                    "Gemma4PpDriver::dispose: model Arc has other refs; \
                     model weights leak until all Sessions drop"
                );
            }
        }
        Ok(())
    }

    /// Build a fresh per-request driver over a shared `Arc<Gemma4PpModel>`.
    /// Allocates a new `Gemma4PpSession` (KV + scratch sized for
    /// `max_tokens`) without re-uploading weights. Used by the server's
    /// inflight pool to spin up N concurrent slots over one weight
    /// upload.
    pub fn new_session(
        model: std::sync::Arc<Gemma4PpModel>,
        max_tokens: usize,
    ) -> Result<Self> {
        let session = Gemma4PpSession::new(&model, max_tokens)?;
        Ok(Self { model, session })
    }

    /// Reset per-request state (KV write tails) so the slot can take a
    /// new request. Weights stay in place.
    pub fn reset_kv(&mut self) -> Result<()> {
        for stage in self.session.stages.iter_mut() {
            for kv in stage.kv_caches.iter_mut().flatten() {
                kv.clear();
            }
        }
        Ok(())
    }

    /// Forward one decode token through the pipeline. Returns the
    /// argmax token id.
    pub fn forward_one_token(&mut self, token_id: u32, position: usize) -> Result<u32> {
        let mut handle = Gemma4PpHandle {
            model: &self.model,
            session: &mut self.session,
        };
        forward_one_token_pp(&mut handle, token_id, position)
    }

    /// Multi-token prefill across the pipeline. Output is the LM-head
    /// logits' argmax for the LAST token of the chunk (consumed
    /// host-side from the driver's logits buffer after the call).
    pub fn forward_prefill(&mut self, tokens: &[u32], start_position: usize) -> Result<u32> {
        {
            let mut handle = Gemma4PpHandle {
                model: &self.model,
                session: &mut self.session,
            };
            forward_prefill_pp(&mut handle, tokens, start_position)?;
        }
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

impl flambeau_runtime::ModelDriver for Gemma4PpDriver {
    fn forward_prefill(&mut self, tokens: &[u32], start_position: usize) -> Result<u32> {
        Gemma4PpDriver::forward_prefill(self, tokens, start_position)
    }
    fn forward_one_token(&mut self, token_id: u32, position: usize) -> Result<u32> {
        Gemma4PpDriver::forward_one_token(self, token_id, position)
    }
    fn forward_prefill_logits(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        let _ = Gemma4PpDriver::forward_prefill(self, tokens, start_position)?;
        copy_logits_into(&self.session.logits_host, logits_out);
        Ok(())
    }
    fn forward_one_token_logits(
        &mut self,
        token_id: u32,
        position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        let _ = Gemma4PpDriver::forward_one_token(self, token_id, position)?;
        copy_logits_into(&self.session.logits_host, logits_out);
        Ok(())
    }
    fn vocab_size(&self) -> usize {
        self.model.cfg.vocab_size
    }
    fn reset_kv(&mut self) -> Result<()> {
        Gemma4PpDriver::reset_kv(self)
    }
    fn dispose(&mut self) -> Result<()> {
        Gemma4PpDriver::dispose(self)
    }
}

fn copy_logits_into(src: &[f32], dst: &mut Vec<f32>) {
    dst.clear();
    dst.extend_from_slice(src);
}

impl Drop for Gemma4PpDriver {
    fn drop(&mut self) {
        // Best-effort cleanup; warn on leak.
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
            tracing::warn!("Gemma4PpDriver dropped without dispose()");
        }
    }
}

// ---------------------------------------------------------------------------
// PpDecodeDriver impl
// ---------------------------------------------------------------------------

impl PpDecodeDriver for Gemma4PpHandle<'_> {
    fn n_ranks(&self) -> usize {
        self.model.stages.len()
    }

    fn layers_per_rank(&self, rank: usize) -> usize {
        self.model.stages[rank].global_layer_indices.len()
    }

    fn cluster(&self) -> &HipCluster {
        &self.model.cluster
    }

    fn hidden_a(&self, rank: usize) -> DevicePtr {
        self.session.stages[rank].hidden_a
    }

    fn hidden_b(&self, rank: usize) -> DevicePtr {
        self.session.stages[rank].hidden_b
    }

    fn hidden_bytes(&self) -> usize {
        self.model.cfg.hidden_size * 2
    }

    fn embed_token(&mut self, token_id: u32) -> Result<()> {
        let device = self.model.cluster.device(0);
        let stream = device.default_stream();
        let model_stage = &self.model.stages[0];
        let session_stage = &mut self.session.stages[0];
        let tok_embd = model_stage
            .token_embd
            .as_ref()
            .ok_or_else(|| anyhow!("embed_token: rank 0 missing token_embd"))?;
        embed_token_host(
            device,
            stream,
            tok_embd.ptr,
            tok_embd.dtype,
            tok_embd.bytes,
            self.model.cfg.vocab_size,
            self.model.cfg.hidden_size,
            token_id,
            session_stage.hidden_a,
        )?;
        // Gemma4 input scale: `inpL = scale(inpL, sqrt(n_embd))`
        // (`gemma4-iswa.cpp:20`). Without this, every downstream rmsnorm
        // computes the wrong scale (parity test #36 caught this on 31B
        // Q4_0).
        let reg = &self.model.regs[0];
        let ops = flambeau_ops::hip::HipOps::new(reg, stream);
        use flambeau_ops::Ops;
        ops.scale_f16(
            session_stage.hidden_a,
            session_stage.hidden_a,
            self.model.cfg.hidden_size,
            (self.model.cfg.hidden_size as f32).sqrt(),
        )
        .context("embed_token sqrt(n_embd) scale")?;
        Ok(())
    }

    fn forward_layer_decode(
        &mut self,
        rank: usize,
        local_idx: usize,
        x_in: DevicePtr,
        x_out: DevicePtr,
        position: usize,
    ) -> Result<()> {
        let device = self.model.cluster.device(rank);
        let stream = device.default_stream();
        let reg = &self.model.regs[rank];
        let ops = HipOps::new(reg, stream);

        let model_stage = &self.model.stages[rank];
        let session_stage = &mut self.session.stages[rank];
        let global_idx = model_stage.global_layer_indices[local_idx];
        let spec = self.model.layout.layers[global_idx];
        let weights = &model_stage.layer_weights[local_idx];

        let kv_local_idx = model_stage.local_kv_share_src[local_idx];
        let kv_ptr: *mut Option<KvCache<F16Contig, HipDevice>> =
            &mut session_stage.kv_caches[kv_local_idx];
        // Borrow moe_scratch by raw pointer so the scratch view (which
        // captures &mut on other fields) doesn't conflict.
        let moe_scratch_ref: Option<&crate::moe::Gemma4MoeScratch> =
            session_stage
                .moe_scratch
                .as_ref()
                .map(|s| s as *const _)
                .map(|p| {
                    // SAFETY: moe_scratch field is disjoint from the
                    // fields the scratch view touches.
                    unsafe { &*p }
                });
        let mut scratch = session_stage.layer_scratch_view();
        // SAFETY: kv_ptr borrows session_stage.kv_caches[kv_local_idx]
        // disjointly from the other fields the scratch view touches.
        let kv = unsafe { &mut *kv_ptr };
        let kv = kv
            .as_mut()
            .ok_or_else(|| anyhow!("rank {rank} local layer {local_idx} (global {global_idx}) kv slot unallocated"))?;

        forward_layer_decode(
            &ops,
            device,
            stream,
            weights,
            &spec,
            self.model.cfg.rms_norm_eps,
            self.model.cfg.feed_forward_length,
            self.model.cfg.hidden_size,
            kv,
            &mut scratch,
            x_in,
            x_out,
            position,
            /*per_layer_slice=*/ None,
            moe_scratch_ref,
        )
    }

    fn output_head(&mut self) -> Result<()> {
        let last = self.model.stages.len() - 1;
        let device = self.model.cluster.device(last);
        let stream = device.default_stream();
        let reg = &self.model.regs[last];
        let ops = HipOps::new(reg, stream);

        let cfg = &self.model.cfg;
        let model_stage = &self.model.stages[last];
        let session_stage = &mut self.session.stages[last];
        let scratch = session_stage
            .output_head_scratch
            .as_mut()
            .ok_or_else(|| anyhow!("output_head: last rank missing scratch"))?;
        let output_norm = model_stage
            .output_norm
            .as_ref()
            .ok_or_else(|| anyhow!("output_head: last rank missing output_norm"))?;
        let lm_head_t = model_stage
            .output
            .as_ref()
            .or(model_stage.token_embd.as_ref())
            .ok_or_else(|| anyhow!("output_head: last rank missing LM head weight"))?;
        let lm_head_dims = model_stage
            .token_embd_dims
            .ok_or_else(|| anyhow!("output_head: last rank missing token_embd_dims"))?;
        let lm_head = lm_head_t.as_weight_handle(lm_head_dims)?;
        let in_ptr = session_stage.hidden_a;
        let logits = forward_output_head(
            &ops,
            in_ptr,
            output_norm.ptr,
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
        let _ = apply_logit_softcap::<HipOps>;
        Ok(())
    }

    fn argmax(&self) -> Result<u32> {
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

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Upload one tensor verbatim from the GGUF mmap to the device, in
/// its native dtype. Returns the resulting [`DeviceTensor`] (the
/// caller does NOT need to track it for dispose — it goes through
/// `stage.token_embd`/`output`/`output_norm` slots which `dispose`
/// already frees). Used for token_embd + LM-head replica.
fn upload_tensor_raw(
    file: &GgufFile,
    info: &TensorInfo,
    device: &HipDevice,
    stream: &HipStream,
) -> Result<DeviceTensor> {
    let bytes = info.size_in_bytes() as usize;
    let data = file
        .tensor_raw(&info.name)
        .with_context(|| format!("tensor_raw `{}`", info.name))?;
    if data.len() < bytes {
        bail!(
            "tensor `{}` mmap slice {} < declared {}",
            info.name,
            data.len(),
            bytes
        );
    }
    let ptr = device
        .alloc(bytes)
        .map_err(|e| anyhow!("hipMalloc {} B for `{}`: {e}", bytes, info.name))?;
    // SAFETY: ptr is a fresh HIP alloc of `bytes`; data is an mmap
    // view of ≥ bytes host bytes.
    unsafe {
        device
            .memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(data.as_ptr() as usize),
                bytes,
            )
            .map_err(|e| anyhow!("memcpy_async `{}`: {e}", info.name))?;
    }
    stream.synchronize()?;
    Ok(DeviceTensor {
        ptr,
        dtype: info.dtype,
        bytes,
    })
}

/// Read an F32 norm weight from the GGUF mmap, cast each lane to F16
/// host-side, upload the F16 buffer to the device, and track the
/// allocation in `raw` for dispose. Returns the device pointer ready
/// for `rmsnorm_f16`. F16 inputs pass through with a raw upload
/// (still tracked).
fn upload_norm_f32_as_f16(
    file: &GgufFile,
    info: &TensorInfo,
    device: &HipDevice,
    stream: &HipStream,
) -> Result<DevicePtr> {
    let elems: usize = info.dims.iter().product::<u64>() as usize;
    match info.dtype {
        GgmlDType::F32 => {
            let raw_bytes = file
                .tensor_raw(&info.name)
                .with_context(|| format!("tensor_raw `{}`", info.name))?;
            if raw_bytes.len() < elems * 4 {
                bail!(
                    "norm `{}` mmap slice {} < expected {}",
                    info.name,
                    raw_bytes.len(),
                    elems * 4
                );
            }
            // SAFETY: dtype == F32 means the mmap region is F32 lanes;
            // mmap is page-aligned which exceeds 4-byte alignment.
            let src: &[f32] = bytemuck::cast_slice(&raw_bytes[..elems * 4]);
            let host: Vec<f16> = src.iter().map(|&v| f16::from_f32(v)).collect();
            let new_bytes = elems * 2;
            let new_ptr = device
                .alloc(new_bytes)
                .map_err(|e| anyhow!("alloc F16 norm `{}`: {e}", info.name))?;
            // SAFETY: new_ptr owns new_bytes; host outlives the sync.
            unsafe {
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        new_ptr,
                        DevicePtr(host.as_ptr() as usize),
                        new_bytes,
                    )
                    .map_err(|e| anyhow!("memcpy F32→F16 `{}`: {e}", info.name))?;
            }
            stream.synchronize()?;
            drop(host);
            Ok(new_ptr)
        }
        GgmlDType::F16 => {
            let t = upload_tensor_raw(file, info, device, stream)?;
            Ok(t.ptr)
        }
        other => bail!(
            "norm `{}`: unsupported dtype {:?} (expected F32 or F16)",
            info.name,
            other
        ),
    }
}


/// Upload every tensor for one layer (attention + dense FFN), with
/// F32→F16 cast for norms. The layer's `attn_k` / `attn_v` /
/// `attn_k_norm` are honoured optional per the alt-attention +
/// shared-KV-tail rules. `raw` collects every device alloc this
/// function makes so the caller can dispose them.
/// Read the per-layer F32 scalar `layer_output_scale` host-side. This
/// isn't a `WeightRole` because flambeau holds the value as an
/// `Option<f32>` (no device buffer), so the typed uploader has nothing
/// to do.
fn read_layer_output_scale(file: &GgufFile, spec: &LayerSpec) -> Result<Option<f32>> {
    let name = AttnNames::for_layer(spec.index).layer_output_scale;
    let Some(info) = file.tensors.get(&name) else {
        return Ok(None);
    };
    if info.dtype != GgmlDType::F32 {
        bail!(
            "layer {}: layer_output_scale must be F32, got {:?}",
            spec.index,
            info.dtype
        );
    }
    let raw_bytes = file
        .tensor_raw(&info.name)
        .with_context(|| format!("tensor_raw `{}`", info.name))?;
    if raw_bytes.len() < 4 {
        bail!("layer {}: layer_output_scale row < 4 bytes", spec.index);
    }
    Ok(Some(f32::from_le_bytes([
        raw_bytes[0],
        raw_bytes[1],
        raw_bytes[2],
        raw_bytes[3],
    ])))
}

fn upload_layer_pp(
    file: &GgufFile,
    spec: &LayerSpec,
    cfg: &Gemma4Config,
    device: &HipDevice,
    stream: &HipStream,
    raw: &mut Vec<(DevicePtr, usize)>,
) -> Result<Gemma4LayerWeights> {
    let il = spec.index;
    let mut tracker = RawAllocTracker::new();
    let mut up = WeightUploader {
        device,
        stream,
        tracker: &mut tracker,
        file,
        cfg,
        world: 1,
        rank: 0,
    };

    let attn_norm = up.upload_norm_required::<AttnNorm>(il)?;
    let attn_q = up.upload_matmul_required::<AttnQ>(il)?;
    let attn_k = up.upload_matmul::<AttnK>(il)?;
    let attn_v = up.upload_matmul::<AttnV>(il)?;
    let attn_output = up.upload_matmul_required::<AttnOutput>(il)?;
    let attn_q_norm = up.upload_norm_required::<AttnQNorm>(il)?;
    let attn_k_norm = up.upload_norm::<AttnKNorm>(il)?;
    let post_attention_norm = up.upload_norm_required::<PostAttnNorm>(il)?;
    // F32 copy of `post_attention_norm` for the F32 attention output
    // path. PP single-rank runs the full q_width output_proj F32 mmvq
    // → F16 cast, which saturates on Q8_0 + head_dim≥256 even for SWA
    // layers (the TP path doesn't saturate because the per-rank partial
    // mmvq output is half the sum). Upload F32 norm for every MoE
    // layer in the PP path. Non-MoE PP (31B Q4_0) is unaffected
    // because Q4_0 rounds the V-norm spike.
    let post_attention_norm_f32_ptr: Option<DevicePtr> =
        if cfg.moe.is_some() {
            let name = crate::names::AttnNames::for_layer(il).post_attention_norm;
            let info = file
                .tensors
                .get(&name)
                .ok_or_else(|| anyhow!("{name} missing for F32 upload"))?;
            Some(
                flambeau_blocks::upload_replicated_tensor(
                    file, info, device, stream, up.tracker,
                )
                .with_context(|| format!("upload F32 post_attention_norm layer {il}"))?
                .ptr,
            )
        } else {
            None
        };
    let ffn_norm = up.upload_norm_required::<FfnNorm>(il)?;
    let ffn_gate = up.upload_matmul_required::<FfnGate>(il)?;
    let ffn_up = up.upload_matmul_required::<FfnUp>(il)?;
    let ffn_down = up.upload_matmul_required::<FfnDown>(il)?;
    let post_ffw_norm = up.upload_norm_required::<PostFfwNorm>(il)?;
    drop(up);

    // Gemma4 quirks the typed roles don't model: shared-KV invariants +
    // host-side F32 scalar `layer_output_scale`.
    if spec.has_kv {
        if attn_k.is_none() {
            bail!("layer {il}: attn_k required for has_kv layer");
        }
        if attn_k_norm.is_none() {
            bail!("layer {il}: attn_k_norm required for has_kv layer");
        }
    }
    let layer_output_scale = read_layer_output_scale(file, spec)?;

    // Drain the local tracker into the PP-driver `raw` vec so dispose
    // chain remains identical.
    for (ptr, bytes) in std::mem::take(&mut tracker.allocs) {
        raw.push((ptr, bytes));
    }
    let _ = tracker.dispose(device);

    let moe = if spec.ffn_kind == FfnKind::Moe {
        let moe_dims = cfg
            .moe
            .ok_or_else(|| anyhow!("layer {} ffn_kind=Moe but cfg.moe is None", spec.index))?;
        // Adapter: PP's raw is `Vec<(DevicePtr, usize)>`; the upload
        // helper writes into a `RawAllocTracker`. Drain into PP's vec.
        let mut moe_tracker = flambeau_blocks::RawAllocTracker::new();
        let weights = crate::weights_hip::upload_moe_layer(
            file,
            spec.index,
            cfg.hidden_size,
            moe_dims,
            device,
            stream,
            &mut moe_tracker,
        )?;
        // Move the tracked allocs into PP's dispose list. Mark the
        // tracker disposed-without-free so its Drop doesn't warn.
        for (ptr, bytes) in std::mem::take(&mut moe_tracker.allocs) {
            raw.push((ptr, bytes));
        }
        let _ = moe_tracker.dispose(device);
        Some(weights)
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
        post_attention_norm_f32: post_attention_norm_f32_ptr,
        layer_output_scale,
        ffn_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
        post_ffw_norm,
        per_layer_embed: None,
        moe,
        tp_moe: None,
    })
}

#[allow(dead_code)]
fn _context_keepalive<E>(e: Result<()>) -> Result<()> {
    use Context as _Use;
    e.context("keepalive")
}

// ---------------------------------------------------------------------------
// PpPrefillDriver impl
// ---------------------------------------------------------------------------

impl PpPrefillDriver for Gemma4PpHandle<'_> {
    fn n_ranks(&self) -> usize {
        self.model.stages.len()
    }

    fn layers_per_rank(&self, rank: usize) -> usize {
        self.model.stages[rank].global_layer_indices.len()
    }

    fn cluster(&self) -> &HipCluster {
        &self.model.cluster
    }

    fn hidden_a(&self, rank: usize) -> DevicePtr {
        self.session.stages[rank].hidden_a
    }

    fn hidden_b(&self, rank: usize) -> DevicePtr {
        self.session.stages[rank].hidden_b
    }

    fn hidden_row_bytes(&self) -> usize {
        self.model.cfg.hidden_size * 2
    }

    fn max_tokens(&self) -> usize {
        // All session stages allocated with the same max_tokens.
        self.session.stages[0].max_tokens
    }

    fn embed_tokens(&mut self, tokens: &[u32]) -> Result<()> {
        let device = self.model.cluster.device(0);
        let stream = device.default_stream();
        let model_stage = &self.model.stages[0];
        let session_stage = &mut self.session.stages[0];
        let tok_embd = model_stage
            .token_embd
            .as_ref()
            .ok_or_else(|| anyhow!("embed_tokens: rank 0 missing token_embd"))?;
        let hidden = self.model.cfg.hidden_size;
        let vocab = self.model.cfg.vocab_size;
        let row_bytes = row_bytes_for_dtype(tok_embd.dtype, hidden)?;
        let mut host = vec![half::f16::from_f32(0.0); tokens.len() * hidden];
        for (i, &tok) in tokens.iter().enumerate() {
            if (tok as usize) >= vocab {
                bail!("token {tok} >= vocab {vocab}");
            }
            let offset = tok as usize * row_bytes;
            if offset + row_bytes > tok_embd.bytes {
                bail!("tok_embd row OOB at token {tok}");
            }
            let src = tok_embd.ptr.offset_bytes(offset);
            let mut row_raw = vec![0u8; row_bytes];
            // SAFETY: src has >= row_bytes valid bytes.
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
            let row_f16: Vec<half::f16> = if tok_embd.dtype == GgmlDType::F16 {
                bytemuck::cast_slice::<u8, half::f16>(&row_raw).to_vec()
            } else {
                let row_f32 = flambeau_quant::dequantize_to_vec(tok_embd.dtype, &row_raw, hidden)
                    .map_err(|e| anyhow!("dequant tok_embd row {tok}: {e}"))?;
                row_f32.into_iter().map(half::f16::from_f32).collect()
            };
            host[i * hidden..(i + 1) * hidden].copy_from_slice(&row_f16);
        }
        // Gemma4 input scale: `inpL = scale(inpL, sqrt(n_embd))`
        // (`gemma4-iswa.cpp:20`). Applied host-side here so the
        // uploaded F16 already has the correct magnitude.
        let scale = (hidden as f32).sqrt();
        for v in host.iter_mut() {
            *v = half::f16::from_f32(v.to_f32() * scale);
        }
        let bytes = host.len() * 2;
        // SAFETY: session_stage.hidden_a sized max_tokens * hidden * 2
        // bytes; tokens.len() * hidden * 2 <= bytes (checked by driver
        // orchestrator before this call).
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                session_stage.hidden_a,
                DevicePtr(host.as_ptr() as usize),
                bytes,
            )?;
        }
        stream.synchronize()?;
        Ok(())
    }

    fn forward_layer_prefill(
        &mut self,
        rank: usize,
        local_idx: usize,
        x_in: DevicePtr,
        x_out: DevicePtr,
        n_tokens: usize,
        start_position: usize,
    ) -> Result<()> {
        let device = self.model.cluster.device(rank);
        let stream = device.default_stream();
        let reg = &self.model.regs[rank];
        let ops = HipOps::new(reg, stream);

        let model_stage = &self.model.stages[rank];
        let session_stage = &mut self.session.stages[rank];
        let global_idx = model_stage.global_layer_indices[local_idx];
        let spec = self.model.layout.layers[global_idx];
        let weights = &model_stage.layer_weights[local_idx];

        let kv_local_idx = model_stage.local_kv_share_src[local_idx];
        let kv_ptr: *mut Option<KvCache<F16Contig, HipDevice>> =
            &mut session_stage.kv_caches[kv_local_idx];
        let mut scratch = session_stage.prefill_scratch_view();
        // SAFETY: kv_ptr disjoint from prefill scratch fields.
        let kv = unsafe { &mut *kv_ptr };
        let kv = kv
            .as_mut()
            .ok_or_else(|| anyhow!("rank {rank} local layer {local_idx} kv unallocated"))?;

        forward_layer_prefill(
            &ops,
            device,
            stream,
            weights,
            &spec,
            self.model.cfg.rms_norm_eps,
            self.model.cfg.feed_forward_length,
            self.model.cfg.hidden_size,
            kv,
            &mut scratch,
            x_in,
            x_out,
            n_tokens,
            start_position,
        )
    }

    fn output_head_last_token(&mut self, l: usize) -> Result<()> {
        let last = self.model.stages.len() - 1;
        let device = self.model.cluster.device(last);
        let stream = device.default_stream();
        let reg = &self.model.regs[last];
        let ops = HipOps::new(reg, stream);
        let cfg = &self.model.cfg;
        let model_stage = &self.model.stages[last];
        let session_stage = &mut self.session.stages[last];
        let scratch = session_stage
            .output_head_scratch
            .as_mut()
            .ok_or_else(|| anyhow!("output_head: last rank missing scratch"))?;
        let output_norm = model_stage
            .output_norm
            .as_ref()
            .ok_or_else(|| anyhow!("output_head: last rank missing output_norm"))?;
        let lm_head_t = model_stage
            .output
            .as_ref()
            .or(model_stage.token_embd.as_ref())
            .ok_or_else(|| anyhow!("output_head: last rank missing LM head weight"))?;
        let lm_head_dims = model_stage
            .token_embd_dims
            .ok_or_else(|| anyhow!("output_head: last rank missing token_embd_dims"))?;
        let lm_head = lm_head_t.as_weight_handle(lm_head_dims)?;
        let last_row_offset = (l - 1) * cfg.hidden_size * 2;
        let in_ptr = session_stage.hidden_a.offset_bytes(last_row_offset);
        let logits = forward_output_head(
            &ops,
            in_ptr,
            output_norm.ptr,
            lm_head,
            cfg.final_logit_softcap,
            scratch,
            cfg.hidden_size,
            cfg.vocab_size,
            cfg.rms_norm_eps,
        )?;
        // SAFETY: logits vocab*4 bytes; host buffer matches.
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
        let _ = apply_logit_softcap::<HipOps>;
        Ok(())
    }
}
