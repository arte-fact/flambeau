//! `LayerKind` — sum-of-blocks for one transformer layer plus the
//! per-half dispatch entry points the topology drivers consume.
//!
//! `AttnState` carries the runtime mutable state that varies by attn
//! variant (KV-cache for full-attn, recurrent state + conv history
//! for delta-net). `AttnScratch` carries the workspace view shaped
//! for whichever block is selected. The dispatch methods on
//! `AttnBlock` and `FfnBlock` are the entry points; mismatched
//! `(block, state, scratch)` triples bail at runtime.

use anyhow::{bail, Result};
use flambeau_backend_hip::{BarP2pAllReduce, HipDevice, HipStream};
use flambeau_core::DevicePtr;
use flambeau_ops::Ops;
use flambeau_runtime::{F16Contig, KvCache, Q8Contig};

use crate::{
    AttnDecodeSlots, AttnPrefillSlots, DeltaNetLayer, DeltaNetLayerDecodeScratch, DenseMlp,
    DenseMlpDecodeScratch, DenseMlpPrefillScratch, MoeExperts, MoeExpertsDecodeScratch,
    StandardAttention, StandardAttentionDecodeScratch, StandardAttentionPrefillScratch,
};

/// Attention-half block for one layer.
pub enum AttnBlock {
    Standard(StandardAttention),
    DeltaNet(DeltaNetLayer),
}

/// Feed-forward-half block for one layer.
pub enum FfnBlock {
    Dense(DenseMlp),
    Moe(MoeExperts),
}

/// One transformer layer's block composition.
pub struct LayerKind {
    pub attn: AttnBlock,
    pub ffn: FfnBlock,
}

/// Runtime mutable attention state borrowed for one decode call.
pub enum AttnState<'a> {
    /// F16 KV-cache; supplied to `Standard` blocks.
    KvF16(&'a mut KvCache<F16Contig, HipDevice>),
    /// Q8 KV-cache; supplied to `Standard` blocks.
    KvQ8(&'a mut KvCache<Q8Contig, HipDevice>),
    /// Recurrent state (delta-net) — `state` and `conv_history` are
    /// mutated in-place by the kernel sequence.
    Recurrent { state: DevicePtr, conv_history: DevicePtr },
}

/// Per-call workspace view sized for one decode step.
pub enum AttnDecodeScratch<'a> {
    Standard(StandardAttentionDecodeScratch<'a>),
    DeltaNet(DeltaNetLayerDecodeScratch),
}

/// Per-call workspace view for one prefill chunk.
pub enum AttnPrefillScratch<'a> {
    Standard(StandardAttentionPrefillScratch<'a>),
}

/// Per-call FFN workspace view (decode).
pub enum FfnDecodeScratch {
    Dense(DenseMlpDecodeScratch),
    Moe(MoeExpertsDecodeScratch),
}

/// Per-call FFN workspace view (prefill). Only `Dense` covered today;
/// MoE prefill keeps the existing free-fn code in qwen3-moe.
pub enum FfnPrefillScratch {
    Dense(DenseMlpPrefillScratch),
}

impl AttnBlock {
    /// Single-token decode. The block writes the pre-residual delta
    /// to `delta_out`; the caller composes the residual + post-attn
    /// norm as it sees fit (the topology driver does this).
    ///
    /// `slots` are graph-capture-only and accepted only by
    /// `Standard`. `DeltaNet + Some(slots)` bails since GDN kernels
    /// are not graph-capture-aware.
    pub fn forward_decode<O: Ops>(
        &self,
        ops: &O,
        device: &HipDevice,
        stream: &HipStream,
        x_in: DevicePtr,
        delta_out: DevicePtr,
        state: AttnState<'_>,
        scratch: AttnDecodeScratch<'_>,
        position: usize,
        slots: Option<AttnDecodeSlots>,
    ) -> Result<()> {
        match (self, state, scratch) {
            (AttnBlock::Standard(blk), AttnState::KvF16(kv), AttnDecodeScratch::Standard(mut s)) => {
                blk.forward_decode(ops, device, stream, x_in, delta_out, kv, &mut s, position, slots)
            }
            (AttnBlock::Standard(blk), AttnState::KvQ8(kv), AttnDecodeScratch::Standard(mut s)) => {
                blk.forward_decode(ops, device, stream, x_in, delta_out, kv, &mut s, position, slots)
            }
            (
                AttnBlock::DeltaNet(blk),
                AttnState::Recurrent { state, conv_history },
                AttnDecodeScratch::DeltaNet(s),
            ) => {
                if slots.is_some() {
                    bail!(
                        "AttnBlock::DeltaNet::forward_decode: graph-capture slots are not supported on the recurrent path"
                    );
                }
                blk.forward_decode(
                    ops,
                    device,
                    stream,
                    x_in,
                    delta_out,
                    state,
                    conv_history,
                    s,
                )
            }
            _ => bail!("AttnBlock::forward_decode: variant mismatch between block, state, and scratch"),
        }
    }

    /// Multi-token prefill. `DeltaNet` prefill is not on this surface;
    /// callers handling recurrent prefill go through their own path.
    pub fn forward_prefill<O: Ops>(
        &self,
        ops: &O,
        device: &HipDevice,
        stream: &HipStream,
        x_in: DevicePtr,
        delta_out: DevicePtr,
        state: AttnState<'_>,
        scratch: AttnPrefillScratch<'_>,
        n_tokens: usize,
        start_position: usize,
        slots: Option<AttnPrefillSlots>,
    ) -> Result<()> {
        match (self, state, scratch) {
            (AttnBlock::Standard(blk), AttnState::KvF16(kv), AttnPrefillScratch::Standard(mut s)) => {
                blk.forward_prefill(
                    ops,
                    device,
                    stream,
                    x_in,
                    delta_out,
                    kv,
                    &mut s,
                    n_tokens,
                    start_position,
                    slots,
                )
            }
            (AttnBlock::Standard(blk), AttnState::KvQ8(kv), AttnPrefillScratch::Standard(mut s)) => {
                blk.forward_prefill(
                    ops,
                    device,
                    stream,
                    x_in,
                    delta_out,
                    kv,
                    &mut s,
                    n_tokens,
                    start_position,
                    slots,
                )
            }
            (AttnBlock::DeltaNet(_), _, _) => bail!(
                "AttnBlock::forward_prefill: DeltaNet prefill is not implemented at the block surface"
            ),
            _ => bail!("AttnBlock::forward_prefill: variant mismatch"),
        }
    }
}

impl FfnBlock {
    /// Decode-step FFN. `extra_residual` is the optional shared-expert
    /// delta that some MoE architectures add before the routed-expert
    /// combine; `Dense` rejects it. `MoE` requires the caller to have
    /// populated the router's `expert_ids` / `expert_weights` (the
    /// current `MoeExpertsDecodeScratch` carries these).
    pub fn forward_decode<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        residual: DevicePtr,
        extra_residual: Option<DevicePtr>,
        x_out: DevicePtr,
        scratch: FfnDecodeScratch,
    ) -> Result<()> {
        match (self, scratch) {
            (FfnBlock::Dense(blk), FfnDecodeScratch::Dense(s)) => {
                if extra_residual.is_some() {
                    bail!("FfnBlock::Dense::forward_decode: extra_residual is not supported on the dense path");
                }
                blk.forward_decode(ops, x_norm, residual, x_out, s)
            }
            (FfnBlock::Moe(blk), FfnDecodeScratch::Moe(s)) => {
                blk.forward_decode(ops, x_norm, residual, extra_residual, x_out, s)
            }
            _ => bail!("FfnBlock::forward_decode: variant mismatch between block and scratch"),
        }
    }

    /// Multi-token prefill. Only `Dense` is on this surface; MoE
    /// prefill stays on its free-fn code.
    pub fn forward_prefill<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        residual: DevicePtr,
        x_out: DevicePtr,
        scratch: FfnPrefillScratch,
        n_tokens: usize,
    ) -> Result<()> {
        match (self, scratch) {
            (FfnBlock::Dense(blk), FfnPrefillScratch::Dense(s)) => {
                blk.forward_prefill(ops, x_norm, residual, x_out, s, n_tokens)
            }
            (FfnBlock::Moe(_), _) => bail!(
                "FfnBlock::forward_prefill: MoE prefill is not implemented at the block surface"
            ),
        }
    }
}

/// Build an `AttnState` from an `AttnBlock` + a `LayerCache`-shaped
/// runtime cache. The model crate is responsible for matching the
/// block variant to the cache variant; this helper just exposes the
/// expected pairings without assuming the model crate's `LayerCache`
/// type. Callers do their own match and call this once.
impl<'a> AttnState<'a> {
    pub fn standard_f16(kv: &'a mut KvCache<F16Contig, HipDevice>) -> Self {
        Self::KvF16(kv)
    }

    pub fn standard_q8(kv: &'a mut KvCache<Q8Contig, HipDevice>) -> Self {
        Self::KvQ8(kv)
    }

    pub fn recurrent(state: DevicePtr, conv_history: DevicePtr) -> Self {
        Self::Recurrent { state, conv_history }
    }
}

/// `out = residual_in + rmsnorm_f16(x, norm_w)` over `n_tokens * hidden`
/// F16 elements. `norm_out_tmp` is the intermediate the RMSNorm writes
/// to; pass the same pointer as `x` for an in-place norm.
///
/// Covers the gemma4 "post_attention_norm + residual add" pattern
/// (called twice per layer: post-attn and post-FFW) used by the PP,
/// single-device, and TP composers.
#[allow(clippy::too_many_arguments)]
pub fn post_norm_residual_f16<O: Ops>(
    ops: &O,
    x: DevicePtr,
    norm_w: DevicePtr,
    norm_out_tmp: DevicePtr,
    residual_in: DevicePtr,
    out: DevicePtr,
    n_tokens: usize,
    hidden: usize,
    rms_eps: f32,
) -> Result<()> {
    ops.rmsnorm_f16(x, norm_w, norm_out_tmp, n_tokens, hidden, rms_eps)?;
    ops.add_f16(residual_in, norm_out_tmp, out, n_tokens * hidden)?;
    Ok(())
}

/// AllReduce-sum the per-rank `partials[]` (in place — each rank ends up
/// holding the full-hidden sum) over the matching `streams[]`. Wraps
/// `BarP2pAllReduce::sum_tp{2,4}` so model crates stop replicating the
/// `[DevicePtr; N]` / `[&HipStream; N]` plumbing and the rank-count
/// match arm.
///
/// # Safety
/// Inherits the contract of [`BarP2pAllReduce::sum_tp2`] /
/// [`BarP2pAllReduce::sum_tp4`]:
/// - every `partials[r]` must point at a buffer of at least `n_elems` F16
///   on rank `r`;
/// - the caller must order subsequent reads of `partials[r]` after the
///   `streams[r]` work completes;
/// - the streams must outlive the launch.
pub unsafe fn tp_allreduce_sum_into(
    ar: &BarP2pAllReduce,
    partials: &[DevicePtr],
    n_elems: usize,
    streams: &[&HipStream],
) -> Result<()> {
    if partials.len() != streams.len() {
        bail!(
            "tp_allreduce_sum_into: partials.len()={} != streams.len()={}",
            partials.len(),
            streams.len(),
        );
    }
    match partials.len() {
        2 => {
            let p: [DevicePtr; 2] = [partials[0], partials[1]];
            let s: [&HipStream; 2] = [streams[0], streams[1]];
            unsafe { ar.sum_tp2(&p, n_elems as u32, &s) }
                .map_err(|e| anyhow::anyhow!("AR sum_tp2: {e}"))?;
        }
        4 => {
            let p: [DevicePtr; 4] = [partials[0], partials[1], partials[2], partials[3]];
            let s: [&HipStream; 4] = [streams[0], streams[1], streams[2], streams[3]];
            unsafe { ar.sum_tp4(&p, n_elems as u32, &s) }
                .map_err(|e| anyhow::anyhow!("AR sum_tp4: {e}"))?;
        }
        n => bail!("tp_allreduce_sum_into: unsupported tp_size {n}"),
    }
    Ok(())
}

