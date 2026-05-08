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
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_core::DevicePtr;
use flambeau_ops::Ops;
use flambeau_runtime::{F16Contig, KvCache, Q8Contig};

use crate::{
    DeltaNetLayer, DeltaNetLayerDecodeScratch, DenseMlp, DenseMlpDecodeScratch,
    DenseMlpPrefillScratch, MoeExperts, MoeExpertsDecodeScratch, StandardAttention,
    StandardAttentionDecodeScratch, StandardAttentionPrefillScratch,
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
    ) -> Result<()> {
        match (self, state, scratch) {
            (AttnBlock::Standard(blk), AttnState::KvF16(kv), AttnDecodeScratch::Standard(mut s)) => {
                blk.forward_decode(ops, device, stream, x_in, delta_out, kv, &mut s, position)
            }
            (AttnBlock::Standard(blk), AttnState::KvQ8(kv), AttnDecodeScratch::Standard(mut s)) => {
                blk.forward_decode(ops, device, stream, x_in, delta_out, kv, &mut s, position)
            }
            (
                AttnBlock::DeltaNet(blk),
                AttnState::Recurrent { state, conv_history },
                AttnDecodeScratch::DeltaNet(s),
            ) => blk.forward_decode(
                ops,
                device,
                stream,
                x_in,
                delta_out,
                state,
                conv_history,
                s,
            ),
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

