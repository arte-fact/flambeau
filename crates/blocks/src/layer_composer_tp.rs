//! Generic TP decode-layer composer.
//!
//! `LayerComposerTp` factors the universal 6-phase TP decode-layer
//! shape out of model crates. The model supplies per-rank attention
//! and FFN forwards plus per-rank post-AR integration hooks; the
//! composer drives the rank-iteration loops + the typed AR
//! transitions in between.
//!
//! Canonical phase order (gemma4-shaped: AR-sum then per-rank
//! norm-then-add):
//! 1. Per-rank attention forward → partial_attn (`RowParallel<0>`).
//! 2. Barrier-fused AR-sum → partial_attn becomes `Replicated`.
//! 3. Per-rank post-AR integration (norm + residual add).
//! 4. Per-rank FFN forward → partial_ffn (`RowParallel<0>`).
//! 5. Barrier-fused AR-sum → partial_ffn becomes `Replicated`.
//! 6. Per-rank post-AR integration.
//!
//! Models with a different post-AR shape (qwen3-moe uses the fused
//! `tp_allreduce_residual` kernel that adds-then-norms in one
//! launch) keep their existing forward and don't implement this
//! trait — the composer is parameterized over the *shape*, not over
//! every possible AR variant.
//!
//! ## Borrow-shape contract
//!
//! Each per-rank hook takes `&mut self` so the impl can reach into
//! `self.stages[r]`, `&self.regs[r]`, `self.tp.cluster().device(r)`
//! via the standard disjoint-field-borrow rule. The composer holds
//! `&self` only for the AR-setup scope, dropped before the next
//! `&mut self` hook call (NLL).

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{BarP2pAllReduce, HipCluster, HipStream};
use flambeau_core::{Device, DevicePtr};

use crate::layer::tp_allreduce_sum_synced;
use crate::tensor_view::{Buffer, RowParallel, F16};
use crate::tp_rank_core::TpRankCore;

/// Model-side contract for a TP decode-layer composer driven by
/// [`forward_decode_layer_tp`]. See module docs for the canonical
/// phase order.
///
/// Per-rank hooks take `&mut self` and rebuild `HipOps` internally;
/// the composer drives the iteration + AR scaffolding.
pub trait LayerComposerTp {
    /// Number of TP ranks (= `cluster.ranks()`).
    fn n_ranks(&self) -> usize;
    /// Hidden-dim element count for `partial_*` AR.
    fn hidden_size(&self) -> usize;
    /// AR primitive (driver-side).
    fn ar(&self) -> &BarP2pAllReduce;
    /// Cluster handle (driver-side).
    fn cluster(&self) -> &HipCluster;
    /// Per-rank universal sync state.
    fn core(&self, rank: usize) -> &TpRankCore;
    /// Per-rank `partial_attn` pointer (row-parallel attn-output).
    fn partial_attn_ptr(&self, rank: usize) -> DevicePtr;
    /// Per-rank `partial_ffn` pointer (row-parallel ffn-down).
    fn partial_ffn_ptr(&self, rank: usize) -> DevicePtr;

    /// Phase 1: per-rank attention forward, writing `partial_attn`.
    fn forward_attn(&mut self, rank: usize, position: usize, il: usize) -> Result<()>;
    /// Phase 3: per-rank post-AR attn integration (norm + residual add).
    fn post_norm_residual_attn(&mut self, rank: usize, il: usize) -> Result<()>;
    /// Phase 4: per-rank FFN forward, writing `partial_ffn`.
    fn forward_ffn(&mut self, rank: usize, il: usize) -> Result<()>;
    /// Phase 6: per-rank post-AR ffn integration (norm + residual add).
    fn post_norm_residual_ffn(&mut self, rank: usize, il: usize) -> Result<()>;
}

/// One TP decode layer: 6 phases, two AR-sum transitions, model-
/// specific per-rank work at each forward + post-norm step.
pub fn forward_decode_layer_tp<L: LayerComposerTp>(
    layer: &mut L,
    position: usize,
    il: usize,
) -> Result<()> {
    let n = layer.n_ranks();

    // Phase 1.
    for r in 0..n {
        layer.forward_attn(r, position, il)?;
    }

    // Phase 2: barrier-fused AR-sum on partial_attn.
    // SAFETY: each rank's partial_attn buffer is `hidden` F16 elems;
    // streams + cluster outlive this scope; `tp_allreduce_sum_synced`
    // orders BAR1 reads behind every peer's Phase-1 producer-done
    // event.
    unsafe {
        ar_sum_partial(layer, partial_attn_ptr_fn)?;
    }

    // Phase 3.
    for r in 0..n {
        layer.post_norm_residual_attn(r, il)?;
    }

    // Phase 4.
    for r in 0..n {
        layer.forward_ffn(r, il)?;
    }

    // Phase 5: barrier-fused AR-sum on partial_ffn.
    // SAFETY: same as Phase 2.
    unsafe {
        ar_sum_partial(layer, partial_ffn_ptr_fn)?;
    }

    // Phase 6.
    for r in 0..n {
        layer.post_norm_residual_ffn(r, il)?;
    }

    Ok(())
}

// Helper: fetch partial pointer for AR. Free fns (not closures) so
// the generic plumbing reads cleanly and the `unsafe` block in
// `ar_sum_partial` has a narrow scope.
fn partial_attn_ptr_fn<L: LayerComposerTp>(layer: &L, rank: usize) -> DevicePtr {
    layer.partial_attn_ptr(rank)
}
fn partial_ffn_ptr_fn<L: LayerComposerTp>(layer: &L, rank: usize) -> DevicePtr {
    layer.partial_ffn_ptr(rank)
}

/// AR-sum one of the per-rank row-parallel partials in place.
///
/// # Safety
/// Caller asserts that the partial pointers returned by `ptr_fn`
/// point at the most-recently-written row-parallel partials and
/// won't be aliased mutably during the AR. The synced primitive
/// enforces cross-rank ordering on top.
unsafe fn ar_sum_partial<L: LayerComposerTp>(
    layer: &L,
    ptr_fn: fn(&L, usize) -> DevicePtr,
) -> Result<()> {
    let n = layer.n_ranks();
    let hidden = layer.hidden_size();
    let cluster = layer.cluster();
    let cores: Vec<&TpRankCore> = (0..n).map(|r| layer.core(r)).collect();
    let streams: Vec<&HipStream> = (0..n).map(|r| cluster.device(r).default_stream()).collect();
    let partials: Vec<Buffer<F16, RowParallel<0>>> = (0..n)
        .map(|r| {
            // SAFETY: the caller of `ar_sum_partial` carries the
            // row-parallel-partial invariant for `ptr_fn(layer, r)`.
            unsafe { Buffer::from_raw_unchecked(ptr_fn(layer, r), hidden) }
        })
        .collect();
    // SAFETY: synced helper orders BAR1 reads behind peer producer
    // events; partials/streams/cluster all outlive this call.
    let _replicated =
        unsafe { tp_allreduce_sum_synced::<0>(layer.ar(), cluster, &cores, &partials, &streams) }?;
    Ok(())
}
