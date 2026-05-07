//! TP-2 — tensor-parallel forward path scratch + driver.
//!
//! Sister of [`super::pp`]. Where PP holds whole layers per rank and
//! threads the hidden state across ranks once per layer chain, TP holds
//! every layer on every rank (sliced) and threads `partial_attn` /
//! `partial_ffn` buffers through the AllReduce kernel twice per layer.
//!
//! ## Scope this session (TP-2a)
//!
//! Only the per-rank scratch types land here:
//! - [`RankForwardScratchTp`] — `hidden_a/b` (replicated, AR-reduced) +
//!   `partial_attn_out` / `partial_ffn_out` (rank-local, AR-source).
//! - [`ShardedForwardOneTokenScratchTp`] — aggregate over the cluster.
//!
//! Forward kernel composition (TP-2b/c/d), parity cert (TP-2d), and
//! perf cert (TP-2e) follow.

use anyhow::{bail, Result};
use flambeau_backend_hip::{HipCluster, HipDevice, HipEvent};
use flambeau_core::{Device, DevicePtr};
use flambeau_runtime::RankId;

use super::io::OutputHeadScratch;
use super::layer::LayerForwardScratch;
use crate::config::Qwen3MoEConfig;

/// Per-rank scratch for tensor-parallel decode.
///
/// Layout vs PP's [`super::pp::RankForwardScratch`]:
/// - `hidden_a` / `hidden_b` — same ping-pong shape, but the contents
///   are *replicated* across ranks (AR kernel writes the same value to
///   every rank's `hidden_a`).
/// - `partial_attn_out` / `partial_ffn_out` — new. Rank-local outputs
///   of the row-parallel `attn_output` and `ffn_down` projections,
///   fed into [`flambeau_backend_hip::BarP2pAllReduce`] which folds
///   them back into `hidden_a` with residual-add.
/// - `layer` is reused as-is for now. Some sub-buffers
///   (per-head Q/K/V scratches) are larger than the per-rank slice
///   strictly needs; TP-2b may introduce a TP-aware sizing variant if
///   the over-allocation matters.
/// - `output_head` lives only on the last rank in the PP path; for TP
///   every rank can carry one (the LM head is replicated in V1) but
///   we still gate to a single rank to avoid 4× wasted scratch.
pub struct RankForwardScratchTp {
    pub rank: RankId,
    pub device_id: i32,
    /// Replicated hidden state. AR-reduced after each layer step.
    pub hidden_a: DevicePtr,
    /// Ping-pong companion. Used by intra-layer ops that need a
    /// non-aliasing destination (rmsnorm, residual-add).
    pub hidden_b: DevicePtr,
    /// Rank-local attention output partial. Sized `hidden × max_batch`
    /// because the row-parallel `attn_output` matmul emits a full-`H`
    /// vector that's only this rank's contribution to the global sum.
    /// AllReduce-residual on this buffer adds it (and 3 peers') into
    /// `hidden_a`.
    pub partial_attn_out: DevicePtr,
    /// Rank-local FFN output partial. Same shape + role as above.
    pub partial_ffn_out: DevicePtr,
    /// Per-layer ops scratch. For now reuses the PP-shaped allocation;
    /// TP-2b may shrink head-sized slabs to per-rank head count.
    pub layer: Option<LayerForwardScratch>,
    /// LM head scratch — populated only on the rank that owns the
    /// argmax. V1 keeps `output.weight` Replicated (TP-1a layout
    /// table), so a single designated rank runs the LM head; the
    /// others have `None` here.
    pub output_head: Option<OutputHeadScratch>,
    /// **TP-3a** — event recorded on the producer stream after a
    /// partial-write kernel (last op of `forward_full_attn_decode_tp`,
    /// `forward_dense_ffn_decode_tp`, or `forward_gdn_decode_tp`).
    /// Peer ranks' AR streams `stream_wait` on this event before
    /// launching the AR kernel that reads this rank's partial buffer.
    /// Replaces the host `Stream::synchronize` in TP-2d's `ar_residual`
    /// with a driver-side DAG edge — host doesn't block.
    pub producer_done_event: HipEvent,
    hidden_bytes: usize,
    partial_bytes: usize,
    disposed: bool,
}

impl RankForwardScratchTp {
    /// Dispose every device allocation owned by this rank's scratch.
    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        // SAFETY: every pointer came from the matching `device.alloc()`
        // in `ShardedForwardOneTokenScratchTp::new`. No aliasing.
        unsafe {
            device.dealloc(self.hidden_a, self.hidden_bytes)?;
            device.dealloc(self.hidden_b, self.hidden_bytes)?;
            device.dealloc(self.partial_attn_out, self.partial_bytes)?;
            device.dealloc(self.partial_ffn_out, self.partial_bytes)?;
        }
        if let Some(s) = self.layer.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.output_head.take() {
            s.dispose(device)?;
        }
        Ok(())
    }

    /// Bytes allocated by this rank's scratch. Diagnostic for the
    /// TP-2a smoke test — should equal `2·hidden + 2·hidden + layer +
    /// optional output_head` per rank.
    pub fn allocated_bytes(&self) -> usize {
        2 * self.hidden_bytes + 2 * self.partial_bytes
    }
}

impl Drop for RankForwardScratchTp {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                rank = self.rank.0,
                "RankForwardScratchTp dropped without dispose(device); buffers leaked"
            );
        }
    }
}

/// Aggregate per-rank scratch for a TP-sharded decode.
///
/// Designed-rank for the LM head defaults to rank 0 (V1 LM head is
/// Replicated; any rank could run it but rank 0 already holds
/// `token_embd` for the embed gather, so reusing the same rank
/// minimises the rank that the runtime needs to "wake up" first).
pub struct ShardedForwardOneTokenScratchTp {
    pub per_rank: Vec<RankForwardScratchTp>,
    /// Rank that runs the LM head + argmax. V1: rank 0.
    pub head_rank: RankId,
}

impl ShardedForwardOneTokenScratchTp {
    /// Allocate scratch for every rank in `cluster`. `cfg.hidden_size`
    /// drives the buffer sizing; the per-rank `partial_*` buffers are
    /// the same shape as `hidden` (they hold a *full*-H partial of one
    /// token's contribution to the AllReduce sum).
    ///
    /// `head_rank` defaults to 0; pass an explicit override only when
    /// the caller has a topology-specific reason (load-balancing
    /// across decode + prefill, etc. — V2-tp-5 territory).
    pub fn new(cfg: &Qwen3MoEConfig, cluster: &HipCluster) -> Result<Self> {
        Self::new_with_head_rank(cfg, cluster, RankId(0))
    }

    /// Like [`Self::new`] but lets the caller pick which rank owns the
    /// LM head scratch.
    pub fn new_with_head_rank(
        cfg: &Qwen3MoEConfig,
        cluster: &HipCluster,
        head_rank: RankId,
    ) -> Result<Self> {
        let hidden_bytes = cfg.hidden_size * 2; // F16
        // V1 max_batch=1 (single-token decode); the partial buffer is a
        // full-H F16 vector. Multi-token batched AR is V2 territory.
        let partial_bytes = cfg.hidden_size * 2;

        if (head_rank.0 as usize) >= cluster.ranks() {
            anyhow::bail!(
                "head_rank {} out of range for cluster ranks {}",
                head_rank.0,
                cluster.ranks()
            );
        }

        let mut per_rank = Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let hidden_a = device.alloc(hidden_bytes)?;
            let hidden_b = device.alloc(hidden_bytes)?;
            let partial_attn_out = device.alloc(partial_bytes)?;
            let partial_ffn_out = device.alloc(partial_bytes)?;
            let layer = Some(LayerForwardScratch::new(cfg, device)?);
            let output_head = if rank_idx as u32 == head_rank.0 {
                Some(OutputHeadScratch::new(cfg, device)?)
            } else {
                None
            };
            let producer_done_event = HipEvent::new(device.id())?;
            per_rank.push(RankForwardScratchTp {
                rank: RankId(rank_idx as u32),
                device_id: device.id(),
                hidden_a,
                hidden_b,
                partial_attn_out,
                partial_ffn_out,
                layer,
                output_head,
                producer_done_event,
                hidden_bytes,
                partial_bytes,
                disposed: false,
            });
        }

        Ok(Self { per_rank, head_rank })
    }

    /// Free every rank's scratch.
    pub fn dispose(mut self, cluster: &HipCluster) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for rs in self.per_rank.drain(..) {
            let rank_idx = rs.rank.0 as usize;
            if let Err(e) = rs.dispose(cluster.device(rank_idx)) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }

    /// Total bytes allocated across all ranks (excluding `LayerForwardScratch`
    /// internals — those are owned by the layer scratch and aggregated
    /// separately). Sanity check for TP-2a's invariant.
    pub fn rank_level_bytes(&self) -> usize {
        self.per_rank.iter().map(|r| r.allocated_bytes()).sum()
    }
}

// =====================================================================
// AUTO-6b1 — ShardedForwardPrefillScratchTp
// =====================================================================

/// Per-rank scratch for an L-batched TP prefill. Mirrors
/// [`RankForwardScratchTp`] but every `[hidden]` buffer is grown to
/// `[max_tokens, hidden]` so a single forward sweep through the layer
/// chain handles all `L` prompt tokens at once. The per-layer
/// kernel-level scratch is the same [`super::layer::LayerPrefillScratch`]
/// the PP path already uses (it is L-aware by design).
///
/// Lifecycle: allocate once per request via
/// [`ShardedForwardPrefillScratchTp::new`], dispose once via
/// [`ShardedForwardPrefillScratchTp::dispose`]. The decode path
/// continues to use the cheaper [`ShardedForwardOneTokenScratchTp`].
pub struct RankForwardPrefillScratchTp {
    pub rank: RankId,
    pub device_id: i32,
    pub max_tokens: usize,
    /// F16 `[max_tokens, hidden]` — replicated residual stream (post-AR).
    pub hidden_a: DevicePtr,
    /// F16 `[max_tokens, hidden]` — ping-pong partner for `hidden_a`.
    pub hidden_b: DevicePtr,
    /// F16 `[max_tokens, hidden]` — pre-AR per-rank attn partial.
    pub partial_attn_out: DevicePtr,
    /// F16 `[max_tokens, hidden]` — pre-AR per-rank FFN partial.
    pub partial_ffn_out: DevicePtr,
    pub layer: Option<super::layer::LayerPrefillScratch>,
    pub output_head: Option<super::io::OutputHeadScratch>,
    /// **P2.9b-i2-C-wire** — single shared `GdnScratch` (single-token
    /// decode workspace) for the batched-decode driver's per-slot GDN
    /// loop on this rank. None on archs without GDN. Sibling of the
    /// PP `RankForwardPrefillScratch.gdn_decode`.
    pub gdn_decode: Option<super::gdn::GdnScratch>,
    /// **#285 batched-GDN** — n_tokens=N workspace for
    /// `forward_gdn_decode_batched_tp`. Sized for `max_tokens` (which
    /// for the batched-decode driver is `INFLIGHT_SLOTS`). None on
    /// archs without GDN.
    pub gdn_decode_batched: Option<super::gdn::GdnPrefillScratch>,
    /// **#290 PP-pipelined decode** — bridge events for the cross-stage
    /// `peer_copy_via_host_async` in `forward_decode_pipelined_hybrid`.
    /// One event per slot (max_tokens), per dst-rank in the next stage's
    /// sub_cluster. Allocated lazily in the pipelined driver and reused
    /// across decode steps; `hipEventRecord` overwrites the prior record.
    /// Only used when this rank lives in a non-final pipeline stage
    /// (i.e., the rank's stage_idx + 1 < n_stages); inner Vec stays
    /// empty otherwise.
    pub pipeline_bridge_events: Vec<flambeau_backend_hip::HipEvent>,
    hidden_bytes: usize,
    partial_bytes: usize,
    disposed: bool,
}

impl RankForwardPrefillScratchTp {
    /// Free this rank's allocations against its owning device. The
    /// caller threads the device handle in (mirrors
    /// [`RankForwardScratchTp::dispose`]).
    pub fn dispose(mut self, device: &flambeau_backend_hip::HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        // SAFETY: every pointer came from `device.alloc` in `new_with_head_rank`.
        unsafe {
            device.dealloc(self.hidden_a, self.hidden_bytes)?;
            device.dealloc(self.hidden_b, self.hidden_bytes)?;
            device.dealloc(self.partial_attn_out, self.partial_bytes)?;
            device.dealloc(self.partial_ffn_out, self.partial_bytes)?;
        }
        if let Some(s) = self.layer.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.output_head.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.gdn_decode.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.gdn_decode_batched.take() {
            s.dispose(device)?;
        }
        Ok(())
    }

    /// Bytes allocated by this rank's scratch (excluding
    /// `LayerPrefillScratch` internals — those are owned by the layer
    /// scratch). Diagnostic for the AUTO-6b1 smoke test.
    pub fn allocated_bytes(&self) -> usize {
        2 * self.hidden_bytes + 2 * self.partial_bytes
    }
}

impl Drop for RankForwardPrefillScratchTp {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                rank = self.rank.0,
                "RankForwardPrefillScratchTp dropped without dispose(device); buffers leaked"
            );
        }
    }
}

/// Aggregate per-rank scratch for an L-batched TP prefill. Counterpart
/// to [`ShardedForwardOneTokenScratchTp`] for the prefill path. The
/// AUTO-6b2 / AUTO-6b3 batched kernel composites consume this; the
/// AUTO-6a per-token loop continues to use the decode scratch.
pub struct ShardedForwardPrefillScratchTp {
    pub per_rank: Vec<RankForwardPrefillScratchTp>,
    /// Rank that runs the LM head. V1 default: rank 0 (matches
    /// decode-time convention).
    pub head_rank: RankId,
    pub max_tokens: usize,
}

impl ShardedForwardPrefillScratchTp {
    /// Allocate per-rank prefill scratch sized for `max_tokens`. The
    /// LM head is held only on `head_rank` (default rank 0) — same
    /// convention as the decode scratch.
    pub fn new(
        cfg: &Qwen3MoEConfig,
        cluster: &HipCluster,
        max_tokens: usize,
    ) -> Result<Self> {
        Self::new_with_head_rank(cfg, cluster, max_tokens, RankId(0))
    }

    /// Like [`Self::new`] but lets the caller pick which rank owns the
    /// LM head scratch.
    pub fn new_with_head_rank(
        cfg: &Qwen3MoEConfig,
        cluster: &HipCluster,
        max_tokens: usize,
        head_rank: RankId,
    ) -> Result<Self> {
        if max_tokens == 0 {
            bail!("ShardedForwardPrefillScratchTp::new max_tokens must be >= 1");
        }
        if (head_rank.0 as usize) >= cluster.ranks() {
            bail!(
                "head_rank {} out of range for cluster ranks {}",
                head_rank.0,
                cluster.ranks()
            );
        }
        let hidden_bytes = max_tokens * cfg.hidden_size * 2;
        let partial_bytes = max_tokens * cfg.hidden_size * 2;

        let mut per_rank = Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let hidden_a = device.alloc(hidden_bytes)?;
            let hidden_b = device.alloc(hidden_bytes)?;
            let partial_attn_out = device.alloc(partial_bytes)?;
            let partial_ffn_out = device.alloc(partial_bytes)?;
            let layer = Some(super::layer::LayerPrefillScratch::new(cfg, device, max_tokens)?);
            let output_head = if rank_idx as u32 == head_rank.0 {
                Some(super::io::OutputHeadScratch::new(cfg, device)?)
            } else {
                None
            };
            // **P2.9b-i2-C-wire** — shared single-token GDN decode scratch
            // for the batched-decode driver's per-slot GDN loop. Allocated
            // only when arch has GDN.
            let gdn_decode = if cfg.gdn.is_some() {
                Some(super::gdn::GdnScratch::new(cfg, device)?)
            } else {
                None
            };
            // **#285 batched-GDN** — n_tokens=N workspace sized for
            // max_tokens (= INFLIGHT_SLOTS in the batched-decode caller).
            let gdn_decode_batched = if cfg.gdn.is_some() {
                Some(super::gdn::GdnPrefillScratch::new(cfg, device, max_tokens)?)
            } else {
                None
            };
            per_rank.push(RankForwardPrefillScratchTp {
                rank: RankId(rank_idx as u32),
                device_id: device.id(),
                max_tokens,
                hidden_a,
                hidden_b,
                partial_attn_out,
                partial_ffn_out,
                layer,
                output_head,
                gdn_decode,
                gdn_decode_batched,
                pipeline_bridge_events: Vec::new(),
                hidden_bytes,
                partial_bytes,
                disposed: false,
            });
        }
        Ok(Self {
            per_rank,
            head_rank,
            max_tokens,
        })
    }

    /// Free every rank's scratch.
    pub fn dispose(mut self, cluster: &HipCluster) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for rs in self.per_rank.drain(..) {
            let rank_idx = rs.rank.0 as usize;
            if let Err(e) = rs.dispose(cluster.device(rank_idx)) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }

    /// Total bytes allocated across all ranks (rank-level only —
    /// excludes `LayerPrefillScratch` internals which the layer
    /// scratch owns separately).
    pub fn rank_level_bytes(&self) -> usize {
        self.per_rank.iter().map(|r| r.allocated_bytes()).sum()
    }
}

// =====================================================================
// TP-2d — forward_one_token_tp end-to-end driver
// =====================================================================

use anyhow::{anyhow, Context};
use flambeau_backend_hip::BarP2pAllReduce;
use flambeau_ops::hip::norm::rmsnorm_f16;

use super::attn_tp::forward_full_attn_decode_tp;
use super::io::{argmax_token_host, forward_embed_decode_host, forward_output_head_decode};
use crate::session::LayerCache;
use crate::tp_sharded::{Qwen3MoETpModel, TpLayerTensor};
use crate::weights::DeviceTensor;

/// FLAMBEAU_TP_PROBE — rank-aware F16 probe at an arbitrary device pointer.
/// Used for inspecting per-rank partial buffers.
fn debug_probe_named_rank(
    scratch: &ShardedForwardOneTokenScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    label: &str,
    il: usize,
    ptr: DevicePtr,
    rank: usize,
) -> anyhow::Result<()> {
    use flambeau_core::CopyDirection;
    let device = cluster.device(rank);
    device.bind()?;
    let stream = device.default_stream();
    let n_bytes = scratch.per_rank[rank].hidden_bytes;
    let n = n_bytes / 2;
    let mut host = vec![0u16; n];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            ptr,
            n_bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream)?;
    let mut nan = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &b in &host {
        let v = half::f16::from_bits(b).to_f32();
        if v.is_nan() {
            nan += 1;
        } else {
            if v < min { min = v; }
            if v > max { max = v; }
            sum += v as f64;
        }
    }
    let mean = sum / (n - nan).max(1) as f64;
    eprintln!(
        "  PROBE {label} rank={rank} il={il:>3}  n={n}  nan={nan:>5}  min={min:.6}  max={max:.6}  mean={mean:.6}"
    );
    Ok(())
}

/// FLAMBEAU_TP_PROBE — download `n_bytes` from `ptr` on rank 0, log
/// min/max/nan/mean. Pointer-explicit variant for probing scratch
/// buffers other than `hidden_a`.
fn debug_probe_rank0_named(
    scratch: &ShardedForwardOneTokenScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    label: &str,
    il: usize,
    ptr: DevicePtr,
) -> anyhow::Result<()> {
    use flambeau_core::CopyDirection;
    let device = cluster.device(0);
    device.bind()?;
    let stream = device.default_stream();
    let n_bytes = scratch.per_rank[0].hidden_bytes;
    let n = n_bytes / 2;
    let mut host = vec![0u16; n];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            ptr,
            n_bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream)?;
    let mut nan = 0usize;
    let mut zero = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &b in &host {
        let v = half::f16::from_bits(b).to_f32();
        if v.is_nan() {
            nan += 1;
        } else {
            if v == 0.0 { zero += 1; }
            if v < min { min = v; }
            if v > max { max = v; }
            sum += v as f64;
        }
    }
    let mean = sum / (n - nan).max(1) as f64;
    eprintln!(
        "  PROBE {label} il={il:>3}  n={n}  nan={nan:>5}  zero={zero:>5}  min={min:.4}  max={max:.4}  mean={mean:.4}"
    );
    Ok(())
}

/// FLAMBEAU_TP_PROBE — download rank-r hidden_a, log min/max/nan.
fn debug_probe_rank_hidden(
    scratch: &ShardedForwardOneTokenScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    label: &str,
    il: usize,
    rank: usize,
) -> anyhow::Result<()> {
    use flambeau_core::CopyDirection;
    let device = cluster.device(rank);
    device.bind()?;
    let stream = device.default_stream();
    let n_bytes = scratch.per_rank[rank].hidden_bytes;
    let n = n_bytes / 2;
    let mut host = vec![0u16; n];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            scratch.per_rank[rank].hidden_a,
            n_bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream)?;
    let mut nan = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &b in &host {
        let v = half::f16::from_bits(b).to_f32();
        if v.is_nan() {
            nan += 1;
        } else {
            if v < min { min = v; }
            if v > max { max = v; }
            sum += v as f64;
            sumsq += (v as f64) * (v as f64);
        }
    }
    let mean = sum / (n - nan).max(1) as f64;
    let l2 = sumsq.sqrt();
    let head: Vec<f32> = host[..host.len().min(4)]
        .iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect();
    let il_str = if il == usize::MAX { "-".to_string() } else { il.to_string() };
    eprintln!(
        "  PROBE {label} rank={rank} il={il_str:>3}  n={n}  nan={nan}  L2={l2:.6}  min={min:.6}  max={max:.6}  mean={mean:.6}  head={head:?}"
    );
    Ok(())
}

/// FLAMBEAU_TP_PROBE — download rank-0 hidden_a, log min/max/nan.
fn debug_probe_rank0_hidden(
    scratch: &ShardedForwardOneTokenScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    label: &str,
    il: usize,
) -> anyhow::Result<()> {
    use flambeau_core::CopyDirection;
    let device = cluster.device(0);
    device.bind()?;
    let stream = device.default_stream();
    let n_bytes = scratch.per_rank[0].hidden_bytes;
    let n = n_bytes / 2;
    let mut host = vec![0u16; n];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            scratch.per_rank[0].hidden_a,
            n_bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream)?;
    let mut nan = 0usize;
    let mut zero = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &b in &host {
        let v = half::f16::from_bits(b).to_f32();
        if v.is_nan() {
            nan += 1;
        } else {
            if v == 0.0 { zero += 1; }
            if v < min { min = v; }
            if v > max { max = v; }
            sum += v as f64;
        }
    }
    let mean = sum / (n - nan).max(1) as f64;
    let il_str = if il == usize::MAX { "-".to_string() } else { il.to_string() };
    eprintln!(
        "  PROBE {label} il={il_str:>3}  n={n}  nan={nan:>5}  zero={zero:>5}  min={min:.4}  max={max:.4}  mean={mean:.4}"
    );
    Ok(())
}

/// V1 TP forward driver — single-token decode through a TP-sharded model.
///
/// ## Layer dispatch
///
/// - **Full-attn layers** route through [`forward_full_attn_decode_tp`]
///   + [`forward_dense_ffn_decode_tp`] with two AllReduce launches per
///   layer (post-attn and post-FFN).
/// - **GDN layers** error out with a `TP-4a required` diagnostic. The
///   `attn_qkv` / `ssm_conv1d` Replicated layout TP-1a installed lets
///   the model *load* on TP, but the head-aware sharding for forward
///   correctness lives in TP-4a.
///
/// ## AllReduce path
///
/// `tp_world == 1`: degenerate. AR is skipped (one rank, no peers). The
/// per-rank partial buffers are bit-identical to PP single-device
/// outputs at this world size — useful for parity validation.
///
/// `tp_world == 2`: `BarP2pAllReduce::residual_tp2`.
///
/// `tp_world == 4`: `BarP2pAllReduce::residual_tp4`.
///
/// Other world sizes are rejected; TP-3 may add tp8/tp3 variants.
///
/// ## Returns
///
/// `(next_token_id, logits_optional)`. `logits` is `None` in the V1
/// greedy-only path; sampler integration is V2.
pub fn forward_one_token_tp(
    model: &Qwen3MoETpModel,
    scratch: &mut ShardedForwardOneTokenScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    ar: &BarP2pAllReduce,
    layer_caches: &mut [Vec<LayerCache>],
    token_id: u32,
    position: usize,
) -> anyhow::Result<u32> {
    forward_one_token_tp_inner(
        model, scratch, cluster, ar, layer_caches, token_id, position,
        LogitsSink::HostArgmax,
    )
}

/// **TP-5a-i2** — variant of [`forward_one_token_tp`] that downloads
/// the F32 logits row into a caller-owned `Vec<f32>` instead of
/// running argmax host-side. Used by the HTTP server when
/// `temperature > 0` / `top_p < 1` (matches PP's
/// `forward_one_token_pp_logits` shape).
pub fn forward_one_token_tp_logits(
    model: &Qwen3MoETpModel,
    scratch: &mut ShardedForwardOneTokenScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    ar: &BarP2pAllReduce,
    layer_caches: &mut [Vec<LayerCache>],
    token_id: u32,
    position: usize,
    logits_out: &mut Vec<f32>,
) -> anyhow::Result<()> {
    forward_one_token_tp_inner(
        model, scratch, cluster, ar, layer_caches, token_id, position,
        LogitsSink::HostLogits(logits_out),
    )
    .map(|_| ())
}

/// **Sampler-D3 Phase B (#211)** — runs the same forward as
/// [`forward_one_token_tp_logits`] but does NOT DtoH the `[vocab]` F32
/// logits row. After return, the head rank's
/// `scratch.per_rank[head].output_head.logits_f32` holds valid F32
/// logits for one token; the caller must consume it (e.g.
/// `topk_softmax_f32`) before the next forward call clobbers it.
///
/// Saves the 600 KB DtoH per token — visible in the chat decode hot
/// path under FLAMBEAU_GPU_SAMPLER=1.
pub fn forward_one_token_tp_keep_logits_on_device(
    model: &Qwen3MoETpModel,
    scratch: &mut ShardedForwardOneTokenScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    ar: &BarP2pAllReduce,
    layer_caches: &mut [Vec<LayerCache>],
    token_id: u32,
    position: usize,
) -> anyhow::Result<()> {
    forward_one_token_tp_inner(
        model, scratch, cluster, ar, layer_caches, token_id, position,
        LogitsSink::KeepOnDevice,
    )
    .map(|_| ())
}

/// **AUTO-6a** — ingest a `prompt_ids` prompt and write the **last**
/// position's F32 logits row into `logits_out`. Mirrors the shape of
/// [`super::pp::forward_prefill_pp_logits`] and
/// [`super::hybrid::forward_prefill_hybrid_logits`] so the server's
/// `prefill_logits` can dispatch through one symmetric entry point per
/// topology.
///
/// Implementation note: today this is a per-token loop over
/// [`forward_one_token_tp_logits`] — same behavior as the inline loop
/// the server used pre-AUTO-6a. AUTO-6b/c swap in batched-across-L
/// kernels behind the same call site (full-attn + dense FFN first,
/// then GDN + MoE). `start_position` is the position the *first*
/// prompt token lands at — non-zero when this prefill is appending to
/// a session that already saw earlier tokens.
/// **#324** — pooled-scratch variant of [`forward_prefill_tp_logits`].
///
/// Uses the caller-supplied `pool_prefill` scratch instead of
/// allocating one inside the batched driver. Caller must size
/// `pool_prefill` for at least `prompt_ids.len()` tokens (typically
/// the per-slot scratch is sized for `FLAMBEAU_PREFILL_UBATCH`,
/// matching the chunk size used in `prefill_logits`).
///
/// Falls through to the per-token loop when the prompt is short
/// (<8 tokens) or KV is Q8 — the pooled scratch is unused on those
/// paths since they go through `forward_one_token_tp_logits`.
pub fn forward_prefill_tp_logits_pooled(
    model: &Qwen3MoETpModel,
    scratch: &mut ShardedForwardOneTokenScratchTp,
    pool_prefill: &mut ShardedForwardPrefillScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    ar: &BarP2pAllReduce,
    layer_caches: &mut [Vec<LayerCache>],
    prompt_ids: &[u32],
    start_position: usize,
    logits_out: &mut Vec<f32>,
) -> anyhow::Result<()> {
    if prompt_ids.is_empty() {
        bail!("forward_prefill_tp_logits_pooled: empty prompt");
    }
    let any_q8_kv = layer_caches.iter().any(|cs| {
        cs.iter().any(|c| matches!(c, LayerCache::FullAttnQ8(_)))
    });
    let batched_opt_out = std::env::var("FLAMBEAU_TP_BATCHED").as_deref() == Ok("0");
    let use_batched = !any_q8_kv && !batched_opt_out;
    if use_batched && prompt_ids.len() >= 8 {
        return forward_prefill_tp_batched_logits(
            model,
            cluster,
            ar,
            layer_caches,
            prompt_ids,
            start_position,
            logits_out,
            Some(pool_prefill),
        )
        .context("TP batched prefill pooled (#324)");
    }
    for (i, &tok) in prompt_ids.iter().enumerate() {
        let pos = start_position + i;
        forward_one_token_tp_logits(
            model, scratch, cluster, ar, layer_caches, tok, pos, logits_out,
        )
        .with_context(|| format!("TP prefill loop @ pos {pos} (pooled)"))?;
    }
    Ok(())
}

pub fn forward_prefill_tp_logits(
    model: &Qwen3MoETpModel,
    scratch: &mut ShardedForwardOneTokenScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    ar: &BarP2pAllReduce,
    layer_caches: &mut [Vec<LayerCache>],
    prompt_ids: &[u32],
    start_position: usize,
    logits_out: &mut Vec<f32>,
) -> anyhow::Result<()> {
    if prompt_ids.is_empty() {
        bail!("forward_prefill_tp_logits: empty prompt");
    }
    // **AUTO-6b2 / AUTO-6c4** — L-batched prefill is the default for
    // TP/hybrid topologies on prompts ≥ 8 tokens. Set
    // `FLAMBEAU_TP_BATCHED=0` to opt out and fall through to the
    // per-token loop (kept for diagnostics + Q8 KV which has no batched
    // attention kernel).
    //
    // Default-on rationale: bench cert
    // `coder_next_80b_cn80b_12_topology_summary.md` measures pp512
    // 801 tok/s (batched) vs the per-token loop at ~36 tok/s — a 22×
    // gap that left live serving running at ~5% of kernel ceiling.
    //
    // The driver handles all four layer flavors:
    //   * full-attn + dense FFN  (Qwen3.5 9B/27B Q4_1, dense)
    //   * GDN + dense FFN        (Qwen3.5 hybrid)
    //   * full-attn + MoE        (Qwen3-Coder-30B)
    //   * GDN + MoE + shared exp (Qwen3.6-35B-A3B, Coder-Next-80B)
    //
    // V1-BENCH-#116 — Q8_0 KV has no batched-prefill kernel
    // (attention_prefill_q8_kv doesn't exist yet). Detect and fall
    // through to the per-token loop; each iter goes through
    // forward_one_token_tp_logits → forward_full_attn_decode_tp<L>
    // which dispatches the right Q8 attention kernel.
    let any_q8_kv = layer_caches.iter().any(|cs| {
        cs.iter().any(|c| matches!(c, LayerCache::FullAttnQ8(_)))
    });
    let batched_opt_out = std::env::var("FLAMBEAU_TP_BATCHED").as_deref() == Ok("0");
    let use_batched = !any_q8_kv && !batched_opt_out;
    if use_batched && prompt_ids.len() >= 8 {
        return forward_prefill_tp_batched_logits(
            model,
            cluster,
            ar,
            layer_caches,
            prompt_ids,
            start_position,
            logits_out,
            None,
        )
        .context("TP batched prefill (AUTO-6b2 / AUTO-6c4)");
    }
    for (i, &tok) in prompt_ids.iter().enumerate() {
        let pos = start_position + i;
        forward_one_token_tp_logits(
            model, scratch, cluster, ar, layer_caches, tok, pos, logits_out,
        )
        .with_context(|| format!("TP prefill loop @ pos {pos}"))?;
    }
    Ok(())
}

/// **AUTO-6b2** — L-batched TP prefill driver for dense (qwen35)
/// arches. Allocates a [`ShardedForwardPrefillScratchTp`] sized to
/// `prompt_ids.len()` on the fly, runs each layer's full-attn +
/// dense-FFN with one AR per side per layer (instead of per-token),
/// then runs the LM head on the last position. Per-token decode
/// continues via `forward_one_token_tp_logits`.
///
/// V2.x deferred: bound the prefill scratch on the inflight session
/// so we don't pay alloc/dispose per request — for AUTO-6b2 this
/// keeps the call-site change minimal.
fn forward_prefill_tp_batched_logits(
    model: &Qwen3MoETpModel,
    cluster: &flambeau_backend_hip::HipCluster,
    ar: &BarP2pAllReduce,
    layer_caches: &mut [Vec<LayerCache>],
    prompt_ids: &[u32],
    start_position: usize,
    logits_out: &mut Vec<f32>,
    pooled: Option<&mut ShardedForwardPrefillScratchTp>,
) -> Result<()> {
    use flambeau_core::Stream;

    let cfg = &model.config;
    let world = cluster.ranks() as u32;
    if world != 1 && world != 2 && world != 4 {
        bail!("TP batched prefill: world ∈ {{1, 2, 4}} (got {world})");
    }
    let n_tokens = prompt_ids.len();

    // **#324** — caller-provided pooled scratch path. When `pooled` is
    // `Some`, we reuse the inflight session's pre-allocated scratch
    // (sized for `prefill_ubatch`), skipping the ~35 MB alloc/dispose
    // each call. The scratch must be sized for at least `n_tokens`;
    // we bail loudly if the caller violated that invariant.
    //
    // When `pooled` is `None`, we fall back to the legacy per-call
    // alloc with a RAII dispose guard — kept for tests and any future
    // caller that hasn't wired pooling.

    // RAII guard so the scratch is disposed on every exit path
    // (only used in the alloc-per-call branch).
    struct PrefillGuard<'c> {
        scratch: Option<ShardedForwardPrefillScratchTp>,
        cluster: &'c flambeau_backend_hip::HipCluster,
    }
    impl Drop for PrefillGuard<'_> {
        fn drop(&mut self) {
            if let Some(s) = self.scratch.take() {
                let _ = s.dispose(self.cluster);
            }
        }
    }

    let mut owned_guard: Option<PrefillGuard<'_>> = None;
    let scratch_ref: &mut ShardedForwardPrefillScratchTp = match pooled {
        Some(s) => {
            // Caller-provided scratch must fit n_tokens.
            if s.per_rank[0].max_tokens < n_tokens {
                bail!(
                    "pooled TP prefill scratch sized {} < n_tokens {}",
                    s.per_rank[0].max_tokens,
                    n_tokens
                );
            }
            s
        }
        None => {
            let prefill = ShardedForwardPrefillScratchTp::new(cfg, cluster, n_tokens)
                .context("alloc TP prefill scratch")?;
            owned_guard = Some(PrefillGuard {
                scratch: Some(prefill),
                cluster,
            });
            owned_guard
                .as_mut()
                .and_then(|g| g.scratch.as_mut())
                .expect("scratch present until drop")
        }
    };

    // 2. Embed all L tokens on every rank. Token_embd is Replicated
    //    so each rank writes the same F16 [L, hidden] into its own
    //    `hidden_a` (loop the per-row embed helper — same as the
    //    decode path, just over L positions).
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;
    for r in 0..cluster.ranks() {
        let device = cluster.device(r);
        device.bind()?;
        let stream = device.default_stream();
        let dst_base = scratch_ref.per_rank[r].hidden_a;
        for (i, &tok) in prompt_ids.iter().enumerate() {
            let dst_row = DevicePtr(dst_base.as_usize() + i * row_bytes);
            forward_embed_decode_host(
                device,
                stream,
                &model.shards[r].token_embd,
                tok,
                dst_row,
                hidden,
            )
            .with_context(|| format!("rank {r} prefill embed pos {i}"))?;
        }
    }

    // 3. Layer loop. Pure-TP: full range, no cache offset.
    //    (Body extracted to forward_prefill_tp_batched_layers for
    //    AUTO-6e1 — hybrid callers pass a stage-local layer range +
    //    il_cache_offset = range.start.)
    forward_prefill_tp_batched_layers(
        model,
        scratch_ref,
        cluster,
        ar,
        layer_caches,
        0..cfg.num_layers,
        0,
        n_tokens,
        start_position,
    )
    .context("TP batched prefill layer loop")?;

    // 4. Output head + logits download on head_rank, last position.
    let head_rank = scratch_ref.head_rank.0 as usize;
    let device = cluster.device(head_rank);
    device.bind()?;
    let stream = device.default_stream();
    let head_shard = &model.shards[head_rank];
    let lm_head = head_shard.output.as_ref().unwrap_or(&head_shard.token_embd);
    let last_row_off = (n_tokens - 1) * row_bytes;
    let hidden_a_last = DevicePtr(scratch_ref.per_rank[head_rank].hidden_a.as_usize() + last_row_off);
    let head_scratch = scratch_ref.per_rank[head_rank]
        .output_head
        .as_mut()
        .ok_or_else(|| anyhow!("head_rank={head_rank} missing OutputHeadScratch"))?;
    let logits_f32 = head_scratch.logits_f32;
    let ops = &model.ops[head_rank];
    forward_output_head_decode(
        ops,
        stream,
        cfg,
        &head_shard.output_norm,
        lm_head,
        head_scratch,
        hidden_a_last,
    )
    .context("output_head_decode (TP batched prefill)")?;
    logits_out.clear();
    logits_out.resize(cfg.vocab_size, 0.0f32);
    // SAFETY: logits_f32 is valid for cfg.vocab_size F32 values on
    // `device`; logits_out.as_mut_ptr() is host memory of matching size.
    unsafe {
        <flambeau_backend_hip::HipDevice as flambeau_core::Device>::memcpy_async(
            device,
            stream,
            flambeau_core::CopyDirection::DeviceToHost,
            DevicePtr(logits_out.as_mut_ptr() as usize),
            logits_f32,
            cfg.vocab_size * 4,
        )?;
    }
    Stream::synchronize(stream)?;
    Ok(())
}

/// **AUTO-6e1** — layer-range-aware body of the L-batched TP prefill.
/// The same per-layer attn → AR → FFN-norm → FFN → AR chain that
/// [`forward_prefill_tp_batched_logits`] runs over `0..n_layers`,
/// extracted so hybrid (PP-of-TP) callers can run only a stage's
/// slice of layers between inter-stage hand-offs.
///
/// Contract:
///  * `hidden_a` on every rank is the residual stream — read at the
///    start of each layer, written at the end. Caller is responsible
///    for embedding the prompt into rank-0..N's `hidden_a` before
///    `il_range.start`, and for consuming the post-final-layer
///    `hidden_a` (output head + last-position logits, or hand-off).
///  * `il_range` enumerates absolute layer indices into
///    `model.shards[r].layers[il]`.
///  * `il_cache_offset` is subtracted from `il` to index into
///    `layer_caches[r]`. Pure-TP callers pass `0` (caches sized to
///    `cfg.num_layers`); hybrid stages pass `stage.layer_range.start`
///    (each stage's caches were allocated for its slice only).
///  * `start_position` is the position the *first* prompt token
///    lands at — propagated to the full-attn prefill so KV slots are
///    written at the right offset.
#[expect(
    clippy::too_many_arguments,
    reason = "matches forward_prefill_tp_batched_logits arg shape; the alternative \
              would be a context struct that just rewraps these pointers"
)]
pub fn forward_prefill_tp_batched_layers(
    model: &Qwen3MoETpModel,
    scratch_ref: &mut ShardedForwardPrefillScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    ar: &BarP2pAllReduce,
    layer_caches: &mut [Vec<LayerCache>],
    il_range: std::ops::Range<usize>,
    il_cache_offset: usize,
    n_tokens: usize,
    start_position: usize,
) -> Result<()> {
    let cfg = &model.config;
    let world = cluster.ranks() as u32;
    if world != 1 && world != 2 && world != 4 {
        bail!("TP batched prefill: world ∈ {{1, 2, 4}} (got {world})");
    }
    let kv_replicated = model.tp.kv_replicated();
    let hidden = cfg.hidden_size;
    let elem_count_l = (n_tokens * hidden) as u32;
    // V1-BENCH-CN-80B-11a — mark on rank 0's stream at each section
    // boundary. The AR (`ar_residual_prefill`) syncs all ranks before
    // launching, so marks taken AFTER an AR have a clean barrier; marks
    // taken BEFORE an AR (post-rank-loop, pre-AR) measure rank 0's
    // local work only — sufficient for relative section comparison
    // since the per-rank loop is balanced.
    let dev0 = cluster.device(0);
    let stream0 = dev0.default_stream();
    dev0.bind()?;
    flambeau_backend_hip::profile::mark("ptp_prefill_start", dev0, stream0)?;
    for il in il_range {
        let il_cache = il - il_cache_offset;
        // 3a. Per-rank attn prefill (full-attn or GDN).
        let is_full_attn = !cfg.is_recurrent(il);
        for r in 0..cluster.ranks() {
            let device = cluster.device(r);
            device.bind()?;
            let stream = device.default_stream();
            let layer_tensors = &model.shards[r].layers[il];
            let hidden_a = scratch_ref.per_rank[r].hidden_a;
            let partial_attn_out = scratch_ref.per_rank[r].partial_attn_out;
            let layer_scratch = scratch_ref.per_rank[r]
                .layer
                .as_mut()
                .ok_or_else(|| anyhow!("rank {r}: missing LayerPrefillScratch"))?;
            let ops = &model.ops[r];
            if is_full_attn {
                let attn_norm = find_by_suffix(layer_tensors, il, "attn_norm.weight")?;
                let attn_q = find_by_suffix(layer_tensors, il, "attn_q.weight")?;
                let attn_k = find_by_suffix(layer_tensors, il, "attn_k.weight")?;
                let attn_v = find_by_suffix(layer_tensors, il, "attn_v.weight")?;
                let attn_output = find_by_suffix(layer_tensors, il, "attn_output.weight")?;
                let attn_q_norm = find_by_suffix(layer_tensors, il, "attn_q_norm.weight")?;
                let attn_k_norm = find_by_suffix(layer_tensors, il, "attn_k_norm.weight")?;
                let kv_cache = match &mut layer_caches[r][il_cache] {
                    LayerCache::FullAttn(kv) => kv,
                    _ => bail!(
                        "rank {r} layer {il}: full-attn path expects FullAttn cache (got non-FullAttn)"
                    ),
                };
                let full = layer_scratch
                    .full_attn
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing FullAttnPrefillScratch"))?;
                super::attn_tp::forward_full_attn_prefill_tp(
                    ops,
                    stream,
                    device,
                    cfg,
                    attn_norm,
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm,
                    attn_k_norm,
                    kv_cache,
                    full,
                    hidden_a,
                    partial_attn_out,
                    n_tokens,
                    start_position,
                    world,
                    kv_replicated,
                )
                .with_context(|| format!("full-attn prefill TP layer {il}"))?;
            } else {
                let attn_norm = find_by_suffix(layer_tensors, il, "attn_norm.weight")?;
                let attn_qkv = find_by_suffix(layer_tensors, il, "attn_qkv.weight")?;
                let attn_gate = find_by_suffix(layer_tensors, il, "attn_gate.weight")?;
                let ssm_alpha = find_by_suffix(layer_tensors, il, "ssm_alpha.weight")?;
                let ssm_beta = find_by_suffix(layer_tensors, il, "ssm_beta.weight")?;
                let ssm_a = find_by_suffix(layer_tensors, il, "ssm_a")?;
                let ssm_dt_bias = find_by_suffix(layer_tensors, il, "ssm_dt.bias")?;
                let ssm_conv1d = find_by_suffix(layer_tensors, il, "ssm_conv1d.weight")?;
                let ssm_norm = find_by_suffix(layer_tensors, il, "ssm_norm.weight")?;
                let ssm_out = find_by_suffix(layer_tensors, il, "ssm_out.weight")?;
                let layer_state = match &mut layer_caches[r][il_cache] {
                    LayerCache::Gdn(state) => state,
                    _ => bail!(
                        "rank {r} layer {il}: GDN path expects Gdn cache (got non-Gdn)"
                    ),
                };
                let gdn_scratch = layer_scratch
                    .gdn
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing GdnPrefillScratch"))?;
                super::gdn_tp::forward_gdn_prefill_tp(
                    ops,
                    stream,
                    device,
                    cfg,
                    attn_norm,
                    attn_qkv,
                    attn_gate,
                    ssm_alpha,
                    ssm_beta,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_conv1d,
                    ssm_norm,
                    ssm_out,
                    layer_state,
                    gdn_scratch,
                    hidden_a,
                    partial_attn_out,
                    n_tokens,
                    world,
                    model.tp.gdn_kq_replicated(),
                )
                .with_context(|| format!("gdn prefill TP layer {il}"))?;
            }
        }
        // V1-BENCH-CN-80B-11a — mark after the attn block (full-attn
        // or GDN) but before the AR.
        dev0.bind()?;
        flambeau_backend_hip::profile::mark(
            if is_full_attn { "ptp_attn_full" } else { "ptp_attn_gdn" },
            dev0,
            stream0,
        )?;
        // 3b. AR(hidden_a, partial_attn_out, L*hidden).
        ar_residual_prefill(ar, scratch_ref, cluster, world, elem_count_l, AttnOrFfn::Attn)?;
        dev0.bind()?;
        flambeau_backend_hip::profile::mark("ptp_attn_ar", dev0, stream0)?;

        // 3c. ffn_norm[L] over hidden_a → mid_norm.
        for r in 0..cluster.ranks() {
            let device = cluster.device(r);
            device.bind()?;
            let stream = device.default_stream();
            let layer_tensors = &model.shards[r].layers[il];
            let ffn_norm = find_by_suffix(layer_tensors, il, "ffn_norm.weight")
                .or_else(|_| find_by_suffix(layer_tensors, il, "post_attention_norm.weight"))?;
            let hidden_a = scratch_ref.per_rank[r].hidden_a;
            let layer_scratch = scratch_ref.per_rank[r]
                .layer
                .as_mut()
                .ok_or_else(|| anyhow!("rank {r}: missing LayerPrefillScratch"))?;
            let mid_norm = layer_scratch.mid_norm_f16;
            let ops = &model.ops[r];
            rmsnorm_f16(
                ops,
                stream,
                hidden_a,
                ffn_norm.ptr,
                mid_norm,
                n_tokens,
                hidden,
                cfg.rms_norm_eps,
            )
            .with_context(|| format!("ffn_norm prefill TP layer {il}"))?;
        }
        dev0.bind()?;
        flambeau_backend_hip::profile::mark("ptp_ffn_norm", dev0, stream0)?;

        // 3d. Per-rank FFN prefill (dense or MoE+optional shared).
        let moe_replicated = !cfg.is_dense_ffn() && model.moe_replicated_at(il);
        let ffn_world = if moe_replicated { 1 } else { world };
        if cfg.is_dense_ffn() {
            for r in 0..cluster.ranks() {
                let device = cluster.device(r);
                device.bind()?;
                let stream = device.default_stream();
                let layer_tensors = &model.shards[r].layers[il];
                let ffn_gate = find_by_suffix(layer_tensors, il, "ffn_gate.weight")?;
                let ffn_up = find_by_suffix(layer_tensors, il, "ffn_up.weight")?;
                let ffn_down = find_by_suffix(layer_tensors, il, "ffn_down.weight")?;
                let partial_ffn_out = scratch_ref.per_rank[r].partial_ffn_out;
                let layer_scratch = scratch_ref.per_rank[r]
                    .layer
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing LayerPrefillScratch"))?;
                let mid_norm = layer_scratch.mid_norm_f16;
                let dense_scratch = layer_scratch
                    .dense_ffn
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing DenseFfnPrefillScratch"))?;
                let ops = &model.ops[r];
                super::dense_ffn_tp::forward_dense_ffn_prefill_tp(
                    ops,
                    stream,
                    cfg,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    dense_scratch,
                    mid_norm,
                    partial_ffn_out,
                    n_tokens,
                    world,
                )
                .with_context(|| format!("dense ffn prefill TP layer {il}"))?;
            }
        } else {
            let has_shared = cfg.shared_expert_intermediate_size.is_some()
                && std::env::var("FLAMBEAU_TP_SKIP_SHARED").is_err();
            for r in 0..cluster.ranks() {
                let device = cluster.device(r);
                device.bind()?;
                let stream = device.default_stream();
                let layer_tensors = &model.shards[r].layers[il];
                let ffn_gate_inp = find_by_suffix(layer_tensors, il, "ffn_gate_inp.weight")?;
                let ffn_gate_exps = find_by_suffix(layer_tensors, il, "ffn_gate_exps.weight")?;
                let ffn_up_exps = find_by_suffix(layer_tensors, il, "ffn_up_exps.weight")?;
                let ffn_down_exps = find_by_suffix(layer_tensors, il, "ffn_down_exps.weight")?;
                let partial_ffn_out = scratch_ref.per_rank[r].partial_ffn_out;
                let shared_delta_f16 = scratch_ref.per_rank[r]
                    .layer
                    .as_ref()
                    .map(|l| l.shared_delta_f16)
                    .unwrap_or(DevicePtr(0));
                let layer_scratch = scratch_ref.per_rank[r]
                    .layer
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing LayerPrefillScratch"))?;
                let mid_norm = layer_scratch.mid_norm_f16;
                let ops = &model.ops[r];

                // 1. Router (Replicated weight; runs identically per rank).
                {
                    let moe_scratch = layer_scratch
                        .moe
                        .as_mut()
                        .ok_or_else(|| anyhow!("rank {r}: missing MoePrefillScratch"))?;
                    super::moe::forward_router_prefill(
                        ops,
                        stream,
                        cfg,
                        ffn_gate_inp,
                        moe_scratch,
                        mid_norm,
                        n_tokens,
                    )
                    .with_context(|| format!("router prefill TP layer {il}"))?;
                }
                if r == 0 {
                    flambeau_backend_hip::profile::mark(
                        "ptp_router",
                        device,
                        stream,
                    )?;
                }

                // 2. (Optional) shared expert → shared_delta_f16.
                if has_shared {
                    let shared_w_gate = find_by_suffix(layer_tensors, il, "ffn_gate_shexp.weight")?;
                    let shared_w_up = find_by_suffix(layer_tensors, il, "ffn_up_shexp.weight")?;
                    let shared_w_down = find_by_suffix(layer_tensors, il, "ffn_down_shexp.weight")?;
                    // CN-80B-15 — qwen3next's sigmoid-gated shared expert
                    // (qwen35moe lacks this projection).
                    let shared_w_gate_inp = find_by_suffix(layer_tensors, il, "ffn_gate_inp_shexp.weight").ok();
                    let shared_scratch = layer_scratch
                        .shared
                        .as_mut()
                        .ok_or_else(|| anyhow!("rank {r}: missing SharedExpertPrefillScratch"))?;
                    super::moe_tp::forward_shared_expert_prefill_tp(
                        ops,
                        stream,
                        cfg,
                        shared_w_gate,
                        shared_w_up,
                        shared_w_down,
                        shared_w_gate_inp,
                        shared_scratch,
                        mid_norm,
                        shared_delta_f16,
                        n_tokens,
                        ffn_world,
                    )
                    .with_context(|| format!("shared expert prefill TP layer {il}"))?;
                }
                if r == 0 && has_shared {
                    flambeau_backend_hip::profile::mark(
                        "ptp_shared",
                        device,
                        stream,
                    )?;
                }

                // 3. MoE FFN forward → partial_ffn_out.
                let moe_scratch = layer_scratch
                    .moe
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing MoePrefillScratch"))?;
                super::moe_tp::forward_moe_ffn_prefill_tp(
                    ops,
                    stream,
                    cfg,
                    ffn_gate_exps,
                    ffn_up_exps,
                    ffn_down_exps,
                    moe_scratch,
                    mid_norm,
                    partial_ffn_out,
                    n_tokens,
                    ffn_world,
                )
                .with_context(|| format!("moe ffn prefill TP layer {il}"))?;
                if r == 0 {
                    flambeau_backend_hip::profile::mark(
                        "ptp_moe_ffn",
                        device,
                        stream,
                    )?;
                }

                // 4. Add shared delta to MoE partial in-place.
                if has_shared {
                    flambeau_ops::hip::mlp::add_f16(
                        ops,
                        stream,
                        partial_ffn_out,
                        shared_delta_f16,
                        partial_ffn_out,
                        n_tokens * hidden,
                    )
                    .with_context(|| {
                        format!("moe (TP) prefill + shared expert add_f16 layer {il}")
                    })?;
                }
            }
        }
        // 3e. AR-residual on FFN output (or replicated add_f16 fallback).
        if ffn_world > 1 {
            ar_residual_prefill(ar, scratch_ref, cluster, world, elem_count_l, AttnOrFfn::Ffn)?;
        } else {
            for r in 0..cluster.ranks() {
                let device = cluster.device(r);
                device.bind()?;
                let stream = device.default_stream();
                let ops = &model.ops[r];
                flambeau_ops::hip::mlp::add_f16(
                    ops,
                    stream,
                    scratch_ref.per_rank[r].hidden_a,
                    scratch_ref.per_rank[r].partial_ffn_out,
                    scratch_ref.per_rank[r].hidden_a,
                    n_tokens * hidden,
                )
                .with_context(|| format!("post-MoE replicated add_f16 layer {il}"))?;
            }
        }
        dev0.bind()?;
        flambeau_backend_hip::profile::mark("ptp_ffn_ar", dev0, stream0)?;
    }
    Ok(())
}

/// **AUTO-6b2** — explicit-elem-count AR variant of [`ar_residual`]
/// for the L-batched prefill path. The decode helper derives elem
/// count from `scratch.per_rank[0].hidden_bytes / 2` (= one F16
/// hidden vector); the prefill scratch allocates `[max_tokens, hidden]`
/// so we pass `elem_count = n_tokens * hidden` directly. Same
/// stream-sync-then-AR shape; the producer_done_event optimization
/// the decode helper uses is dropped here for simplicity (one
/// `synchronize` per rank before the AR launch — measured cost is
/// ~50 µs/rank, negligible against per-layer prefill kernel wall).
fn ar_residual_prefill(
    ar: &BarP2pAllReduce,
    scratch: &mut ShardedForwardPrefillScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    world: u32,
    elem_count: u32,
    kind: AttnOrFfn,
) -> Result<()> {
    use flambeau_core::Stream;

    let partial_ptr = |r: usize| match kind {
        AttnOrFfn::Attn => scratch.per_rank[r].partial_attn_out,
        AttnOrFfn::Ffn => scratch.per_rank[r].partial_ffn_out,
    };
    let hidden_ptr = |r: usize| scratch.per_rank[r].hidden_a;

    // 1. Sync each rank's stream so peer reads see the partial-write
    //    completed (event-based ordering is the decode optimization;
    //    deferred for AUTO-6b2 — single synchronize per rank).
    for r in 0..cluster.ranks() {
        let device = cluster.device(r);
        device.bind()?;
        Stream::synchronize(device.default_stream())?;
    }

    // 2. Launch AR on each rank's stream. Same kernels as the decode
    //    path; only the `elem_count` argument grows.
    match world {
        1 => {
            // Degenerate single-rank: in-place add residual + partial.
            let device = cluster.device(0);
            device.bind()?;
            let stream = device.default_stream();
            let ops = &flambeau_ops::hip::OpsRegistry::new(device)
                .map_err(|e| anyhow!("OpsRegistry::new (single-rank AR): {e}"))?;
            flambeau_ops::hip::mlp::add_f16(
                ops,
                stream,
                hidden_ptr(0),
                partial_ptr(0),
                hidden_ptr(0),
                elem_count as usize,
            )
            .context("ar_residual_prefill: world=1 add_f16")?;
        }
        2 => {
            let hidden = [hidden_ptr(0), hidden_ptr(1)];
            let partial = [partial_ptr(0), partial_ptr(1)];
            let s0 = cluster.device(0).default_stream();
            let s1 = cluster.device(1).default_stream();
            let streams = [s0, s1];
            // SAFETY: every rank's hidden + partial point to live
            // device allocations of `elem_count * 2` bytes (alloc'd
            // in ShardedForwardPrefillScratchTp::new). Producer
            // streams synced above ⇒ peer reads are valid.
            unsafe { ar.residual_tp2(&hidden, &partial, elem_count, &streams)? };
        }
        4 => {
            let hidden = [hidden_ptr(0), hidden_ptr(1), hidden_ptr(2), hidden_ptr(3)];
            let partial = [partial_ptr(0), partial_ptr(1), partial_ptr(2), partial_ptr(3)];
            let s0 = cluster.device(0).default_stream();
            let s1 = cluster.device(1).default_stream();
            let s2 = cluster.device(2).default_stream();
            let s3 = cluster.device(3).default_stream();
            let streams = [s0, s1, s2, s3];
            // SAFETY: same as the tp2 arm, scaled to 4 ranks.
            unsafe { ar.residual_tp4(&hidden, &partial, elem_count, &streams)? };
        }
        _ => bail!("ar_residual_prefill: unsupported world {world}"),
    }
    Ok(())
}

/// What to do with the F32 logits row after the LM head emits it on
/// the head rank's device.
///
/// **Sampler-D3 Phase B (#211)** added the third variant — a "skip
/// the postlude" mode the server uses when running the GPU top-K
/// sampler directly on the head-rank's `OutputHeadScratch::logits_f32`.
pub(crate) enum LogitsSink<'a> {
    /// Run host-side argmax + return the predicted token. Existing
    /// behaviour for `forward_one_token_tp` (greedy single-token).
    HostArgmax,
    /// DtoH `[vocab]` F32 logits to the caller's `Vec<f32>`. Returns 0.
    /// Existing behaviour for `forward_one_token_tp_logits` /
    /// `forward_prefill_tp_logits`.
    HostLogits(&'a mut Vec<f32>),
    /// Leave the logits on device — caller is responsible for consuming
    /// `head_scratch.logits_f32` before the next forward call clobbers
    /// it. Returns 0. Used by the GPU-side sampler path.
    KeepOnDevice,
}

/// Shared body for `forward_one_token_tp` (argmax host-side) and
/// `forward_one_token_tp_logits` (download F32 row to host) and
/// `forward_one_token_tp_keep_logits_on_device` (Sampler-D3 Phase B).
fn forward_one_token_tp_inner(
    model: &Qwen3MoETpModel,
    scratch: &mut ShardedForwardOneTokenScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    ar: &BarP2pAllReduce,
    layer_caches: &mut [Vec<LayerCache>],
    token_id: u32,
    position: usize,
    sink: LogitsSink<'_>,
) -> anyhow::Result<u32> {
    let cfg = &model.config;
    let world = cluster.ranks() as u32;
    if world != 1 && world != 2 && world != 4 {
        bail!("TP-2d supports world ∈ {{1, 2, 4}}; got {world}");
    }
    if scratch.per_rank.len() != cluster.ranks() {
        bail!(
            "scratch.per_rank.len()={} != cluster.ranks()={}",
            scratch.per_rank.len(),
            cluster.ranks()
        );
    }
    if layer_caches.len() != cluster.ranks() {
        bail!(
            "layer_caches.len()={} != cluster.ranks()={}",
            layer_caches.len(),
            cluster.ranks()
        );
    }

    // 1. Embed gather on every rank — token_embd is Replicated, so each
    //    rank dequantises and writes the F16 hidden into its own
    //    hidden_a. Deterministic → bit-identical across ranks.
    for r in 0..cluster.ranks() {
        let device = cluster.device(r);
        device.bind()?;
        let stream = device.default_stream();
        forward_embed_decode_host(
            device,
            stream,
            &model.shards[r].token_embd,
            token_id,
            scratch.per_rank[r].hidden_a,
            cfg.hidden_size,
        )
        .with_context(|| format!("rank {r} embed"))?;
    }

    // 2. Layer loop. Every rank runs every layer (full TP topology;
    //    not pipeline-parallel).
    //
    // Dispatch matches the non-TP paths (`forward/layer.rs`,
    // `forward/pp.rs`): `cfg.is_recurrent(il)` is the source of truth.
    // The earlier local check `interval > 0 && (il+1) % interval == 0`
    // was wrong for dense arches (qwen35) where `full_attention_interval`
    // is None → interval == 0 → every layer routed to GDN, which the
    // dense model has no tensors for, producing all-NaN logits.
    let n_layers = cfg.num_layers;
    let layer_limit: usize = std::env::var("FLAMBEAU_TP_LAYER_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(n_layers);
    let probe = std::env::var("FLAMBEAU_TP_PROBE").is_ok();
    if probe {
        debug_probe_rank0_hidden(scratch, cluster, "embed", usize::MAX)?;
    }
    // CN-80B-22 — per-layer-type wall time for decode profiling. The mark
    // is a thread-local check + HipEvent record on rank 0's stream when
    // `profile::enable()` was called; otherwise it's a single bool load.
    // Aggregates across the n_run iterations: total/mean per name reveals
    // whether full-attn or GDN dominates, and at which ctx the curve
    // bends. See the `profile_tp_decode` test for the harness.
    let dev0 = cluster.device(0);
    let stream0 = dev0.default_stream();
    let n_run = layer_limit.min(n_layers);
    for il in 0..n_run {
        let is_full_attn = !cfg.is_recurrent(il);
        if flambeau_backend_hip::profile::is_enabled() {
            dev0.bind()?;
            flambeau_backend_hip::profile::mark("tp_dec_layer_start", dev0, stream0)?;
        }
        if is_full_attn {
            forward_full_attn_layer_tp(
                model,
                scratch,
                cluster,
                ar,
                layer_caches,
                il,
                il, // pure-TP: caches sized to cfg.num_layers (absolute idx).
                position,
                world,
            )
            .with_context(|| format!("full-attn layer {il}"))?;
            if flambeau_backend_hip::profile::is_enabled() {
                dev0.bind()?;
                flambeau_backend_hip::profile::mark("tp_dec_full_attn", dev0, stream0)?;
            }
        } else {
            forward_gdn_layer_tp(
                model,
                scratch,
                cluster,
                ar,
                layer_caches,
                il,
                il, // pure-TP: caches sized to cfg.num_layers (absolute idx).
                world,
            )
            .with_context(|| format!("gdn layer {il}"))?;
            if flambeau_backend_hip::profile::is_enabled() {
                dev0.bind()?;
                flambeau_backend_hip::profile::mark("tp_dec_gdn", dev0, stream0)?;
            }
        }
        if probe {
            for r in 0..cluster.ranks() {
                debug_probe_rank_hidden(scratch, cluster, "after-layer hidden_a", il, r)?;
            }
        }
    }
    if flambeau_backend_hip::profile::is_enabled() {
        dev0.bind()?;
        flambeau_backend_hip::profile::mark("tp_dec_post_layers", dev0, stream0)?;
    }

    // 3. Output head + argmax on head_rank only. LM head + token_embd
    //    are Replicated in V1, so head_rank's local copy is sufficient.
    let head_rank = scratch.head_rank.0 as usize;
    let device = cluster.device(head_rank);
    device.bind()?;
    let stream = device.default_stream();
    let head_shard = &model.shards[head_rank];
    let lm_head = head_shard.output.as_ref().unwrap_or(&head_shard.token_embd);
    // Snapshot the DevicePtr by Copy *before* taking the mutable
    // borrow on `output_head` — avoids overlapping borrows on
    // `scratch.per_rank`.
    let hidden_a = scratch.per_rank[head_rank].hidden_a;
    let head_scratch = scratch.per_rank[head_rank]
        .output_head
        .as_mut()
        .ok_or_else(|| anyhow!("head_rank={head_rank} missing OutputHeadScratch"))?;
    let logits_f32 = head_scratch.logits_f32;
    let ops = &model.ops[head_rank];
    forward_output_head_decode(
        ops,
        stream,
        cfg,
        &head_shard.output_norm,
        lm_head,
        head_scratch,
        hidden_a,
    )
    .context("forward_output_head_decode (TP)")?;

    // 4. Dispatch on sink: DtoH, host argmax, or leave on device.
    match sink {
        LogitsSink::HostLogits(out) => {
            // Download `vocab_size` F32 logits to host. Mirrors PP's
            // `download_logits_host` shape — caller owns the buffer.
            out.clear();
            out.resize(cfg.vocab_size, 0.0f32);
            // SAFETY: logits_f32 is valid for cfg.vocab_size F32 values
            // on `device`; out.as_mut_ptr() is host memory of matching size.
            unsafe {
                <flambeau_backend_hip::HipDevice as flambeau_core::Device>::memcpy_async(
                    device,
                    stream,
                    flambeau_core::CopyDirection::DeviceToHost,
                    flambeau_core::DevicePtr(out.as_mut_ptr() as usize),
                    logits_f32,
                    cfg.vocab_size * 4,
                )?;
            }
            flambeau_core::Stream::synchronize(stream)?;
            Ok(0)
        }
        LogitsSink::HostArgmax => {
            let token = argmax_token_host(device, stream, logits_f32, cfg.vocab_size)
                .context("argmax_token_host (TP)")?;
            Ok(token)
        }
        LogitsSink::KeepOnDevice => {
            // No sync here — the caller's downstream kernel (topk on
            // the same default_stream) will serialise device-side
            // against the output_head writes via stream ordering.
            // CPU-side blocking would just stall the dispatch loop.
            Ok(0)
        }
    }
}

/// Per-rank dispatch of one full-attn layer + dense FFN + 2 AllReduces.
///
/// `il` is the absolute layer index used to look up weight tensors in
/// `model.shards[r].layers[il]`. `il_cache` is the index into
/// `layer_caches[r]`; callers under the pure-TP path pass `il_cache =
/// il` (caches are sized to `cfg.num_layers`). The AUTO-4d hybrid
/// driver passes `il_cache = il - layer_range.start` because each
/// stage allocates only its slice of layer caches.
pub(crate) fn forward_full_attn_layer_tp(
    model: &Qwen3MoETpModel,
    scratch: &mut ShardedForwardOneTokenScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    ar: &BarP2pAllReduce,
    layer_caches: &mut [Vec<LayerCache>],
    il: usize,
    il_cache: usize,
    position: usize,
    world: u32,
) -> anyhow::Result<()> {
    if std::env::var("FLAMBEAU_TP_PROBE").is_ok() {
        eprintln!("  PROBE entering forward_full_attn_layer_tp il={il} world={world}");
    }
    let cfg = &model.config;

    // 1. Per-rank attn_tp → partial_attn_out.
    for r in 0..cluster.ranks() {
        let device = cluster.device(r);
        device.bind()?;
        let stream = device.default_stream();
        let layer_tensors = &model.shards[r].layers[il];
        let attn_norm = find_by_suffix(layer_tensors, il, "attn_norm.weight")?;
        let attn_q = find_by_suffix(layer_tensors, il, "attn_q.weight")?;
        let attn_k = find_by_suffix(layer_tensors, il, "attn_k.weight")?;
        let attn_v = find_by_suffix(layer_tensors, il, "attn_v.weight")?;
        let attn_output = find_by_suffix(layer_tensors, il, "attn_output.weight")?;
        let attn_q_norm = find_by_suffix(layer_tensors, il, "attn_q_norm.weight")?;
        let attn_k_norm = find_by_suffix(layer_tensors, il, "attn_k_norm.weight")?;

        // V1-BENCH-#116 — dispatch on cache variant; the generic
        // `forward_full_attn_decode_tp<L>` body picks the right
        // attention kernel (F16 vs Q8_0) via L::NAME.
        let hidden_a = scratch.per_rank[r].hidden_a;
        let partial_attn_out = scratch.per_rank[r].partial_attn_out;
        let layer_scratch = scratch.per_rank[r]
            .layer
            .as_mut()
            .ok_or_else(|| anyhow!("rank {r}: missing LayerForwardScratch"))?;
        let full = layer_scratch
            .full_attn
            .as_mut()
            .ok_or_else(|| anyhow!("rank {r}: missing FullAttnScratch"))?;

        let ops = &model.ops[r];
        match &mut layer_caches[r][il_cache] {
            LayerCache::FullAttn(kv) => forward_full_attn_decode_tp(
                &ops, stream, device, cfg,
                attn_norm, attn_q, attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm,
                kv, full, hidden_a, partial_attn_out, position, world, model.tp.kv_replicated(),
            )?,
            LayerCache::FullAttnQ8(kv) => forward_full_attn_decode_tp(
                &ops, stream, device, cfg,
                attn_norm, attn_q, attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm,
                kv, full, hidden_a, partial_attn_out, position, world, model.tp.kv_replicated(),
            )?,
            _ => bail!("rank {r} layer {il}: expected FullAttn cache (got non-FullAttn variant)"),
        }
    }
    if std::env::var("FLAMBEAU_TP_PROBE").is_ok() {
        let p = scratch.per_rank[0].partial_attn_out;
        debug_probe_rank0_named(scratch, cluster, "post-attn partial", il, p)?;
    }

    // 2+3a. Fused AR-residual + post-attention RMSNorm (TP-3b-i2).
    //   world > 1: BarP2pAllReduce::residual_rmsnorm_tp{2,4} replaces
    //   the AR launch + the per-rank rmsnorm_f16 launch with a single
    //   kernel call per rank.
    //   world = 1: degenerate — fall back to add_f16 + rmsnorm_f16.
    let post_norm_ptrs: Vec<DevicePtr> = (0..cluster.ranks())
        .map(|r| {
            find_by_suffix(&model.shards[r].layers[il], il, "post_attention_norm.weight")
                .or_else(|_| find_by_suffix(&model.shards[r].layers[il], il, "ffn_norm.weight"))
                .map(|t| t.ptr)
        })
        .collect::<anyhow::Result<_>>()?;
    let mid_norm_ptrs: Vec<DevicePtr> = (0..cluster.ranks())
        .map(|r| {
            scratch.per_rank[r]
                .layer
                .as_ref()
                .map(|l| l.mid_norm_f16)
                .ok_or_else(|| anyhow!("rank {r}: missing LayerForwardScratch"))
        })
        .collect::<anyhow::Result<_>>()?;
    let probe = std::env::var("FLAMBEAU_TP_PROBE").is_ok();
    if world > 1 {
        ar_residual_rmsnorm(
            ar,
            scratch,
            cluster,
            world,
            AttnOrFfn::Attn,
            &post_norm_ptrs,
            &mid_norm_ptrs,
            cfg.rms_norm_eps,
        )?;
    } else {
        // world=1: degenerate. AR is `add_f16`, then plain rmsnorm.
        for r in 0..cluster.ranks() {
            let device = cluster.device(r);
            device.bind()?;
            let stream = device.default_stream();
            let ops = &model.ops[r];
            flambeau_ops::hip::mlp::add_f16(
                &ops,
                stream,
                scratch.per_rank[r].hidden_a,
                scratch.per_rank[r].partial_attn_out,
                scratch.per_rank[r].hidden_a,
                cfg.hidden_size,
            )?;
            rmsnorm_f16(
                &ops,
                stream,
                scratch.per_rank[r].hidden_a,
                post_norm_ptrs[r],
                mid_norm_ptrs[r],
                1,
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )
            .context("dense ffn (TP world=1) pre-norm")?;
        }
    }

    if probe {
        debug_probe_rank0_hidden(scratch, cluster, "post-attn-AR hidden_a", il)?;
        let p = mid_norm_ptrs[0];
        debug_probe_rank0_named(scratch, cluster, "post-attn-AR mid_norm", il, p)?;
    }

    // 3b. FFN consuming the AR'd-and-normed mid_norm. Dense path
    //     (arch=qwen35) calls forward_dense_ffn_decode_tp; MoE path
    //     (qwen3moe / qwen35moe / qwen36moe) calls
    //     forward_moe_ffn_decode_tp via forward_ffn_block_tp.
    //
    // **TP-7-arch** — when the loader had to fall back to Replicated
    // upload for this layer's MoE expert tensors (K-quant misalignment),
    // each rank already holds the full MoE weights and computes the
    // full FFN output. Pass `ffn_world = 1` so the MoE kernels emit a
    // single-rank result, and skip the post-FFN AR (the per-rank
    // outputs are already identical full-hidden updates).
    let ffn_world = if !cfg.is_dense_ffn() && model.moe_replicated_at(il) {
        1
    } else {
        world
    };
    forward_ffn_block_tp(
        model, scratch, cluster, &mid_norm_ptrs, il, ffn_world, false,
    )?;
    if probe {
        let p = scratch.per_rank[0].partial_ffn_out;
        debug_probe_rank0_named(scratch, cluster, "post-ffn partial", il, p)?;
    }

    // 4. AR-residual on FFN output. Skipped when the FFN is replicated
    //    (TP-7-arch); the per-rank partial is already a full update.
    if ffn_world > 1 {
        ar_residual(ar, scratch, cluster, world, AttnOrFfn::Ffn)?;
    } else {
        for r in 0..cluster.ranks() {
            let device = cluster.device(r);
            device.bind()?;
            let stream = device.default_stream();
            let ops = &model.ops[r];
            flambeau_ops::hip::mlp::add_f16(
                &ops,
                stream,
                scratch.per_rank[r].hidden_a,
                scratch.per_rank[r].partial_ffn_out,
                scratch.per_rank[r].hidden_a,
                cfg.hidden_size,
            )?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum AttnOrFfn {
    Attn,
    Ffn,
}

/// **P2.9b-i2-D-wire** — public re-export of [`AttnOrFfn`] so the
/// hybrid batched-decode driver can call [`ar_residual_prefill_pub`].
#[derive(Clone, Copy)]
pub enum AttnOrFfnPub {
    Attn,
    Ffn,
}

/// **P2.9b-i2-D-wire** — public wrapper around the private
/// `ar_residual_prefill` helper, used by the hybrid batched-decode
/// driver (sibling of this `forward_decode_batched_tp` driver) to
/// schedule per-stage `BarP2pAllReduce` calls on `[N, hidden]`
/// partials.
pub fn ar_residual_prefill_pub(
    ar: &BarP2pAllReduce,
    scratch: &mut ShardedForwardPrefillScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    world: u32,
    elem_count: u32,
    kind: AttnOrFfnPub,
) -> Result<()> {
    let kind_priv = match kind {
        AttnOrFfnPub::Attn => AttnOrFfn::Attn,
        AttnOrFfnPub::Ffn => AttnOrFfn::Ffn,
    };
    ar_residual_prefill(ar, scratch, cluster, world, elem_count, kind_priv)
}

/// **TP-3a** — async AR via HIP events. Replaces TP-2d's per-rank
/// host `Stream::synchronize` with driver-side cross-stream waits.
///
/// Pattern:
/// 1. Each rank records its `producer_done_event` on its compute
///    stream (the stream that just wrote the partial buffer). The
///    record is non-blocking on the host.
/// 2. Each rank's AR launch first issues `stream_wait` for every
///    *other* rank's producer event. The waits go on the AR launch's
///    stream, which on V1 is the same compute stream — so subsequent
///    work on that stream blocks until peers signal, but the host
///    doesn't.
/// 3. The AR kernel launches on each rank's stream. By the time it
///    reads peer partials, every peer's producer has signalled.
///
/// Same-rank ordering (rank `r`'s producer on stream `S_r` followed
/// by rank `r`'s AR also on `S_r`) auto-serialises via stream order;
/// no event needed.
fn ar_residual(
    ar: &BarP2pAllReduce,
    scratch: &ShardedForwardOneTokenScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    world: u32,
    which: AttnOrFfn,
) -> anyhow::Result<()> {
    // Build the per-rank pointer arrays.
    let partial_ptr = |r: usize| match which {
        AttnOrFfn::Attn => scratch.per_rank[r].partial_attn_out,
        AttnOrFfn::Ffn => scratch.per_rank[r].partial_ffn_out,
    };
    let hidden_ptr = |r: usize| scratch.per_rank[r].hidden_a;

    // 1. Each rank records its producer-done event on its own stream
    //    after the partial-write kernel. record() is host-non-blocking.
    for r in 0..cluster.ranks() {
        let device = cluster.device(r);
        device.bind()?;
        scratch.per_rank[r]
            .producer_done_event
            .record(device.default_stream())?;
    }

    // 2. Each rank's AR launch waits on every other rank's producer
    //    event before the AR kernel reads that peer's partial buffer.
    //    Same-rank waits are unnecessary (stream ordering auto-serialises).
    for r in 0..cluster.ranks() {
        let device = cluster.device(r);
        device.bind()?;
        let stream = device.default_stream();
        for peer in 0..cluster.ranks() {
            if peer != r {
                scratch.per_rank[peer]
                    .producer_done_event
                    .stream_wait(stream)?;
            }
        }
    }

    // 3. Launch AR on each rank's stream. The driver schedules the
    //    launch as soon as the per-rank wait list resolves.
    let elem_count = scratch.per_rank[0].hidden_bytes / 2; // bytes/F16
    match world {
        2 => {
            let hidden = [hidden_ptr(0), hidden_ptr(1)];
            let partial = [partial_ptr(0), partial_ptr(1)];
            let s0 = cluster.device(0).default_stream();
            let s1 = cluster.device(1).default_stream();
            let streams = [s0, s1];
            // SAFETY: every rank's hidden + partial point to live device
            // allocations of `elem_count * 2` bytes (alloc'd in
            // ShardedForwardOneTokenScratchTp::new). Producer streams
            // synced above ⇒ peer reads are valid.
            unsafe { ar.residual_tp2(&hidden, &partial, elem_count as u32, &streams)? };
        }
        4 => {
            let hidden = [hidden_ptr(0), hidden_ptr(1), hidden_ptr(2), hidden_ptr(3)];
            let partial = [
                partial_ptr(0),
                partial_ptr(1),
                partial_ptr(2),
                partial_ptr(3),
            ];
            let s0 = cluster.device(0).default_stream();
            let s1 = cluster.device(1).default_stream();
            let s2 = cluster.device(2).default_stream();
            let s3 = cluster.device(3).default_stream();
            let streams = [s0, s1, s2, s3];
            // SAFETY: same as the tp2 arm.
            unsafe { ar.residual_tp4(&hidden, &partial, elem_count as u32, &streams)? };
        }
        _ => bail!("ar_residual: unsupported world {world}"),
    }
    Ok(())
}

/// **TP-4b-i2 / 4c-i2** — per-rank FFN dispatch. Branches on
/// `cfg.is_dense_ffn()`: dense routes through
/// [`super::dense_ffn_tp::forward_dense_ffn_decode_tp`]; MoE routes
/// through [`super::moe_tp::forward_moe_ffn_decode_tp`] preceded by
/// [`super::moe::forward_router_decode`] (router is Replicated, runs
/// identically per rank).
///
/// Shared expert (qwen35moe / qwen36moe) is folded by the layer
/// driver via the second-residual path; for the simplest TP case
/// (qwen3moe Coder-30B, no shared expert) this function emits the
/// per-rank FFN partial directly.
fn forward_ffn_block_tp(
    model: &Qwen3MoETpModel,
    scratch: &mut ShardedForwardOneTokenScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    mid_norm_ptrs: &[DevicePtr],
    il: usize,
    world: u32,
    // **TP-perf-c2** — when true, the per-rank FFN's x_q8_1 was already
    // populated by the upstream fused AR+RMSNorm+Q8_1. Only honoured
    // for the dense-FFN path; MoE always re-quantises because the
    // router needs F16 input.
    pre_quantized: bool,
) -> anyhow::Result<()> {
    let cfg = &model.config;
    if cfg.is_dense_ffn() {
        for r in 0..cluster.ranks() {
            let device = cluster.device(r);
            device.bind()?;
            let stream = device.default_stream();
            let layer_tensors = &model.shards[r].layers[il];
            let ffn_gate = find_by_suffix(layer_tensors, il, "ffn_gate.weight")?;
            let ffn_up = find_by_suffix(layer_tensors, il, "ffn_up.weight")?;
            let ffn_down = find_by_suffix(layer_tensors, il, "ffn_down.weight")?;
            let partial_ffn_out = scratch.per_rank[r].partial_ffn_out;
            let mid_norm_f16 = mid_norm_ptrs[r];
            let layer_scratch = scratch.per_rank[r].layer.as_mut().unwrap();
            let dense_scratch = layer_scratch
                .dense_ffn
                .as_mut()
                .ok_or_else(|| anyhow!("rank {r}: missing DenseFfnScratch"))?;
            let ops = &model.ops[r];
            super::dense_ffn_tp::forward_dense_ffn_decode_tp(
                &ops,
                stream,
                cfg,
                ffn_gate,
                ffn_up,
                ffn_down,
                dense_scratch,
                mid_norm_f16,
                partial_ffn_out,
                world,
                pre_quantized,
            )?;
        }
    } else {
        // MoE FFN dispatch.
        for r in 0..cluster.ranks() {
            let device = cluster.device(r);
            device.bind()?;
            let stream = device.default_stream();
            let layer_tensors = &model.shards[r].layers[il];
            let ffn_gate_inp = find_by_suffix(layer_tensors, il, "ffn_gate_inp.weight")?;
            let ffn_gate_exps = find_by_suffix(layer_tensors, il, "ffn_gate_exps.weight")?;
            let ffn_up_exps = find_by_suffix(layer_tensors, il, "ffn_up_exps.weight")?;
            let ffn_down_exps = find_by_suffix(layer_tensors, il, "ffn_down_exps.weight")?;
            let partial_ffn_out = scratch.per_rank[r].partial_ffn_out;
            let mid_norm_f16 = mid_norm_ptrs[r];
            // Snapshot all DevicePtrs we'll need before taking the
            // mutable layer_scratch borrow (which may sub-borrow moe_scratch
            // and shared_scratch).
            let shared_delta_f16 = scratch.per_rank[r]
                .layer
                .as_ref()
                .map(|l| l.shared_delta_f16)
                .unwrap_or(DevicePtr(0));
            // B5 bisect: FLAMBEAU_TP_SKIP_SHARED=1 skips the shared expert
            // path entirely (no shared partial added to MoE partial).
            let skip_shared = std::env::var("FLAMBEAU_TP_SKIP_SHARED").is_ok();
            let has_shared = cfg.shared_expert_intermediate_size.is_some() && !skip_shared;
            let layer_scratch = scratch.per_rank[r].layer.as_mut().unwrap();
            let ops = &model.ops[r];

            // 1. Router (Replicated weight; runs identically per rank).
            {
                let moe_scratch = layer_scratch
                    .moe
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing MoeScratch"))?;
                super::moe::forward_router_decode(
                    &ops,
                    stream,
                    cfg,
                    ffn_gate_inp,
                    moe_scratch,
                    mid_norm_f16,
                )?;
                // B5 router-divergence bisect: dump expert_ids per layer.
                // Two env gates:
                //   FLAMBEAU_TP_LAYER0_BISECT=1 — layer 0 only, both ranks.
                //   FLAMBEAU_PARITY_LAYER_DUMP=1 — every layer, rank 0 only,
                //                                  format matching layer.rs PP.
                let layer0_bisect =
                    std::env::var("FLAMBEAU_TP_LAYER0_BISECT").is_ok() && il == 0;
                let layer_dump = std::env::var("FLAMBEAU_PARITY_LAYER_DUMP").is_ok();
                if (layer0_bisect && r < cluster.ranks()) || (layer_dump && r == 0) {
                    use flambeau_core::CopyDirection;
                    let device = cluster.device(r);
                    device.bind()?;
                    let stream2 = device.default_stream();
                    let top_k = cfg.num_experts_per_tok;
                    let mut ids = vec![0i32; top_k];
                    let mut wts = vec![0f32; top_k];
                    unsafe {
                        device.memcpy_async(
                            stream2,
                            CopyDirection::DeviceToHost,
                            DevicePtr(ids.as_mut_ptr() as usize),
                            moe_scratch.expert_ids,
                            top_k * std::mem::size_of::<i32>(),
                        )?;
                        device.memcpy_async(
                            stream2,
                            CopyDirection::DeviceToHost,
                            DevicePtr(wts.as_mut_ptr() as usize),
                            moe_scratch.expert_weights,
                            top_k * std::mem::size_of::<f32>(),
                        )?;
                    }
                    flambeau_core::Stream::synchronize(stream2)?;
                    if layer_dump && r == 0 {
                        eprintln!(
                            "[router-dump] TP il={il} expert_ids={ids:?} weights={wts:?}"
                        );
                    }
                    if layer0_bisect {
                        eprintln!(
                            "  PROBE router rank={r} il={il} expert_ids={ids:?} weights={wts:?}"
                        );
                    }
                }
            }

            // 2. (Optional) shared expert → shared_delta_f16.
            if has_shared {
                let shared_w_gate =
                    find_by_suffix(layer_tensors, il, "ffn_gate_shexp.weight")?;
                let shared_w_up = find_by_suffix(layer_tensors, il, "ffn_up_shexp.weight")?;
                let shared_w_down =
                    find_by_suffix(layer_tensors, il, "ffn_down_shexp.weight")?;
                // CN-80B-15 — qwen3next's sigmoid-gated shared expert.
                let shared_w_gate_inp =
                    find_by_suffix(layer_tensors, il, "ffn_gate_inp_shexp.weight").ok();
                let shared_scratch = layer_scratch
                    .shared
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing SharedExpertScratch"))?;
                super::moe_tp::forward_shared_expert_decode_tp(
                    &ops,
                    stream,
                    cfg,
                    shared_w_gate,
                    shared_w_up,
                    shared_w_down,
                    shared_w_gate_inp,
                    shared_scratch,
                    mid_norm_f16,
                    shared_delta_f16,
                    world,
                )?;
            }

            // 3. MoE FFN forward → partial_ffn_out.
            //    For shared-expert arches, partial_ffn_out is overwritten
            //    by moe_combine_no_residual; we then add shared_delta via
            //    add_f16. (Single in-layer launch overhead — TP-4d-i3
            //    can replace with a moe_combine_with_extra variant if
            //    profiling shows it matters.)
            let moe_scratch = layer_scratch
                .moe
                .as_mut()
                .ok_or_else(|| anyhow!("rank {r}: missing MoeScratch"))?;
            super::moe_tp::forward_moe_ffn_decode_tp(
                &ops,
                stream,
                cfg,
                ffn_gate_exps,
                ffn_up_exps,
                ffn_down_exps,
                moe_scratch,
                mid_norm_f16,
                partial_ffn_out,
                world,
            )?;
            // B5 bisect — dump MoE partial + shared delta separately
            // (gated on FLAMBEAU_TP_LAYER0_BISECT). MoE partial is the
            // RowParallel-sliced sum-over-experts; shared delta is the
            // shared-expert RowParallel partial. Both per-rank partials
            // get AR'd later, so per-rank values are sliced (asymmetric).
            if std::env::var("FLAMBEAU_TP_LAYER0_BISECT").is_ok() && il == 0 {
                debug_probe_named_rank(
                    scratch, cluster, "moe partial (pre-shared-add)", il, partial_ffn_out, r,
                )?;
                if has_shared {
                    debug_probe_named_rank(
                        scratch, cluster, "shared delta", il, shared_delta_f16, r,
                    )?;
                }
            }
            if has_shared {
                flambeau_ops::hip::mlp::add_f16(
                    &ops,
                    stream,
                    partial_ffn_out,
                    shared_delta_f16,
                    partial_ffn_out,
                    cfg.hidden_size,
                )
                .context("moe (TP) + shared expert add_f16")?;
                if std::env::var("FLAMBEAU_TP_LAYER0_BISECT").is_ok() && il == 0 {
                    debug_probe_named_rank(
                        scratch, cluster, "moe partial (post-shared-add)", il, partial_ffn_out, r,
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// **TP-3b-i2** — fused AR + residual + RMSNorm in one call per rank.
/// Replaces the (`ar_residual` + per-rank `rmsnorm_f16`) pair on the
/// post-attn boundary inside a layer.
///
/// `weights[r]` is rank `r`'s `Replicated` post-attention RMSNorm
/// weight (F16 `[hidden]`); `out_norm[r]` is rank `r`'s F16 `[hidden]`
/// destination buffer (typically `LayerForwardScratch::mid_norm_f16`).
///
/// Producer-stream ordering uses the same event protocol as
/// [`ar_residual`]: per-rank `producer_done_event.record(stream)`,
/// then per-rank `stream_wait` on every peer's event before launching
/// the fused kernel. Bit-exact equivalent to ar_residual followed by
/// rmsnorm_f16 — same FP32 reduction order on the same operands.
fn ar_residual_rmsnorm(
    ar: &BarP2pAllReduce,
    scratch: &ShardedForwardOneTokenScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    world: u32,
    which: AttnOrFfn,
    weights: &[DevicePtr],
    out_norm: &[DevicePtr],
    eps: f32,
) -> anyhow::Result<()> {
    let partial_ptr = |r: usize| match which {
        AttnOrFfn::Attn => scratch.per_rank[r].partial_attn_out,
        AttnOrFfn::Ffn => scratch.per_rank[r].partial_ffn_out,
    };
    let hidden_ptr = |r: usize| scratch.per_rank[r].hidden_a;

    // 1. Per-rank producer-done event record (host non-blocking).
    for r in 0..cluster.ranks() {
        let device = cluster.device(r);
        device.bind()?;
        scratch.per_rank[r]
            .producer_done_event
            .record(device.default_stream())?;
    }
    // 2. Each rank waits on every peer's producer event.
    for r in 0..cluster.ranks() {
        let device = cluster.device(r);
        device.bind()?;
        let stream = device.default_stream();
        for peer in 0..cluster.ranks() {
            if peer != r {
                scratch.per_rank[peer]
                    .producer_done_event
                    .stream_wait(stream)?;
            }
        }
    }
    // 3. Launch the fused AR+norm.
    let elem_count = scratch.per_rank[0].hidden_bytes / 2; // bytes/F16
    let n = elem_count as u32;
    match world {
        2 => {
            let hidden = [hidden_ptr(0), hidden_ptr(1)];
            let partial = [partial_ptr(0), partial_ptr(1)];
            let w = [weights[0], weights[1]];
            let o = [out_norm[0], out_norm[1]];
            let s0 = cluster.device(0).default_stream();
            let s1 = cluster.device(1).default_stream();
            let streams = [s0, s1];
            // SAFETY: caller (forward_*_layer_tp) guarantees:
            //   - hidden / partial / out_norm / weights all live
            //     `elem_count`-element F16 allocs on the matching rank's device,
            //   - producer streams are sync'd via the event protocol above.
            unsafe {
                ar.residual_rmsnorm_tp2(&hidden, &partial, &w, &o, n, eps, &streams)?
            };
        }
        4 => {
            let hidden = [hidden_ptr(0), hidden_ptr(1), hidden_ptr(2), hidden_ptr(3)];
            let partial = [partial_ptr(0), partial_ptr(1), partial_ptr(2), partial_ptr(3)];
            let w = [weights[0], weights[1], weights[2], weights[3]];
            let o = [out_norm[0], out_norm[1], out_norm[2], out_norm[3]];
            let s0 = cluster.device(0).default_stream();
            let s1 = cluster.device(1).default_stream();
            let s2 = cluster.device(2).default_stream();
            let s3 = cluster.device(3).default_stream();
            let streams = [s0, s1, s2, s3];
            // SAFETY: same as the tp2 arm.
            unsafe {
                ar.residual_rmsnorm_tp4(&hidden, &partial, &w, &o, n, eps, &streams)?
            };
        }
        _ => bail!("ar_residual_rmsnorm: unsupported world {world}"),
    }
    Ok(())
}

/// Look up a per-layer tensor by its suffix after `blk.<il>.`. The TP
/// layer tensor list is small (~13 entries / layer); linear scan is
/// fine for V1.
fn find_by_suffix<'a>(
    tensors: &'a [TpLayerTensor],
    il: usize,
    suffix: &str,
) -> anyhow::Result<&'a DeviceTensor> {
    let want = format!("blk.{il}.{suffix}");
    tensors
        .iter()
        .find(|t| t.name.as_ref() == want.as_str())
        .map(|t| &t.tensor)
        .ok_or_else(|| anyhow!("layer {il}: missing tensor with suffix `{suffix}`"))
}

/// Per-rank dispatch of one GDN layer + 2 AllReduces (TP-4a).
///
/// Mirrors [`forward_full_attn_layer_tp`] for the GDN flavour: GDN's
/// "delta" plays the same residual-stream role as full-attn's, so we
/// fold the per-rank GDN partial into `hidden_a` via the same
/// post-attn AR. The post-norm + dense FFN + post-FFN AR mirror the
/// full-attn path exactly.
pub(crate) fn forward_gdn_layer_tp(
    model: &Qwen3MoETpModel,
    scratch: &mut ShardedForwardOneTokenScratchTp,
    cluster: &flambeau_backend_hip::HipCluster,
    ar: &BarP2pAllReduce,
    layer_caches: &mut [Vec<LayerCache>],
    il: usize,
    il_cache: usize,
    world: u32,
) -> anyhow::Result<()> {
    let probe = std::env::var("FLAMBEAU_TP_PROBE").is_ok();
    if probe {
        eprintln!("  PROBE entering forward_gdn_layer_tp il={il} world={world}");
    }
    let cfg = &model.config;

    // 1. Per-rank GDN forward → partial_attn_out.
    for r in 0..cluster.ranks() {
        let device = cluster.device(r);
        device.bind()?;
        let stream = device.default_stream();
        let layer_tensors = &model.shards[r].layers[il];
        let attn_norm = find_by_suffix(layer_tensors, il, "attn_norm.weight")?;
        let attn_qkv = find_by_suffix(layer_tensors, il, "attn_qkv.weight")?;
        let attn_gate = find_by_suffix(layer_tensors, il, "attn_gate.weight")?;
        let ssm_alpha = find_by_suffix(layer_tensors, il, "ssm_alpha.weight")?;
        let ssm_beta = find_by_suffix(layer_tensors, il, "ssm_beta.weight")?;
        let ssm_a = find_by_suffix(layer_tensors, il, "ssm_a")?;
        let ssm_dt_bias = find_by_suffix(layer_tensors, il, "ssm_dt.bias")?;
        let ssm_conv1d = find_by_suffix(layer_tensors, il, "ssm_conv1d.weight")?;
        let ssm_norm = find_by_suffix(layer_tensors, il, "ssm_norm.weight")?;
        let ssm_out = find_by_suffix(layer_tensors, il, "ssm_out.weight")?;

        let layer_state = match &mut layer_caches[r][il_cache] {
            LayerCache::Gdn(state) => state,
            _ => bail!("rank {r} layer {il}: expected Gdn cache (got non-Gdn variant)"),
        };
        // Snapshot Copy DevicePtrs before mutable borrow on layer scratch.
        let hidden_a = scratch.per_rank[r].hidden_a;
        let partial_attn_out = scratch.per_rank[r].partial_attn_out;
        let layer_scratch = scratch.per_rank[r]
            .layer
            .as_mut()
            .ok_or_else(|| anyhow!("rank {r}: missing LayerForwardScratch"))?;
        let gdn_scratch = layer_scratch
            .gdn
            .as_mut()
            .ok_or_else(|| anyhow!("rank {r}: missing GdnScratch"))?;
        let ops = &model.ops[r];

        super::gdn_tp::forward_gdn_decode_tp(
            &ops,
            stream,
            device,
            cfg,
            attn_norm,
            attn_qkv,
            attn_gate,
            ssm_alpha,
            ssm_beta,
            ssm_a,
            ssm_dt_bias,
            ssm_conv1d,
            ssm_norm,
            ssm_out,
            layer_state,
            gdn_scratch,
            hidden_a,
            partial_attn_out,
            world,
            model.tp.gdn_kq_replicated(),
        )?;
    }
    if probe {
        for r in 0..cluster.ranks() {
            let p = scratch.per_rank[r].partial_attn_out;
            // Reuse the F32 / F16 probe path with rank-r device bound by reading
            // bytes from rank r's pointer.
            debug_probe_named_rank(scratch, cluster, "post-gdn partial", il, p, r)?;
        }
    }

    // 2+3a. Fused AR-residual + post-attention RMSNorm (TP-3b-i2).
    let post_norm_ptrs: Vec<DevicePtr> = (0..cluster.ranks())
        .map(|r| {
            find_by_suffix(&model.shards[r].layers[il], il, "post_attention_norm.weight")
                .or_else(|_| find_by_suffix(&model.shards[r].layers[il], il, "ffn_norm.weight"))
                .map(|t| t.ptr)
        })
        .collect::<anyhow::Result<_>>()?;
    let mid_norm_ptrs: Vec<DevicePtr> = (0..cluster.ranks())
        .map(|r| {
            scratch.per_rank[r]
                .layer
                .as_ref()
                .map(|l| l.mid_norm_f16)
                .ok_or_else(|| anyhow!("rank {r}: missing LayerForwardScratch"))
        })
        .collect::<anyhow::Result<_>>()?;
    if world > 1 {
        ar_residual_rmsnorm(
            ar,
            scratch,
            cluster,
            world,
            AttnOrFfn::Attn,
            &post_norm_ptrs,
            &mid_norm_ptrs,
            cfg.rms_norm_eps,
        )?;
    } else {
        for r in 0..cluster.ranks() {
            let device = cluster.device(r);
            device.bind()?;
            let stream = device.default_stream();
            let ops = &model.ops[r];
            flambeau_ops::hip::mlp::add_f16(
                &ops,
                stream,
                scratch.per_rank[r].hidden_a,
                scratch.per_rank[r].partial_attn_out,
                scratch.per_rank[r].hidden_a,
                cfg.hidden_size,
            )?;
            rmsnorm_f16(
                &ops,
                stream,
                scratch.per_rank[r].hidden_a,
                post_norm_ptrs[r],
                mid_norm_ptrs[r],
                1,
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )
            .context("dense ffn (TP world=1, gdn-layer) pre-norm")?;
        }
    }

    // B5 bisect — dump mid_norm_f16 (post-AR-attn + post-attn-norm).
    if std::env::var("FLAMBEAU_TP_LAYER0_BISECT").is_ok() && il == 0 {
        for r in 0..cluster.ranks() {
            debug_probe_named_rank(scratch, cluster, "mid_norm_f16", il, mid_norm_ptrs[r], r)?;
        }
        // Also probe hidden_a (the post-AR-attn pre-norm value).
        for r in 0..cluster.ranks() {
            debug_probe_rank_hidden(scratch, cluster, "post-AR-attn hidden_a", il, r)?;
        }
    }

    // 3b. FFN block (dense or MoE — see forward_ffn_block_tp).
    // **TP-7-arch** — same Replicated-MoE override as the full-attn
    // layer: pass `ffn_world = 1` and skip the post-FFN AR.
    let ffn_world = if !cfg.is_dense_ffn() && model.moe_replicated_at(il) {
        1
    } else {
        world
    };
    forward_ffn_block_tp(
        model, scratch, cluster, &mid_norm_ptrs, il, ffn_world, false,
    )?;

    // 4. AR-residual on FFN output (skipped when MoE replicated).
    if ffn_world > 1 {
        ar_residual(ar, scratch, cluster, world, AttnOrFfn::Ffn)?;
    } else {
        for r in 0..cluster.ranks() {
            let device = cluster.device(r);
            device.bind()?;
            let stream = device.default_stream();
            let ops = &model.ops[r];
            flambeau_ops::hip::mlp::add_f16(
                &ops,
                stream,
                scratch.per_rank[r].hidden_a,
                scratch.per_rank[r].partial_ffn_out,
                scratch.per_rank[r].hidden_a,
                cfg.hidden_size,
            )?;
        }
    }
    Ok(())
}



// ---------------------------------------------------------------------------
// **P2.9b-i2-C-wire** — batched-decode driver for TP topology.
// ---------------------------------------------------------------------------

/// Drive `slots.len()` concurrent decode steps through the TP topology
/// with real per-layer batching, returning per-slot `[vocab]` F32 logits.
///
/// Mirrors [`super::pp::forward_decode_batched_pp`] but with all-rank
/// participation per layer + AllReduce. Each slot has its own per-rank
/// KV caches (in `sessions[s].caches[rank][layer]`); the layer body
/// runs once per layer with N inputs/outputs in the per-rank batched
/// scratch.
///
/// `sessions` parallel array — `sessions[s]` is the slot for
/// `BatchSlot { idx: s, .. }` (caller indexes via `BatchSlot.idx`).
///
/// `scratch` is the shared per-rank batched workspace, sized for
/// `>= slots.len()` tokens.
///
/// PP-i2-A1-wire pattern adapted for TP:
/// - Embed N tokens replicated on every rank.
/// - Per layer:
///   * full-attn → `forward_full_attn_layer_decode_batched_tp`,
///     writes per-rank partial; AR sums to replicated `hidden_a`.
///   * GDN → per-slot loop calling `forward_gdn_decode_tp` (recurrent;
///     not batchable across slots without kernel rewrite). Writes
///     partial; AR.
///   * post-attn add+rmsnorm batched at n_tokens=N (replicated).
///   * Per-rank FFN/MoE batched at n_tokens=N (existing prefill TP
///     kernels). AR.
/// - Output head per-slot on `head_rank`.
pub fn forward_decode_batched_tp(
    model: &Qwen3MoETpModel,
    sessions: &mut [&mut crate::tp_sharded::Qwen3MoETpSession],
    cluster: &flambeau_backend_hip::HipCluster,
    ar: &BarP2pAllReduce,
    scratch: &mut ShardedForwardPrefillScratchTp,
    slots: &[super::batched::BatchSlot],
    logits_out: &mut [&mut Vec<f32>],
) -> Result<()> {
    let n = slots.len();
    if n == 0 {
        bail!("forward_decode_batched_tp: empty slot list");
    }
    if sessions.len() != logits_out.len() {
        bail!(
            "forward_decode_batched_tp: sessions({}) != logits_out({})",
            sessions.len(),
            logits_out.len(),
        );
    }
    for s in slots {
        if s.idx >= sessions.len() {
            bail!(
                "forward_decode_batched_tp: BatchSlot.idx {} OOB (n={})",
                s.idx,
                sessions.len()
            );
        }
    }
    let world = cluster.ranks() as u32;
    if world != 1 && world != 2 && world != 4 {
        bail!("forward_decode_batched_tp: world ∈ {{1, 2, 4}} (got {world})");
    }
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;
    let elem_count_l = (n * hidden) as u32;
    let kv_replicated = model.tp.kv_replicated();
    let kq_replicated = model.tp.gdn_kq_replicated();

    // Sanity: per-rank scratch must be sized for >= n tokens.
    for (r, rs) in scratch.per_rank.iter().enumerate() {
        if rs.max_tokens < n {
            bail!(
                "forward_decode_batched_tp: rank {r} scratch.max_tokens={} < n={n}",
                rs.max_tokens
            );
        }
    }

    // 1. Embed N tokens replicated on every rank. token_embd is
    //    Replicated (each rank's shard has the full embedding); each
    //    rank dequant/uploads its own copy bit-identically.
    for r in 0..cluster.ranks() {
        let device = cluster.device(r);
        device.bind()?;
        let stream = device.default_stream();
        let token_embd = &model.shards[r].token_embd;
        for (s_pos, slot) in slots.iter().enumerate() {
            super::io::forward_embed_decode_host(
                device,
                stream,
                token_embd,
                slot.token_id,
                scratch.per_rank[r].hidden_a.offset_bytes(s_pos * row_bytes),
                hidden,
            )?;
        }
    }

    // Per-slot positions (same across ranks since each slot's per-rank
    // caches share the same tail per layer).
    let slot_positions: Vec<usize> = slots.iter().map(|s| s.position).collect();

    // 2. Layer loop.
    let n_layers = cfg.num_layers;
    for il in 0..n_layers {
        let is_full_attn = !cfg.is_recurrent(il);

        // 3a. Per-rank attention forward. Full-attn uses the new
        //     batched-decode TP function; GDN loops slots with the
        //     existing per-token decode kernel.
        for r in 0..cluster.ranks() {
            let device = cluster.device(r);
            device.bind()?;
            let stream = device.default_stream();
            let layer_tensors = &model.shards[r].layers[il];
            let hidden_a = scratch.per_rank[r].hidden_a;
            let partial_attn_out = scratch.per_rank[r].partial_attn_out;
            let layer_scratch = scratch.per_rank[r]
                .layer
                .as_mut()
                .ok_or_else(|| anyhow!("rank {r}: missing LayerPrefillScratch"))?;
            let ops = &model.ops[r];
            if is_full_attn {
                let attn_norm = find_by_suffix(layer_tensors, il, "attn_norm.weight")?;
                let attn_q = find_by_suffix(layer_tensors, il, "attn_q.weight")?;
                let attn_k = find_by_suffix(layer_tensors, il, "attn_k.weight")?;
                let attn_v = find_by_suffix(layer_tensors, il, "attn_v.weight")?;
                let attn_output = find_by_suffix(layer_tensors, il, "attn_output.weight")?;
                let attn_q_norm = find_by_suffix(layer_tensors, il, "attn_q_norm.weight")?;
                let attn_k_norm = find_by_suffix(layer_tensors, il, "attn_k_norm.weight")?;
                let full = layer_scratch
                    .full_attn
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing FullAttnPrefillScratch"))?;
                // Gather per-slot KV caches for THIS layer on THIS rank.
                // SAFETY rationale: each session is unique in `sessions[]`;
                // we form one disjoint &mut to each session's
                // `caches[r][il]`. Use raw-pointer split to satisfy the
                // borrow checker — the gathered guards alias only their
                // respective sessions' per-rank-per-layer cache and don't
                // overlap.
                let sessions_ptr = sessions.as_mut_ptr();
                let mut slot_kv_caches: Vec<
                    &mut flambeau_runtime::KvCache<flambeau_runtime::F16Contig, HipDevice>,
                > = Vec::with_capacity(n);
                for s in 0..n {
                    // SAFETY: indices 0..n are distinct; each session
                    // contributes one cache that doesn't alias others.
                    unsafe {
                        let session_ref: &mut crate::tp_sharded::Qwen3MoETpSession =
                            &mut **sessions_ptr.add(s);
                        match &mut session_ref.caches[r][il] {
                            LayerCache::FullAttn(kv) => slot_kv_caches.push(kv),
                            _ => bail!(
                                "TP batched decode: slot {s} rank {r} layer {il} \
                                 expected FullAttn cache (Q8 KV is V2)"
                            ),
                        }
                    }
                }
                super::attn_tp::forward_full_attn_layer_decode_batched_tp(
                    ops,
                    stream,
                    device,
                    cfg,
                    attn_norm,
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm,
                    attn_k_norm,
                    &mut slot_kv_caches,
                    full,
                    hidden_a,
                    partial_attn_out,
                    &slot_positions,
                    world,
                    kv_replicated,
                )
                .with_context(|| format!("TP batched-decode full-attn layer {il} rank {r}"))?;
            } else {
                // GDN — per-slot loop. Each slot has its own per-rank
                // GdnLayerState in caches[r][il]. The driver shares the
                // single `gdn_decode` scratch on this rank across slots
                // (sequential calls don't conflict on scratch buffers).
                let attn_norm = find_by_suffix(layer_tensors, il, "attn_norm.weight")?;
                let attn_qkv = find_by_suffix(layer_tensors, il, "attn_qkv.weight")?;
                let attn_gate = find_by_suffix(layer_tensors, il, "attn_gate.weight")?;
                let ssm_alpha = find_by_suffix(layer_tensors, il, "ssm_alpha.weight")?;
                let ssm_beta = find_by_suffix(layer_tensors, il, "ssm_beta.weight")?;
                let ssm_a = find_by_suffix(layer_tensors, il, "ssm_a")?;
                let ssm_dt_bias = find_by_suffix(layer_tensors, il, "ssm_dt.bias")?;
                let ssm_conv1d = find_by_suffix(layer_tensors, il, "ssm_conv1d.weight")?;
                let ssm_norm = find_by_suffix(layer_tensors, il, "ssm_norm.weight")?;
                let ssm_out = find_by_suffix(layer_tensors, il, "ssm_out.weight")?;
                // Pull the rank's shared GdnScratch out via a separate
                // borrow to avoid aliasing with layer_scratch above.
                // Safe split-borrow: gdn_decode and layer are disjoint
                // fields of RankForwardPrefillScratchTp.
                let sessions_ptr = sessions.as_mut_ptr();
                let rank_scratch_ptr: *mut RankForwardPrefillScratchTp =
                    &mut scratch.per_rank[r];
                // SAFETY: gdn_decode and layer fields are disjoint within
                // RankForwardPrefillScratchTp; we already borrowed `layer`
                // via `layer_scratch` above and are now reaching into
                // `gdn_decode` through the same struct's raw pointer.
                let gdn = unsafe {
                    (*rank_scratch_ptr)
                        .gdn_decode
                        .as_mut()
                        .ok_or_else(|| anyhow!("rank {r}: missing gdn_decode scratch"))?
                };
                for s in 0..n {
                    // SAFETY: indices 0..n distinct; each session's GDN
                    // state at caches[r][il] is unique.
                    let layer_state = unsafe {
                        let session_ref: &mut crate::tp_sharded::Qwen3MoETpSession =
                            &mut **sessions_ptr.add(s);
                        match &mut session_ref.caches[r][il] {
                            LayerCache::Gdn(state) => state,
                            _ => bail!(
                                "TP batched decode: slot {s} rank {r} layer {il} \
                                 expected Gdn cache"
                            ),
                        }
                    };
                    let slot_x_in = hidden_a.offset_bytes(s * row_bytes);
                    let slot_partial = partial_attn_out.offset_bytes(s * row_bytes);
                    super::gdn_tp::forward_gdn_decode_tp(
                        ops,
                        stream,
                        device,
                        cfg,
                        attn_norm,
                        attn_qkv,
                        attn_gate,
                        ssm_alpha,
                        ssm_beta,
                        ssm_a,
                        ssm_dt_bias,
                        ssm_conv1d,
                        ssm_norm,
                        ssm_out,
                        layer_state,
                        gdn,
                        slot_x_in,
                        slot_partial,
                        world,
                        kq_replicated,
                    )
                    .with_context(|| {
                        format!("TP batched-decode GDN slot {s} layer {il} rank {r}")
                    })?;
                }
            }
        }

        // 3b. AR(hidden_a, partial_attn_out, N*hidden).
        ar_residual_prefill(ar, scratch, cluster, world, elem_count_l, AttnOrFfn::Attn)?;

        // 3c. ffn_norm[N] over hidden_a → mid_norm (per-rank, replicated).
        for r in 0..cluster.ranks() {
            let device = cluster.device(r);
            device.bind()?;
            let stream = device.default_stream();
            let layer_tensors = &model.shards[r].layers[il];
            let ffn_norm = find_by_suffix(layer_tensors, il, "ffn_norm.weight")
                .or_else(|_| find_by_suffix(layer_tensors, il, "post_attention_norm.weight"))?;
            let hidden_a = scratch.per_rank[r].hidden_a;
            let layer_scratch = scratch.per_rank[r]
                .layer
                .as_mut()
                .ok_or_else(|| anyhow!("rank {r}: missing LayerPrefillScratch"))?;
            let mid_norm = layer_scratch.mid_norm_f16;
            let ops = &model.ops[r];
            rmsnorm_f16(
                ops,
                stream,
                hidden_a,
                ffn_norm.ptr,
                mid_norm,
                n,
                hidden,
                cfg.rms_norm_eps,
            )
            .with_context(|| format!("TP batched-decode ffn_norm layer {il}"))?;
        }

        // 3d. Per-rank FFN forward (dense or MoE). Reuse prefill TP
        //     functions — they accept arbitrary n_tokens at fixed
        //     [N, hidden] mid_norm.
        let moe_replicated = !cfg.is_dense_ffn() && model.moe_replicated_at(il);
        let ffn_world = if moe_replicated { 1 } else { world };
        if cfg.is_dense_ffn() {
            for r in 0..cluster.ranks() {
                let device = cluster.device(r);
                device.bind()?;
                let stream = device.default_stream();
                let layer_tensors = &model.shards[r].layers[il];
                let ffn_gate = find_by_suffix(layer_tensors, il, "ffn_gate.weight")?;
                let ffn_up = find_by_suffix(layer_tensors, il, "ffn_up.weight")?;
                let ffn_down = find_by_suffix(layer_tensors, il, "ffn_down.weight")?;
                let partial_ffn_out = scratch.per_rank[r].partial_ffn_out;
                let layer_scratch = scratch.per_rank[r]
                    .layer
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing LayerPrefillScratch"))?;
                let mid_norm = layer_scratch.mid_norm_f16;
                let dense_scratch = layer_scratch
                    .dense_ffn
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing DenseFfnPrefillScratch"))?;
                let ops = &model.ops[r];
                super::dense_ffn_tp::forward_dense_ffn_prefill_tp(
                    ops,
                    stream,
                    cfg,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    dense_scratch,
                    mid_norm,
                    partial_ffn_out,
                    n,
                    world,
                )
                .with_context(|| format!("TP batched-decode dense ffn layer {il}"))?;
            }
        } else {
            let has_shared = cfg.shared_expert_intermediate_size.is_some()
                && std::env::var("FLAMBEAU_TP_SKIP_SHARED").is_err();
            for r in 0..cluster.ranks() {
                let device = cluster.device(r);
                device.bind()?;
                let stream = device.default_stream();
                let layer_tensors = &model.shards[r].layers[il];
                let ffn_gate_inp = find_by_suffix(layer_tensors, il, "ffn_gate_inp.weight")?;
                let ffn_gate_exps = find_by_suffix(layer_tensors, il, "ffn_gate_exps.weight")?;
                let ffn_up_exps = find_by_suffix(layer_tensors, il, "ffn_up_exps.weight")?;
                let ffn_down_exps = find_by_suffix(layer_tensors, il, "ffn_down_exps.weight")?;
                let partial_ffn_out = scratch.per_rank[r].partial_ffn_out;
                let shared_delta_f16 = scratch.per_rank[r]
                    .layer
                    .as_ref()
                    .map(|l| l.shared_delta_f16)
                    .unwrap_or(DevicePtr(0));
                let layer_scratch = scratch.per_rank[r]
                    .layer
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing LayerPrefillScratch"))?;
                let mid_norm = layer_scratch.mid_norm_f16;
                let ops = &model.ops[r];

                {
                    let moe_scratch = layer_scratch
                        .moe
                        .as_mut()
                        .ok_or_else(|| anyhow!("rank {r}: missing MoePrefillScratch"))?;
                    super::moe::forward_router_prefill(
                        ops, stream, cfg, ffn_gate_inp, moe_scratch, mid_norm, n,
                    )
                    .with_context(|| format!("TP batched-decode router layer {il}"))?;
                }
                if has_shared {
                    let shared_w_gate = find_by_suffix(layer_tensors, il, "ffn_gate_shexp.weight")?;
                    let shared_w_up = find_by_suffix(layer_tensors, il, "ffn_up_shexp.weight")?;
                    let shared_w_down = find_by_suffix(layer_tensors, il, "ffn_down_shexp.weight")?;
                    let shared_w_gate_inp =
                        find_by_suffix(layer_tensors, il, "ffn_gate_inp_shexp.weight").ok();
                    let shared_scratch = layer_scratch
                        .shared
                        .as_mut()
                        .ok_or_else(|| anyhow!("rank {r}: missing SharedExpertPrefillScratch"))?;
                    super::moe_tp::forward_shared_expert_prefill_tp(
                        ops,
                        stream,
                        cfg,
                        shared_w_gate,
                        shared_w_up,
                        shared_w_down,
                        shared_w_gate_inp,
                        shared_scratch,
                        mid_norm,
                        shared_delta_f16,
                        n,
                        ffn_world,
                    )
                    .with_context(|| format!("TP batched-decode shared expert layer {il}"))?;
                }
                let moe_scratch = layer_scratch
                    .moe
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing MoePrefillScratch"))?;
                super::moe_tp::forward_moe_ffn_prefill_tp(
                    ops,
                    stream,
                    cfg,
                    ffn_gate_exps,
                    ffn_up_exps,
                    ffn_down_exps,
                    moe_scratch,
                    mid_norm,
                    partial_ffn_out,
                    n,
                    ffn_world,
                )
                .with_context(|| format!("TP batched-decode moe ffn layer {il}"))?;

                if has_shared {
                    flambeau_ops::hip::mlp::add_f16(
                        ops,
                        stream,
                        partial_ffn_out,
                        shared_delta_f16,
                        partial_ffn_out,
                        n * hidden,
                    )
                    .with_context(|| {
                        format!("TP batched-decode + shared expert add layer {il}")
                    })?;
                }
            }
        }

        // 3e. AR-residual on FFN output (or replicated add_f16 fallback).
        if ffn_world > 1 {
            ar_residual_prefill(ar, scratch, cluster, world, elem_count_l, AttnOrFfn::Ffn)?;
        } else {
            for r in 0..cluster.ranks() {
                let device = cluster.device(r);
                device.bind()?;
                let stream = device.default_stream();
                let ops = &model.ops[r];
                flambeau_ops::hip::mlp::add_f16(
                    ops,
                    stream,
                    scratch.per_rank[r].hidden_a,
                    scratch.per_rank[r].partial_ffn_out,
                    scratch.per_rank[r].hidden_a,
                    n * hidden,
                )
                .with_context(|| format!("TP batched-decode replicated ffn add layer {il}"))?;
            }
        }
    }

    // 4. Output head per slot on head_rank.
    let head_rank = scratch.head_rank.0 as usize;
    let head_device = cluster.device(head_rank);
    head_device.bind()?;
    let head_shard = &model.shards[head_rank];
    let output_norm = &head_shard.output_norm;
    let lm_head = head_shard.output.as_ref().unwrap_or(&head_shard.token_embd);
    let head_hidden_a = scratch.per_rank[head_rank].hidden_a;
    let head_scratch = scratch.per_rank[head_rank]
        .output_head
        .as_mut()
        .ok_or_else(|| anyhow!("head rank missing output_head scratch"))?;
    let ops = &model.ops[head_rank];
    for (s_pos, slot) in slots.iter().enumerate() {
        let x_final_row = head_hidden_a.offset_bytes(s_pos * row_bytes);
        super::io::forward_output_head_decode(
            ops,
            head_device.default_stream(),
            cfg,
            output_norm,
            lm_head,
            head_scratch,
            x_final_row,
        )
        .with_context(|| format!("TP batched-decode output head slot {}", slot.idx))?;
        super::io::download_logits_host(
            head_device,
            head_device.default_stream(),
            head_scratch.logits_f32,
            cfg.vocab_size,
            logits_out[slot.idx],
        )
        .with_context(|| format!("TP batched-decode logits download slot {}", slot.idx))?;
    }

    Ok(())
}

