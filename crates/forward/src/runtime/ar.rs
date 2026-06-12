//! AllReduce coordinators. Two backends:
//! * [`ArCoordinator`] / [`ar_sum_f32`] — host-bounce (DtoH → CPU sum
//!   → HtoD). Used as the universal fallback.
//! * [`BarArCoordinator`] / [`bar_ar_sum_f32`] — BAR1-authorised P2P.
//!   Each rank publishes its partial pointer to a shared slab, all
//!   ranks synchronize, then each rank pulls every peer partial into
//!   rank-local scratch via the DMA copy engine and launches its own
//!   `sum_tp{2,4}_f32_rank` kernel reading only that local scratch. No
//!   DtoH/HtoD bytes leave the device. The copy engine is used rather
//!   than an in-kernel BAR1 aperture read because the latter is
//!   non-coherent on gfx906 PCIe P2P — a peer's write can still sit in
//!   its L2 while the aperture maps DRAM, which made greedy decode
//!   non-deterministic at temp=0. See `doc/DETERMINISM_INVESTIGATION.md`.

use anyhow::Result;
use flambeau_backend_hip::{BarP2pAllReduce, HipDevice, HipEvent, HipStream};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_runtime::{
    ArPostAttnRmsNormHookBuffers, ArResidualRmsNormHookBuffers, CollectiveDType, CollectiveError,
    CollectiveResult, DeviceAllReduce, FusedAllReduce,
};
use std::sync::{Arc, Barrier, Mutex};

/// Above this `n_elems`, the AR call rides the host-sync producer
/// ordering path (`Stream::synchronize`) instead of the event-based
/// stream_wait path. Picked to separate decode (`n_tokens=1`,
/// `n_elems = hidden ≤ ~8192`) from prefill (`n_tokens >> 1`,
/// `n_elems = n_tokens * hidden ≫ 32k`).
///
/// Why the gate exists: event-based ordering decouples host from GPU,
/// which is a win when per-call kernel work is tiny (decode) — the
/// host queues many ARs ahead. But at prefill, each AR kernel is
/// large (~3-15 MB) and the host can queue dozens of layers ahead of
/// the GPU; HIP appears to serialize at high queue depth on gfx906,
/// which manifests as a 45% prefill regression on 27B-Q4_0 TP2.
/// Pre-lever's `Stream::synchronize` naturally throttled queue depth.
/// See cert `decode_gap_levers_design_2026_05_19.md` and the bench
/// in `flambeau_v2_vs_legacy_tp2.py` for the bisect.
const EVENT_PATH_MAX_ELEMS: usize = 65_536;

/// One coordinator per stage / per TP cluster. Workers in the same
/// stage share an `Arc<ArCoordinator>` and call `ar_sum` through the
/// TpHooks / HybridHooks callback.
pub struct ArCoordinator {
    pub n_ranks: usize,
    partials: Mutex<Vec<Option<Vec<f32>>>>,
    barrier: Barrier,
}

impl ArCoordinator {
    pub fn new(n_ranks: usize) -> Self {
        Self {
            n_ranks,
            partials: Mutex::new(vec![None; n_ranks]),
            barrier: Barrier::new(n_ranks),
        }
    }
}

/// DtoH this rank's partial → barrier → host sum → barrier → reset
/// (rank 0) → barrier → HtoD the summed result back.
pub fn ar_sum_f32(
    coord: &ArCoordinator,
    rank: usize,
    buf: DevicePtr,
    n_elems: usize,
    device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    let bytes = n_elems * 4;
    let mut host = vec![0.0_f32; n_elems];
    // SAFETY: host has n_elems*4 bytes; buf owns the same.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            buf,
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream)?;
    {
        let mut p = coord.partials.lock().unwrap();
        p[rank] = Some(host);
    }
    coord.barrier.wait();

    let summed: Vec<f32> = {
        let p = coord.partials.lock().unwrap();
        let mut s = p[0].as_ref().unwrap().clone();
        for r in 1..coord.n_ranks {
            let other = p[r].as_ref().unwrap();
            for i in 0..n_elems {
                s[i] += other[i];
            }
        }
        s
    };
    coord.barrier.wait();
    if rank == 0 {
        let mut p = coord.partials.lock().unwrap();
        for r in 0..coord.n_ranks {
            p[r] = None;
        }
    }
    coord.barrier.wait();

    // SAFETY: buf owns bytes; summed is hidden host F32.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            buf,
            DevicePtr(summed.as_ptr() as usize),
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream)?;
    Ok(())
}

/// Boxed callback used by `TpHooks::ar_sum_f32` /
/// `HybridHooks::ar_sum_f32`. Args: `(rank, n_ranks, buf, n_elems,
/// device, stream)` — `(rank, n_ranks)` go unused for the host-bounce
/// path since the coordinator carries them.
pub type ArCallback = Box<
    dyn FnMut(usize, usize, DevicePtr, usize, &HipDevice, &HipStream) -> Result<()> + Send,
>;

/// Build the boxed callback that TpHooks / HybridHooks expect.
pub fn make_ar_callback(coord: Arc<ArCoordinator>, rank: usize) -> ArCallback {
    Box::new(move |_r, _nr, buf, n_elems, dev, st| ar_sum_f32(&coord, rank, buf, n_elems, dev, st))
}

/// BAR1 P2P AllReduce coordinator. Each rank's worker thread calls
/// [`bar_ar_sum_f32`] from inside its `TopologyHooks::ar_sum_f32`
/// callback; the call exchanges per-rank `partial` device pointers,
/// drains the producer stream, then launches `sum_tp{2,4}_f32_rank` on
/// the calling rank's stream. The kernel reads peer pointers via BAR1
/// — no DtoH/HtoD bytes leave the device.
pub struct BarArCoordinator {
    pub bar: Arc<BarP2pAllReduce>,
    partials: Mutex<Vec<Option<DevicePtr>>>,
    /// Per-rank producer-done events, one per rank, each created on
    /// the rank's device. Used by the small-`n` event-based ordering
    /// path so callers can avoid host `Stream::synchronize`; legacy's
    /// `cross_rank_event_barrier` pattern. Large-`n` AR calls keep
    /// host-sync — see `EVENT_PATH_MAX_ELEMS`.
    events: Vec<HipEvent>,
    /// Per-rank "pulls done" events for the DtoD path. After a rank has
    /// copied every peer's partial into local scratch it records its
    /// event; before a rank overwrites its own partial with the AR sum it
    /// waits on every peer's event. This read-all-then-write-all fence is
    /// what the host barriers cannot provide (they order host threads,
    /// not the async GPU copies/sums) — without it a peer's in-place sum
    /// can clobber its partial mid-pull, corrupting ~30% of AR calls.
    pull_events: Vec<HipEvent>,
    barrier: Barrier,
    /// Per-rank receive scratch for the DtoD path, holding `(n_ranks-1)`
    /// contiguous peer-partial slots. Grown on demand; alive for the
    /// coordinator's lifetime.
    recv_staging: Vec<Mutex<Option<PeerDeviceBuffer>>>,
}

impl BarArCoordinator {
    pub fn new(bar: Arc<BarP2pAllReduce>) -> Result<Self> {
        let n = bar.ranks();
        let mut events = Vec::with_capacity(n);
        let mut pull_events = Vec::with_capacity(n);
        for r in 0..n {
            let dev = bar.device_id(r);
            flambeau_backend_hip::bind(dev)?;
            events.push(HipEvent::new(dev)?);
            pull_events.push(HipEvent::new(dev)?);
        }
        Ok(Self {
            bar,
            partials: Mutex::new(vec![None; n]),
            events,
            pull_events,
            barrier: Barrier::new(n),
            recv_staging: (0..n).map(|_| Mutex::new(None)).collect(),
        })
    }

    pub fn ranks(&self) -> usize {
        self.bar.ranks()
    }

    /// Rank-local receive scratch for the DtoD path, grown to `bytes`.
    /// Allocated on `device` (the rank's own device).
    fn ensure_recv_staging(
        &self,
        rank: usize,
        bytes: usize,
        device: &HipDevice,
    ) -> Result<DevicePtr> {
        let mut g = self.recv_staging[rank].lock().unwrap();
        let grow = match g.as_ref() {
            None => true,
            Some(b) => b.bytes < bytes,
        };
        if grow {
            if let Some(old) = g.take() {
                // SAFETY: returned by an earlier alloc on this device;
                // grow only happens when a larger payload arrives, and
                // the publish barrier ordered all prior AR reads of it.
                unsafe { Device::dealloc(device, old.ptr, old.bytes)? };
            }
            device.bind()?;
            let ptr = Device::alloc(device, bytes)?;
            *g = Some(PeerDeviceBuffer { ptr, bytes });
        }
        Ok(g.as_ref().expect("recv_staging just ensured").ptr)
    }
}

/// Producer ordering for the event-based fast path: record this
/// rank's event on its own stream, publish the partial pointer, sync
/// host threads at a barrier, then queue a `stream_wait` against every
/// peer event so the upcoming BAR1 launch sees committed peer writes.
/// No host `Stream::synchronize` — CPU stays decoupled from GPU.
fn ar_publish_with_events(
    coord: &BarArCoordinator,
    rank: usize,
    partial: DevicePtr,
    stream: &HipStream,
) -> Result<Vec<DevicePtr>> {
    coord.events[rank].record(stream)?;
    {
        let mut p = coord.partials.lock().unwrap();
        p[rank] = Some(partial);
    }
    coord.barrier.wait();
    let snap: Vec<DevicePtr> = {
        let p = coord.partials.lock().unwrap();
        (0..coord.ranks())
            .map(|r| p[r].expect("BarArCoordinator: peer pointer unpublished"))
            .collect()
    };
    for r in 0..coord.ranks() {
        if r != rank {
            coord.events[r].stream_wait(stream)?;
        }
    }
    Ok(snap)
}

/// Producer ordering for the host-sync path: drain own producer
/// stream so the partial is committed in HBM before any peer reads
/// it via BAR1, then publish + host barrier + snap peers. Matches
/// pre-Lever-3 semantics; used at prefill-scale `n_elems` to avoid
/// the high-queue-depth path that hurts gfx906.
fn ar_publish_with_host_sync(
    coord: &BarArCoordinator,
    rank: usize,
    partial: DevicePtr,
    stream: &HipStream,
) -> Result<Vec<DevicePtr>> {
    flambeau_core::Stream::synchronize(stream)?;
    {
        let mut p = coord.partials.lock().unwrap();
        p[rank] = Some(partial);
    }
    coord.barrier.wait();
    let snap: Vec<DevicePtr> = {
        let p = coord.partials.lock().unwrap();
        (0..coord.ranks())
            .map(|r| p[r].expect("BarArCoordinator: peer pointer unpublished"))
            .collect()
    };
    Ok(snap)
}

/// Slab-reset epilogue shared by both publish paths: barrier, rank 0
/// clears the partials slab, barrier again so the next AR call sees
/// `None` slots.
fn ar_epilogue(coord: &BarArCoordinator, rank: usize) {
    coord.barrier.wait();
    if rank == 0 {
        let mut p = coord.partials.lock().unwrap();
        for r in 0..coord.ranks() {
            p[r] = None;
        }
    }
    coord.barrier.wait();
}

/// Fused BAR1 AR + post-attn / post-ffn rmsnorm + residual-add.
/// Collapses the gemma4 2-launch sequence (`ar_sum_f32` +
/// `rmsnorm_f32_to_f16_add_residual`) into one launch. F32 over
/// BAR1 (same payload as `bar_ar_sum_f32` — measured null for F16
/// payload halving in S1; the launch-count win is what matters).
/// `n_rows` blocks of 256 threads; per-row hidden `n` must satisfy
/// `n <= 8192`. `resid_out` must NOT alias `resid_in`.
/// Per-call knobs for [`bar_ar_postattn_residual_rmsnorm_f32_to_f16`].
#[derive(Copy, Clone, Debug)]
pub struct BarArPostAttnNormParams {
    pub n_rows: usize,
    pub n: usize,
    pub eps: f32,
}

pub fn bar_ar_postattn_residual_rmsnorm_f32_to_f16(
    coord: &BarArCoordinator,
    rank: usize,
    bufs: crate::core::ArPostAttnRmsNormHookBuffers,
    params: BarArPostAttnNormParams,
    device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    let BarArPostAttnNormParams { n_rows, n, eps } = params;
    let crate::core::ArPostAttnRmsNormHookBuffers {
        proj_local_f32,
        post_norm_w_f16,
        resid_in_f16,
        resid_out_f16,
    } = bufs;
    let n_ranks = coord.ranks();
    if n_ranks == 1 {
        anyhow::bail!(
            "bar_ar_postattn_residual_rmsnorm_f32_to_f16: TP=1 caller — fall back \
             to the split-path manually"
        );
    }
    let total_elems = n_rows * n;
    let shape = flambeau_backend_hip::ArPostAttnNormShape {
        n_rows: n_rows as u32,
        n: n as u32,
    };
    // proj partials are F32 (4 bytes/elem). Pull peers into local
    // scratch; the kernel sums proj_local + proj_peer(s) coherently.
    let PeerPulls { peers, local_peers } =
        ar_publish_pull(coord, rank, proj_local_f32, total_elems, 4, device, stream)?;
    // SAFETY: proj_local + local peer copies + resid/weight are
    // rank-local; resid_out not aliased to resid_in (caller contract).
    unsafe {
        match n_ranks {
            2 => coord.bar.postattn_residual_rmsnorm_f32_to_f16_tp2_rank(
                rank,
                flambeau_backend_hip::ArPostAttnNormRankBuffersTp2 {
                    proj_local: peers[rank],
                    peer: local_peers[0],
                    post_norm_w: post_norm_w_f16,
                    resid_in: resid_in_f16,
                    resid_out: resid_out_f16,
                },
                shape,
                eps,
                stream,
            )?,
            4 => {
                let peer3 = [local_peers[0], local_peers[1], local_peers[2]];
                coord.bar.postattn_residual_rmsnorm_f32_to_f16_tp4_rank(
                    rank,
                    flambeau_backend_hip::ArPostAttnNormRankBuffersTp4 {
                        proj_local: peers[rank],
                        peers: peer3,
                        post_norm_w: post_norm_w_f16,
                        resid_in: resid_in_f16,
                        resid_out: resid_out_f16,
                    },
                    shape,
                    eps,
                    stream,
                )?
            }
            other => anyhow::bail!("BarArCoordinator: unsupported n_ranks={other}"),
        }
    }
    ar_epilogue(coord, rank);
    Ok(())
}

/// BAR1 P2P AR-sum (F16 payload). Halves the cross-rank traffic vs
/// [`bar_ar_sum_f32`] at the price of F16-saturating any
/// partial-sum element above ±65504. Caller is responsible for the
/// safety predicate (e.g. `output_proj_safe_for_f16_ar` on gemma4).
pub fn bar_ar_sum_f16(
    coord: &BarArCoordinator,
    rank: usize,
    buf: DevicePtr,
    n_elems: usize,
    device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    let n_ranks = coord.ranks();
    if n_ranks == 1 {
        return Ok(());
    }
    // Coherent F16 sum: pull each peer partial (2 bytes/elem) into
    // local scratch + flush, then the existing kernel reads local.
    let PeerPulls { peers, local_peers } =
        ar_publish_pull(coord, rank, buf, n_elems, 2, device, stream)?;
    // SAFETY: all pointers rank-local, valid for n_elems F16.
    unsafe {
        match n_ranks {
            2 => coord.bar.sum_tp2_rank(
                rank,
                peers[rank],
                local_peers[0],
                n_elems as u32,
                stream,
            )?,
            4 => {
                let p3 = [local_peers[0], local_peers[1], local_peers[2]];
                coord
                    .bar
                    .sum_tp4_rank(rank, peers[rank], p3, n_elems as u32, stream)?
            }
            other => anyhow::bail!("bar_ar_sum_f16: unsupported n_ranks={other}"),
        }
    }
    ar_epilogue(coord, rank);
    Ok(())
}

/// BAR1 P2P fused AR + residual-add. `partial_f16` is the
/// rank-local F16 partial (caller has cast from F32 if needed).
/// `residual_inout_f16` is the rank-local residual; updated in place
/// to `residual + Σ peer-partials`. TP=2 only.
pub fn bar_ar_residual_f16(
    coord: &BarArCoordinator,
    rank: usize,
    residual_inout_f16: DevicePtr,
    partial_f16: DevicePtr,
    n_elems: usize,
    device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    let n_ranks = coord.ranks();
    if n_ranks == 1 {
        // SD path: caller must do residual_add separately. We don't
        // touch resid here so the no-fuse fallback semantics hold.
        return Ok(());
    }
    if n_ranks != 2 {
        anyhow::bail!("bar_ar_residual_f16: only TP=2 supported (got {n_ranks})");
    }
    let PeerPulls { peers, local_peers } =
        ar_publish_pull(coord, rank, partial_f16, n_elems, 2, device, stream)?;
    // Keep canonical (rank0, rank1) partial order so both ranks reduce
    // the same two physical partials in the same slots: own partial in
    // this rank's slot, the staged peer copy in the peer's slot.
    let canon0 = if rank == 0 { peers[0] } else { local_peers[0] };
    let canon1 = if rank == 1 { peers[1] } else { local_peers[0] };
    // SAFETY: hidden + both partials are rank-local, valid for n_elems F16.
    unsafe {
        coord.bar.residual_tp2_rank(
            rank,
            residual_inout_f16,
            canon0,
            canon1,
            n_elems as u32,
            stream,
        )?;
    }
    ar_epilogue(coord, rank);
    Ok(())
}

/// BAR1 P2P fused AR + residual-add + RMSNorm. `partial_f16` is the
/// rank-local F16 partial; `residual_inout_f16` is the per-rank
/// residual (updated in-place to `residual + Σ peers`); `out_norm`
/// receives the rmsnormed result. TP=2 only.
pub fn bar_ar_residual_rmsnorm_f16(
    coord: &BarArCoordinator,
    rank: usize,
    bufs: crate::core::ArResidualRmsNormHookBuffers,
    n_elems: usize,
    eps: f32,
    device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    let crate::core::ArResidualRmsNormHookBuffers {
        residual_inout: residual_inout_f16,
        partial_f16,
        rms_weight,
        out_norm,
    } = bufs;
    let n_ranks = coord.ranks();
    if n_ranks != 2 {
        anyhow::bail!("bar_ar_residual_rmsnorm_f16: only TP=2 supported (got {n_ranks})");
    }
    let PeerPulls { peers, local_peers } =
        ar_publish_pull(coord, rank, partial_f16, n_elems, 2, device, stream)?;
    let canon0 = if rank == 0 { peers[0] } else { local_peers[0] };
    let canon1 = if rank == 1 { peers[1] } else { local_peers[0] };
    // SAFETY: hidden/weights/out + both partials are rank-local.
    unsafe {
        coord.bar.residual_rmsnorm_tp2_rank(
            rank,
            flambeau_backend_hip::ArResidualRmsNormRankBuffers {
                hidden: residual_inout_f16,
                partial_canonical_rank0: canon0,
                partial_canonical_rank1: canon1,
                rms_weight,
                out_norm,
            },
            n_elems as u32,
            eps,
            stream,
        )?;
    }
    ar_epilogue(coord, rank);
    Ok(())
}

/// Coherent peer exchange shared by every BAR1 AR path. Returns
/// rank-local pointers to every rank's partial so the caller's kernel
/// reads only coherent local memory: `peers` is the published-pointer
/// snapshot in rank order (`peers[rank] == buf`, this rank's own
/// partial); `local_peers` are copy-engine staging copies of each peer's
/// partial, in ascending-peer-rank order (the `r == rank` slot skipped).
///
/// Steps: on the event path, L2→DRAM flush `buf` so the peers'
/// copy-engine pull is coherent (the host-sync path drains the stream
/// instead); publish + order; copy-engine pull each peer partial into
/// rank-local staging; then a read-all-then-write-all pull-done fence.
/// `peer_bytes_per_elem` (2 for F16, 4 for F32) sizes the flush word
/// count and the pull/staging bytes — a single source of truth so no
/// path can mis-size the F16 case.
struct PeerPulls {
    peers: Vec<DevicePtr>,
    local_peers: Vec<DevicePtr>,
}

fn ar_publish_pull(
    coord: &BarArCoordinator,
    rank: usize,
    buf: DevicePtr,
    n_elems: usize,
    peer_bytes_per_elem: usize,
    device: &HipDevice,
    stream: &HipStream,
) -> Result<PeerPulls> {
    let n_ranks = coord.ranks();
    let n_bytes = n_elems * peer_bytes_per_elem;
    let event_path = n_elems <= EVENT_PATH_MAX_ELEMS;
    if event_path {
        // The event path orders peers after this rank's producer event but
        // does not flush the producer's writes from L2 to DRAM; a peer's
        // copy-engine pull then sources a timing-dependent (boot-varying)
        // value. Flush `buf` to DRAM before the producer event is recorded
        // so the pull is coherent. `l2_flush` counts u32 words — for F16
        // (2 bytes/elem) that is `n_elems.div_ceil(2)`, not `n_elems`;
        // flushing too few words leaves the buffer's upper half stale in L2
        // and silently reintroduces non-determinism. See
        // `doc/DETERMINISM_INVESTIGATION.md`.
        // SAFETY: `buf` is this rank's partial on `device`; `stream`
        // belongs to that device, ordered after the producer.
        let n_words = (n_bytes as u32).div_ceil(4);
        unsafe { coord.bar.l2_flush(rank, buf, n_words, stream)? };
    }
    let peers = if event_path {
        ar_publish_with_events(coord, rank, buf, stream)?
    } else {
        ar_publish_with_host_sync(coord, rank, buf, stream)?
    };
    let staging = coord.ensure_recv_staging(rank, (n_ranks - 1) * n_bytes, device)?;
    // Pull every peer partial into local scratch on this rank's stream.
    // The publish helper already ordered `stream` after each peer's
    // producer (event or host-sync drain), so the copy reads a committed
    // partial; the copy engine sources it coherently.
    let mut local_peers: Vec<DevicePtr> = Vec::with_capacity(n_ranks - 1);
    let mut off = 0usize;
    for (r, &peer_ptr) in peers.iter().enumerate() {
        if r == rank {
            continue;
        }
        let dst = DevicePtr(staging.0 + off);
        // SAFETY: dst owns n_bytes within the (n_ranks-1)*n_bytes staging
        // slab on this device; peer_ptr owns n_bytes on device r; peer
        // access authorised at cluster bring-up; stream belongs to `device`.
        unsafe {
            device.memcpy_peer_in_async(stream, dst, peer_ptr, coord.bar.device_id(r), n_bytes)?;
        }
        local_peers.push(dst);
        off += n_bytes;
    }
    // Read-all-then-write-all fence. Strictly required for the in-place sum
    // paths (a peer's `buf += peer` would clobber a partial another rank is
    // mid-pull); conservative for the residual/rmsnorm/postattn paths (they
    // write `hidden`/`out`, leaving the published partial read-only) but
    // kept for one uniform coherent core.
    coord.pull_events[rank].record(stream)?;
    coord.barrier.wait();
    for (r, ev) in coord.pull_events.iter().enumerate() {
        if r != rank {
            ev.stream_wait(stream)?;
        }
    }
    Ok(PeerPulls { peers, local_peers })
}

/// BAR1 P2P AllReduce-sum (F32). Pulls each peer partial into rank-local
/// scratch via the DMA copy engine, then runs the
/// `sum_tp{2,4}_f32_rank` kernel with the peer arg pointed at that local
/// scratch — so the kernel reads only coherent rank-local memory. The
/// copy engine sources peer bytes coherently where an in-kernel BAR1
/// aperture read is stale on gfx906 PCIe P2P, so this is bit-deterministic
/// at temp=0. Same on-device byte volume as a direct BAR1 read (one peer
/// copy per rank), no host bounce. See `doc/DETERMINISM_INVESTIGATION.md`.
pub fn bar_ar_sum_f32(
    coord: &BarArCoordinator,
    rank: usize,
    buf: DevicePtr,
    n_elems: usize,
    device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    let n_ranks = coord.ranks();
    if n_ranks == 1 {
        return Ok(());
    }
    let PeerPulls { peers, local_peers } =
        ar_publish_pull(coord, rank, buf, n_elems, 4, device, stream)?;
    // Local partial + local peer copies. Same-stream ordering serializes
    // the sum after the pulls; the kernel reads only rank-local memory.
    // SAFETY: all pointers are rank-local DevicePtrs valid for n_elems F32.
    unsafe {
        match n_ranks {
            2 => coord.bar.sum_tp2_f32_rank(
                rank,
                peers[rank],
                local_peers[0],
                n_elems as u32,
                stream,
            )?,
            4 => {
                let p3 = [local_peers[0], local_peers[1], local_peers[2]];
                coord
                    .bar
                    .sum_tp4_f32_rank(rank, peers[rank], p3, n_elems as u32, stream)?
            }
            other => anyhow::bail!("bar_ar_sum_f32: unsupported n_ranks={other}"),
        }
    }
    ar_epilogue(coord, rank);
    Ok(())
}

/// Build the boxed callback that TpHooks / HybridHooks expect, with
/// the BAR1 P2P backend (coherent copy-engine AR-sum).
pub fn make_bar_ar_callback(coord: Arc<BarArCoordinator>, rank: usize) -> ArCallback {
    Box::new(move |_r, _nr, buf, n_elems, dev, st| {
        bar_ar_sum_f32(&coord, rank, buf, n_elems, dev, st)
    })
}

fn ar_device_err(e: anyhow::Error, backend: &'static str) -> CollectiveError {
    CollectiveError::Device {
        backend,
        ctx: "all_reduce_sum",
        message: format!("{e:#}"),
    }
}

/// Per-rank handle binding a shared [`BarArCoordinator`] to one rank so it
/// satisfies the [`DeviceAllReduce`] seam — the device-pointer analog of the
/// byte-buffer `AllReduce`. The seam the generic forward engine threads its
/// AR through once it is backend-generic (A2.4 / C3).
pub struct BarArRank {
    pub coord: Arc<BarArCoordinator>,
    pub rank: usize,
}

impl DeviceAllReduce for BarArRank {
    type Device = HipDevice;

    fn all_reduce_sum(
        &self,
        buf: DevicePtr,
        n_elems: usize,
        dtype: CollectiveDType,
        device: &HipDevice,
        stream: &HipStream,
    ) -> CollectiveResult<()> {
        match dtype {
            CollectiveDType::F32 => {
                bar_ar_sum_f32(&self.coord, self.rank, buf, n_elems, device, stream)
            }
            CollectiveDType::F16 => {
                bar_ar_sum_f16(&self.coord, self.rank, buf, n_elems, device, stream)
            }
        }
        .map_err(|e| ar_device_err(e, "hip-bar1"))
    }
}

/// Host-bounce sibling of [`BarArRank`] for clusters without a fully-
/// connected peer matrix. F32 only — the F16 payload needs the BAR1 path.
pub struct HostArRank {
    pub coord: Arc<ArCoordinator>,
    pub rank: usize,
}

impl DeviceAllReduce for HostArRank {
    type Device = HipDevice;

    fn all_reduce_sum(
        &self,
        buf: DevicePtr,
        n_elems: usize,
        dtype: CollectiveDType,
        device: &HipDevice,
        stream: &HipStream,
    ) -> CollectiveResult<()> {
        match dtype {
            CollectiveDType::F32 => ar_sum_f32(&self.coord, self.rank, buf, n_elems, device, stream)
                .map_err(|e| ar_device_err(e, "host-bounce")),
            CollectiveDType::F16 => Err(CollectiveError::Device {
                backend: "host-bounce",
                ctx: "all_reduce_sum",
                message: "host-bounce AllReduce is F32-only; F16 requires the BAR1 path".to_string(),
            }),
        }
    }
}

impl FusedAllReduce for BarArRank {
    fn supports_ar_sum_f16(&self) -> bool {
        matches!(self.coord.ranks(), 2 | 4)
    }

    fn ar_sum_f16(
        &self,
        buf: DevicePtr,
        n_elems: usize,
        device: &HipDevice,
        stream: &HipStream,
    ) -> CollectiveResult<()> {
        bar_ar_sum_f16(&self.coord, self.rank, buf, n_elems, device, stream)
            .map_err(|e| ar_device_err(e, "hip-bar1"))
    }

    fn supports_ar_residual_f16(&self) -> bool {
        self.coord.ranks() == 2
    }

    fn ar_residual_f16(
        &self,
        residual_inout: DevicePtr,
        partial_f16: DevicePtr,
        n_elems: usize,
        device: &HipDevice,
        stream: &HipStream,
    ) -> CollectiveResult<()> {
        bar_ar_residual_f16(
            &self.coord,
            self.rank,
            residual_inout,
            partial_f16,
            n_elems,
            device,
            stream,
        )
        .map_err(|e| ar_device_err(e, "hip-bar1"))
    }

    fn supports_ar_residual_rmsnorm_f16(&self) -> bool {
        self.coord.ranks() == 2
    }

    fn ar_residual_rmsnorm_f16(
        &self,
        bufs: ArResidualRmsNormHookBuffers,
        n_elems: usize,
        eps: f32,
        device: &HipDevice,
        stream: &HipStream,
    ) -> CollectiveResult<()> {
        bar_ar_residual_rmsnorm_f16(&self.coord, self.rank, bufs, n_elems, eps, device, stream)
            .map_err(|e| ar_device_err(e, "hip-bar1"))
    }

    fn supports_ar_postattn_residual_rmsnorm_f32_to_f16(&self) -> bool {
        matches!(self.coord.ranks(), 2 | 4)
    }

    fn ar_postattn_residual_rmsnorm_f32_to_f16(
        &self,
        bufs: ArPostAttnRmsNormHookBuffers,
        n_rows: usize,
        n: usize,
        eps: f32,
        device: &HipDevice,
        stream: &HipStream,
    ) -> CollectiveResult<()> {
        bar_ar_postattn_residual_rmsnorm_f32_to_f16(
            &self.coord,
            self.rank,
            bufs,
            BarArPostAttnNormParams { n_rows, n, eps },
            device,
            stream,
        )
        .map_err(|e| ar_device_err(e, "hip-bar1"))
    }
}

/// Per-edge handoff slot between two consecutive PP stages
/// (producer rank → consumer rank). Holds a destination buffer
/// allocated on the **consumer's** device + an event handshake.
///
/// Producer's `peer_send`:
///   * grows `dst` on demand to cover `n_tokens * hidden * 2` bytes,
///   * enqueues `hipMemcpyPeerAsync(dst, dst_device, src, src_device,
///     bytes, producer_stream)` — direct device-to-device, no host
///     bounce. This is the same pattern llama.cpp uses in
///     `ggml_backend_cuda_cpy_tensor_async`.
///   * records `send_done` on the producer stream after the copy.
///
/// Consumer's `peer_recv`:
///   * stream-waits on `send_done` (driver-side),
///   * returns a `Tensor` view of `dst.ptr` — no second copy needed,
///     because `dst` already lives on the consumer's device.
///
/// `buf` (the host bounce vec) is retained as a fallback used by the
/// `HybStage` path which hasn't been migrated to the device path yet
/// — see TODO in `engine.rs`.
pub struct PeerSlot {
    pub buf: Mutex<Vec<half::f16>>,
    pub send_done: Mutex<Option<HipEvent>>,
    /// Pre-allocated device buffer on the CONSUMER's device. Lazy:
    /// `None` until first `peer_send` allocates with the actual
    /// hidden×n_tokens size.
    pub dst: Mutex<Option<PeerDeviceBuffer>>,
    /// Consumer device ID — used by the producer to call
    /// `hipMemcpyPeerAsync`. `None` for the hybrid path (still on
    /// host bounce).
    pub consumer_device_id: Option<i32>,
    /// Consumer's HipDevice handle, kept alive so allocations made
    /// on it remain valid for the slot's lifetime. Producer doesn't
    /// touch this — it only uses `consumer_device_id` for the peer
    /// copy call. The Arc keeps `dst` valid.
    pub consumer_device: Option<Arc<HipDevice>>,
}

pub struct PeerDeviceBuffer {
    pub ptr: DevicePtr,
    pub bytes: usize,
}

pub type PeerBuffer = Arc<PeerSlot>;

/// Build a peer slot for the **legacy / hybrid** path — host-bounce
/// only, no device buffer. Used by [`launch_hybrid`] and any caller
/// that hasn't migrated to the per-edge BAR1 P2P handoff.
pub fn new_peer_buffer(hidden: usize) -> PeerBuffer {
    Arc::new(PeerSlot {
        buf: Mutex::new(vec![half::f16::ZERO; hidden]),
        send_done: Mutex::new(None),
        dst: Mutex::new(None),
        consumer_device_id: None,
        consumer_device: None,
    })
}

/// Build a per-edge PP slot wired for direct cross-device peer copy.
/// `consumer_device_id` is the device the buffer will be allocated on.
/// The actual allocation is deferred to the first `peer_send`, when
/// the byte size becomes known.
///
/// # Errors
/// Returns a [`HipDevice::new`] failure if `consumer_device_id` is
/// invalid (out of range, or the underlying HIP context init fails).
pub fn new_peer_edge(consumer_device_id: i32) -> anyhow::Result<PeerBuffer> {
    let dev = HipDevice::new(consumer_device_id)?;
    Ok(Arc::new(PeerSlot {
        buf: Mutex::new(Vec::new()),
        send_done: Mutex::new(None),
        dst: Mutex::new(None),
        consumer_device_id: Some(consumer_device_id),
        consumer_device: Some(Arc::new(dev)),
    }))
}

/// Build a per-edge slot with `dst` pre-allocated to `max_bytes` on
/// the consumer's device. Eliminates the alloc-on-first-send race
/// when both ranks of a stage start in parallel and the receiver
/// reaches `peer_recv` before the producer's lazy `peer_send` alloc.
/// Used by `launch_hybrid` since stage workers wake up concurrently.
///
/// # Errors
/// Returns errors from [`HipDevice::new`] or [`Device::alloc`].
pub fn new_peer_edge_prealloc(
    consumer_device_id: i32,
    max_bytes: usize,
) -> anyhow::Result<PeerBuffer> {
    let dev = HipDevice::new(consumer_device_id)?;
    dev.bind()?;
    let ptr = flambeau_core::Device::alloc(&dev, max_bytes)?;
    Ok(Arc::new(PeerSlot {
        buf: Mutex::new(Vec::new()),
        send_done: Mutex::new(None),
        dst: Mutex::new(Some(PeerDeviceBuffer {
            ptr,
            bytes: max_bytes,
        })),
        consumer_device_id: Some(consumer_device_id),
        consumer_device: Some(Arc::new(dev)),
    }))
}
