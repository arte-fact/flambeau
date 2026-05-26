//! AllReduce coordinators. Two backends:
//! * [`ArCoordinator`] / [`ar_sum_f32`] — host-bounce (DtoH → CPU sum
//!   → HtoD). Used as the universal fallback.
//! * [`BarArCoordinator`] / [`bar_ar_sum_f32`] — BAR1 P2P. Each rank
//!   publishes its partial pointer to a shared slab, all ranks
//!   synchronize, then each rank launches its own
//!   `sum_tp{2,4}_f32_rank` kernel on its own stream. ~100× less
//!   per-call overhead than the host bounce on TP2 because there
//!   are no DtoH/HtoD bytes — just BAR1 reads inside the kernel.

use anyhow::Result;
use flambeau_backend_hip::{BarP2pAllReduce, HipDevice, HipEvent, HipStream};
use flambeau_core::{CopyDirection, Device, DevicePtr};
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

/// Build the boxed callback that TpHooks / HybridHooks expect.
pub fn make_ar_callback(
    coord: Arc<ArCoordinator>,
    rank: usize,
) -> Box<dyn FnMut(usize, usize, DevicePtr, usize, &HipDevice, &HipStream) -> Result<()> + Send> {
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
    barrier: Barrier,
}

impl BarArCoordinator {
    pub fn new(bar: Arc<BarP2pAllReduce>) -> Result<Self> {
        let n = bar.ranks();
        let mut events = Vec::with_capacity(n);
        for r in 0..n {
            let dev = bar.device_id(r);
            flambeau_backend_hip::bind(dev)?;
            events.push(HipEvent::new(dev)?);
        }
        Ok(Self {
            bar,
            partials: Mutex::new(vec![None; n]),
            events,
            barrier: Barrier::new(n),
        })
    }

    pub fn ranks(&self) -> usize {
        self.bar.ranks()
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

/// BAR1 P2P AR-sum. Picks event-based ordering for decode-shape
/// (small `n_elems`) and host-sync for prefill-shape (large
/// `n_elems`) — see `EVENT_PATH_MAX_ELEMS`.
pub fn bar_ar_sum_f32(
    coord: &BarArCoordinator,
    rank: usize,
    buf: DevicePtr,
    n_elems: usize,
    _device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    let n_ranks = coord.ranks();
    if n_ranks == 1 {
        return Ok(());
    }
    let peers = if n_elems <= EVENT_PATH_MAX_ELEMS {
        ar_publish_with_events(coord, rank, buf, stream)?
    } else {
        ar_publish_with_host_sync(coord, rank, buf, stream)?
    };
    // SAFETY: partials are pool-owned DevicePtrs alive for the request;
    // producer ordering held by the publish helper; each rank launches
    // on its own stream; BarP2pAllReduce validates per-rank cluster
    // device.
    unsafe {
        match n_ranks {
            2 => coord.bar.sum_tp2_f32_rank(
                rank,
                peers[rank],
                peers[1 - rank],
                n_elems as u32,
                stream,
            )?,
            4 => {
                let peer3 = [
                    peers[(rank + 1) % 4],
                    peers[(rank + 2) % 4],
                    peers[(rank + 3) % 4],
                ];
                coord
                    .bar
                    .sum_tp4_f32_rank(rank, peers[rank], peer3, n_elems as u32, stream)?
            }
            other => anyhow::bail!("BarArCoordinator: unsupported n_ranks={other}"),
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
    _device: &HipDevice,
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
    let peers = if n_elems <= EVENT_PATH_MAX_ELEMS {
        ar_publish_with_events(coord, rank, partial_f16, stream)?
    } else {
        ar_publish_with_host_sync(coord, rank, partial_f16, stream)?
    };
    // SAFETY: partials are pool-owned F16 alive for the request;
    // producer ordering held by the publish helper.
    unsafe {
        coord.bar.residual_tp2_rank(
            rank,
            residual_inout_f16,
            peers[0],
            peers[1],
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
#[allow(clippy::too_many_arguments)]
pub fn bar_ar_residual_rmsnorm_f16(
    coord: &BarArCoordinator,
    rank: usize,
    residual_inout_f16: DevicePtr,
    partial_f16: DevicePtr,
    rms_weight: DevicePtr,
    out_norm: DevicePtr,
    n_elems: usize,
    eps: f32,
    _device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    let n_ranks = coord.ranks();
    if n_ranks != 2 {
        anyhow::bail!("bar_ar_residual_rmsnorm_f16: only TP=2 supported (got {n_ranks})");
    }
    let peers = if n_elems <= EVENT_PATH_MAX_ELEMS {
        ar_publish_with_events(coord, rank, partial_f16, stream)?
    } else {
        ar_publish_with_host_sync(coord, rank, partial_f16, stream)?
    };
    // SAFETY: partials are pool-owned F16 alive for the request;
    // producer ordering held by the publish helper.
    unsafe {
        coord.bar.residual_rmsnorm_tp2_rank(
            rank,
            residual_inout_f16,
            peers[0],
            peers[1],
            rms_weight,
            out_norm,
            n_elems as u32,
            eps,
            stream,
        )?;
    }
    ar_epilogue(coord, rank);
    Ok(())
}

/// Build the boxed callback that TpHooks / HybridHooks expect, with
/// the BAR1 P2P backend.
pub fn make_bar_ar_callback(
    coord: Arc<BarArCoordinator>,
    rank: usize,
) -> Box<dyn FnMut(usize, usize, DevicePtr, usize, &HipDevice, &HipStream) -> Result<()> + Send> {
    Box::new(move |_r, _nr, buf, n_elems, dev, st| {
        bar_ar_sum_f32(&coord, rank, buf, n_elems, dev, st)
    })
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
