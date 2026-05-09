//! Multi-device HIP cluster — PP host-bounce + TP BAR1 peer-access primitives.
//! A `HipCluster` owns one `HipDevice` per rank plus a pinned host
//! bounce buffer per rank. The PP-side primitive is `peer_copy_via_host`:
//! a **CPU-bounce peer copy** that stages a DeviceToHost transfer on the
//! source rank and a HostToDevice transfer on the destination rank through
//! pinned host memory. The TP-side primitive () is the on-construction
//! probe of `hipDeviceCanAccessPeer` + authorisation via
//! `hipDeviceEnablePeerAccess`; the resulting matrix is exposed via
//! [`HipCluster::peer_access_matrix`] and consumed by the BAR1
//! AllReduce path.
//! Why not direct `hipMemcpyPeerAsync`? The V1 target rig is PCIe-only
//! (no xGMI / no NVLink / no kernel `CONFIG_HSA_AMD_P2P`). On that
//! topology, direct peer copies appear to submit successfully but the
//! stream ends up in an unsync'able state (candle's
//! `hip_backend/cluster.rs` documents the same behaviour on ROCm
//! 7.1.1 + gfx906). The host-bounce fallback is boring and reliable.
//! Bandwidth envelope (candle measured on MI50 Gen3 x16): single-chunk
//! pinned-host bounce hits ~6.75 GB/s. Pipelined multi-chunk with events
//! and a dedicated DtoH stream climbs to ~10–11 GB/s. B ships the
//! single-chunk path — it covers every stage-boundary transfer on the
//! decode/prefill hot path (hidden-state F16 at L ≤ 128 tokens ≤ 512 KB
//! per hop), which sits comfortably below the 4 MiB chunk threshold the
//! pipelined path was designed to exceed. Pipelined copy lands later if
//! we ever shuffle bulk (weight re-sharding, KV migration) between ranks.

use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::sys::{
    error_string, hipDeviceCanAccessPeer, hipDeviceEnablePeerAccess, hipHostFree, hipHostMalloc,
    hipMemcpyAsync, hipMemcpyKind, hipStreamSynchronize, HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED,
    HIP_HOST_MALLOC_PORTABLE, HIP_SUCCESS,
};
use crate::{HipDevice, HipStream};
use flambeau_core::{Device, DeviceError, DevicePtr, DeviceResult};

/// One entry per rank: the pinned host bounce buffer.
/// C3-refactor: the hot path (`peer_copy_via_host`) reads `ptr` + `bytes`
/// lock-free via atomics. The grow path is serialised by `grow_lock` to
/// prevent concurrent reallocations, but after session warmup the buffer
/// never grows again — `reserve_bounce_capacity` should be called at
/// session init so the first real copy hits the lock-free fast path.
struct RankBounce {
    /// Pinned host pointer. `null` until first grow.
    ptr: AtomicPtr<c_void>,
    /// Current capacity in bytes.
    bytes: AtomicUsize,
    /// Serialises grow-path writers. Uncontended in steady state — readers
    /// never touch this.
    grow_lock: Mutex<()>,
}

impl std::fmt::Debug for RankBounce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RankBounce")
            .field("ptr", &(self.ptr.load(Ordering::Relaxed) as usize))
            .field("bytes", &self.bytes.load(Ordering::Relaxed))
            .field("grow_lock", &"<Mutex>")
            .finish()
    }
}

impl RankBounce {
    fn empty() -> Self {
        Self {
            ptr: AtomicPtr::new(ptr::null_mut()),
            bytes: AtomicUsize::new(0),
            grow_lock: Mutex::new(()),
        }
    }
}

/// Multi-device cluster holding one `HipDevice` + one pinned bounce slab
/// per rank. Only the peer-copy primitive is wired today; C+ add
/// pipeline-level orchestration on top.
#[derive(Debug)]
pub struct HipCluster {
    devices: Vec<HipDevice>,
    /// Pinned-host bounce buffers, one per rank. Atomic/lock-free on the
    /// hot path — see [`RankBounce`]. Used by the blocking
    /// `peer_copy_via_host` path.
    bounces: Vec<RankBounce>,
    /// 5.a — per-rank auxiliary streams for pipeline-parallel ubatch
    /// pipelining. `aux_streams[rank][lane]` is an independent HIP stream on
    /// rank `rank`. Populated lazily by [`HipCluster::reserve_aux_streams`].
    /// Empty by default so the non-pipelined path stays byte-identical.
    aux_streams: Vec<std::sync::Mutex<Vec<HipStream>>>,
    /// 5.g — per-rank × per-lane pinned bounces for the async
    /// peer-copy path. Previously shared one bounce per rank, which
    /// serialised concurrent async peer-copies across lanes. Each lane
    /// now gets its own pinned slab. Populated via
    /// [`HipCluster::reserve_lane_bounces`]. The Mutex is only held
    /// during the grow phase; the read path is atomic via RankBounce's
    /// AtomicPtr / AtomicUsize.
    lane_bounces: Vec<std::sync::Mutex<Vec<RankBounce>>>,
    /// N×N matrix of authorised BAR1 peer-access edges.
    /// `peer_access[i][j] == true` iff rank `i`'s HIP device may
    /// dereference pointers owned by rank `j` directly through PCIe BAR1
    /// (the prerequisite for the kernel-launched P2P AllReduce path).
    /// Probed once at construction; the diagonal is always `true` (a
    /// device trivially "accesses" its own memory).
    /// Populated even when peer-enable failed: callers consult this matrix
    /// to decide whether to engage the BAR1 AllReduce kernel or fall back
    /// to the host-bounce path on a per-rank-pair basis.
    peer_access: Vec<Vec<bool>>,
    /// Per-rank serialiser around `peer_copy_via_host`. The blocking
    /// peer-copy uses the shared `bounces[src_rank]` host buffer; two
    /// concurrent host threads racing through it (e.g. two streaming
    /// decode requests in pp+tp during their stage-boundary hand-off)
    /// would clobber each other's data in the bounce. One mutex per
    /// src_rank keeps cross-rank parallelism while serialising same-rank
    /// access. Cheap: peer_copy is ~100 µs per stage transition.
    peer_copy_lock: Vec<std::sync::Mutex<()>>,
}

impl HipCluster {
    /// Open one `HipDevice` per rank. `device_ids[r]` is the HIP ordinal
    /// for rank `r`. Typically `0..N` when every card is usable.
    /// As part of construction the cluster probes the BAR1 peer-access
    /// matrix (`hipDeviceCanAccessPeer` for every off-diagonal pair) and
    /// authorises every reachable edge via `hipDeviceEnablePeerAccess`.
    /// Pairs that report unreachable, or whose enable fails for reasons
    /// other than "already enabled", are recorded as `false` in
    /// [`Self::peer_access_matrix`] and the host-bounce AllReduce stays
    /// available as a fallback for that pair.
    pub fn new(device_ids: &[i32]) -> DeviceResult<Self> {
        if device_ids.is_empty() {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: "HipCluster::new needs at least one device id".into(),
            });
        }
        let mut devices = Vec::with_capacity(device_ids.len());
        for &id in device_ids {
            devices.push(HipDevice::new(id)?);
        }
        let bounces = (0..device_ids.len()).map(|_| RankBounce::empty()).collect();
        let aux_streams = (0..device_ids.len())
            .map(|_| std::sync::Mutex::new(Vec::new()))
            .collect();
        let lane_bounces = (0..device_ids.len())
            .map(|_| std::sync::Mutex::new(Vec::new()))
            .collect();
        let peer_access = probe_and_enable_peer_access(&devices)?;
        let peer_copy_lock = (0..device_ids.len())
            .map(|_| std::sync::Mutex::new(()))
            .collect();
        Ok(Self {
            devices,
            bounces,
            aux_streams,
            lane_bounces,
            peer_access,
            peer_copy_lock,
        })
    }

    /// `peer_access[i][j]` — `true` iff rank `i`'s device may dereference
    /// pointers owned by rank `j` directly via PCIe BAR1.
    /// The diagonal is always `true`. Off-diagonal entries are `true` only
    /// when both `hipDeviceCanAccessPeer` returned 1 *and* the matching
    /// `hipDeviceEnablePeerAccess` either succeeded or returned the benign
    /// "already enabled" status. Use [`Self::can_peer_access`] for a
    /// single-pair query.
    pub fn peer_access_matrix(&self) -> &[Vec<bool>] {
        &self.peer_access
    }

    /// Convenience accessor: is BAR1 peer access from rank `src` to rank
    /// `dst` authorised? Out-of-range ranks return `false`.
    pub fn can_peer_access(&self, src: usize, dst: usize) -> bool {
        self.peer_access
            .get(src)
            .and_then(|row| row.get(dst).copied())
            .unwrap_or(false)
    }

    /// Are all off-diagonal edges in the cluster authorised? The BAR1
    /// AllReduce path requires a fully-reachable matrix — a single 0
    /// edge means at least one rank can't read at least one peer, and
    /// the kernel-launched path is unsafe to engage.
    pub fn peer_access_full(&self) -> bool {
        for i in 0..self.peer_access.len() {
            for j in 0..self.peer_access.len() {
                if i != j && !self.peer_access[i][j] {
                    return false;
                }
            }
        }
        true
    }

    /// 5.g — reserve `n_lanes` per-lane pinned bounce slabs per rank,
    /// each pre-grown to `bytes_per_rank` bytes. Required before using
    /// [`Self::peer_copy_via_host_async_laned`] at multiple lanes
    /// concurrently, so each lane has its own pinned memory (no
    /// serialisation via shared host buffer).
    pub fn reserve_lane_bounces(&self, n_lanes: usize, bytes_per_rank: usize) -> DeviceResult<()> {
        for rank in 0..self.devices.len() {
            let mut slot = self.lane_bounces[rank].lock().map_err(|_| DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: "lane_bounces mutex poisoned".into(),
            })?;
            while slot.len() < n_lanes {
                slot.push(RankBounce::empty());
            }
            if bytes_per_rank > 0 {
                // Grow each lane's slab under its own grow_lock; reuse the
                // existing RankBounce grow logic.
                for b in slot.iter() {
                    self.ensure_bounce_in(b, rank, bytes_per_rank)?;
                }
            }
        }
        Ok(())
    }

    /// 5.a — ensure each rank has at least `n_lanes` auxiliary streams
    /// beyond its default stream. Used by the ubatch-pipelined prefill path
    /// so rank `r` can drive ubatch lane `lane` on its own stream without
    /// serialising behind the default stream. Idempotent: growing from 2 to
    /// 4 lanes creates 2 new streams; shrinking does nothing.
    pub fn reserve_aux_streams(&self, n_lanes: usize) -> DeviceResult<()> {
        for rank in 0..self.devices.len() {
            let device = &self.devices[rank];
            device.bind()?;
            let mut slot = self.aux_streams[rank].lock().map_err(|_| DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: "aux_streams mutex poisoned".into(),
            })?;
            while slot.len() < n_lanes {
                // 5.g — non-blocking so lanes truly overlap on the same
                // device (blocking streams serialise via the null stream).
                slot.push(HipStream::new_non_blocking(device.id())?);
            }
        }
        Ok(())
    }

    /// 5.a — run `f` with rank `r`'s aux stream for ubatch lane `lane`.
    /// The lane must have been pre-reserved via
    /// [`HipCluster::reserve_aux_streams`] or this returns an error.
    /// Closure API (rather than returning `&HipStream`) so the aux-streams
    /// Mutex stays held for the duration of the borrow — the `HipStream` is
    /// Send+Sync, but its lifetime is tied to the Vec entry, which is behind
    /// the mutex.
    pub fn with_aux_stream<F, T>(&self, rank: usize, lane: usize, f: F) -> DeviceResult<T>
    where
        F: FnOnce(&HipStream) -> DeviceResult<T>,
    {
        if rank >= self.devices.len() {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!("aux_stream: rank {rank} out of range (N={})", self.devices.len()),
            });
        }
        let guard = self.aux_streams[rank].lock().map_err(|_| DeviceError::Backend {
            backend: "hip",
            code: -1,
            message: "aux_streams mutex poisoned".into(),
        })?;
        if lane >= guard.len() {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!(
                    "aux_stream: lane {lane} not reserved on rank {rank} (have {}); call reserve_aux_streams first",
                    guard.len()
                ),
            });
        }
        f(&guard[lane])
    }

    /// 5.a — number of aux streams currently reserved on rank `r`.
    /// Useful for assertions at the ubatch-loop site.
    pub fn aux_stream_count(&self, rank: usize) -> DeviceResult<usize> {
        if rank >= self.devices.len() {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!("aux_stream_count: rank {rank} out of range"),
            });
        }
        let guard = self.aux_streams[rank].lock().map_err(|_| DeviceError::Backend {
            backend: "hip",
            code: -1,
            message: "aux_streams mutex poisoned".into(),
        })?;
        Ok(guard.len())
    }

    /// Pre-grow every rank's pinned bounce buffer to `bytes_per_rank`.
    /// Call this once at session init with the largest expected
    /// stage-boundary payload (`hidden_dim * sizeof::<f16>() *
    /// max_prefill_tokens`). After this call, `peer_copy_via_host` never
    /// takes the grow lock — it only does the atomic fast-path read.
    /// # Errors
    /// Returns `DeviceError::Alloc` if any rank's pinned allocation fails.
    pub fn reserve_bounce_capacity(&self, bytes_per_rank: usize) -> DeviceResult<()> {
        if bytes_per_rank == 0 {
            return Ok(());
        }
        for rank in 0..self.devices.len() {
            let _ = self.ensure_bounce(rank, bytes_per_rank)?;
        }
        Ok(())
    }

    /// Number of ranks in the cluster.
    pub fn ranks(&self) -> usize {
        self.devices.len()
    }

    /// `HipDevice` for the given rank.
    pub fn device(&self, rank: usize) -> &HipDevice {
        &self.devices[rank]
    }

    /// Free the pinned bounce pools. Safe to call multiple times; a dropped
    /// cluster without an explicit call leaks the pinned host memory (we
    /// log a warn from `Drop` in that case).
    pub fn dispose(mut self) -> DeviceResult<()> {
        // 5.a — drop aux streams first; each rank binds its device
        // before hipStreamDestroy implicit-runs in HipStream's Drop.
        for (rank, slot) in self.aux_streams.drain(..).enumerate() {
            if let Ok(streams) = slot.into_inner() {
                if !streams.is_empty() {
                    self.devices[rank].bind()?;
                }
                drop(streams); // explicit — HipStream::Drop calls hipStreamDestroy
            }
        }
        for bounce in self.bounces.drain(..) {
            Self::free_bounce(&bounce)?;
        }
        // 5.g — also free per-lane bounces.
        for slot in self.lane_bounces.drain(..) {
            if let Ok(bounces) = slot.into_inner() {
                for bounce in bounces {
                    Self::free_bounce(&bounce)?;
                }
            }
        }
        Ok(())
    }

    fn free_bounce(bounce: &RankBounce) -> DeviceResult<()> {
        let p = bounce.ptr.swap(ptr::null_mut(), Ordering::AcqRel);
        if !p.is_null() {
            // SAFETY: `p` came from `hipHostMalloc` inside `ensure_bounce_in`.
            // Atomically swapped out; no concurrent reader will see the stale
            // pointer.
            let rc = unsafe { hipHostFree(p) };
            if rc != HIP_SUCCESS {
                return Err(DeviceError::Backend {
                    backend: "hip",
                    code: rc,
                    message: format!("hipHostFree: {}", error_string(rc)),
                });
            }
        }
        Ok(())
    }

    /// Ensure rank `r`'s pinned bounce buffer is at least `need` bytes.
    fn ensure_bounce(&self, rank: usize, need: usize) -> DeviceResult<*mut c_void> {
        self.ensure_bounce_in(&self.bounces[rank], rank, need)
    }

    /// 5.g — helper that grows an arbitrary `RankBounce` slot
    /// associated with `rank` (used for both the default per-rank bounce
    /// and the per-lane lane_bounces).
    fn ensure_bounce_in(
        &self,
        slot: &RankBounce,
        rank: usize,
        need: usize,
    ) -> DeviceResult<*mut c_void> {
        // Fast path: the buffer is already large enough. Acquire ordering
        // pairs with the Release store in the grow path below so a reader
        // that observes `bytes >= need` is guaranteed to also observe the
        // matching `ptr` write.
        if slot.bytes.load(Ordering::Acquire) >= need {
            let p = slot.ptr.load(Ordering::Acquire);
            if !p.is_null() {
                return Ok(p);
            }
        }

        // Cold path: serialise growers.
        let _guard = slot.grow_lock.lock().map_err(|_poisoned| DeviceError::Backend {
            backend: "hip",
            code: -1,
            message: "HipCluster grow_lock poisoned".into(),
        })?;
        // Double-check — another writer may have grown the buffer while we
        // were waiting for the lock.
        if slot.bytes.load(Ordering::Acquire) >= need {
            let p = slot.ptr.load(Ordering::Acquire);
            if !p.is_null() {
                return Ok(p);
            }
        }

        // Free the old slab (if any). Readers still holding the stale `ptr`
        // could race here in principle, but in our discipline the grow
        // happens at session init before any peer copies fire (see
        // `reserve_bounce_capacity`), so steady-state readers never see
        // this transition.
        let old_ptr = slot.ptr.swap(ptr::null_mut(), Ordering::AcqRel);
        if !old_ptr.is_null() {
            // SAFETY: `old_ptr` came from a prior `hipHostMalloc` in this same
            // function. We atomically swapped it out so no concurrent grower
            // will double-free; concurrent readers on the fast path are not
            // permitted in the documented usage (session init before launches).
            let rc = unsafe { hipHostFree(old_ptr) };
            if rc != HIP_SUCCESS {
                return Err(DeviceError::Backend {
                    backend: "hip",
                    code: -1,
                    message: format!("hipHostFree (grow): {}", error_string(rc)),
                });
            }
        }
        // Bind the source device before allocating pinned memory — on
        // ROCm the current device at alloc time determines the pinned
        // page's NUMA affinity.
        self.devices[rank].bind()?;
        let mut new_ptr: *mut c_void = ptr::null_mut();
        // SAFETY: `hipHostMalloc` writes a host pointer through the out-pointer
        // and reads nothing from it. `&mut new_ptr` is valid for writes of
        // `sizeof(void*)`. Returned pointer ownership is transferred into `slot`.
        let rc = unsafe { hipHostMalloc(&raw mut new_ptr, need, HIP_HOST_MALLOC_PORTABLE) };
        if rc != HIP_SUCCESS {
            return Err(DeviceError::Alloc {
                backend: "hip",
                device: self.devices[rank].id(),
                bytes: need,
                reason: format!("hipHostMalloc: {}", error_string(rc)),
            });
        }
        // Release ordering on both stores pairs with the Acquire loads in
        // the fast path: readers that observe the new `bytes` value are
        // guaranteed to see the matching `ptr`.
        slot.ptr.store(new_ptr, Ordering::Release);
        slot.bytes.store(need, Ordering::Release);
        Ok(new_ptr)
    }

    /// Single-chunk host-bounce peer copy: DtoH on src rank → sync →
    /// HtoD on dst rank → sync. Returns when the destination buffer
    /// holds the transferred bytes.
    /// Correctness-first path. For the PP hot path (≤ 512 KB per hop),
    /// the single-memcpy overhead is one DtoH + one HtoD, each a few µs
    /// on MI50 PCIe 3.0 x16. A pipelined multi-chunk variant is the
    /// natural follow-up if we ever have to shuffle > 4 MiB between ranks.
    /// # Safety
    /// - `src_ptr` must point to at least `bytes` valid device bytes on
    /// the source rank's HIP device.
    /// - `dst_ptr` must point to at least `bytes` valid device bytes on
    /// the destination rank's HIP device.
    /// - No other stream on either device may concurrently access the
    /// source or destination regions.
    pub unsafe fn peer_copy_via_host(
        &self,
        dst_ptr: DevicePtr,
        dst_rank: usize,
        src_ptr: DevicePtr,
        src_rank: usize,
        bytes: usize,
    ) -> DeviceResult<()> {
        if bytes == 0 {
            return Ok(());
        }
        if src_rank >= self.devices.len() || dst_rank >= self.devices.len() {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!(
                    "peer_copy_via_host: rank out of range (src={src_rank}, dst={dst_rank}, N={})",
                    self.devices.len()
                ),
            });
        }
        // Same-rank short-circuit: plain device-to-device on the dst
        // device (identical to src).
        if src_rank == dst_rank {
            let device = &self.devices[dst_rank];
            device.bind()?;
            // SAFETY: caller's contract on this `unsafe fn` requires both
            // `src_ptr` and `dst_ptr` to be valid for `bytes` on this device,
            // and non-aliased with any other pending op on `default_stream`.
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    flambeau_core::CopyDirection::DeviceToDevice,
                    dst_ptr,
                    src_ptr,
                    bytes,
                )?;
            }
            <HipDevice as Device>::synchronize(device)?;
            return Ok(());
        }

        // Serialise concurrent peer_copy_via_host calls on the same
        // src_rank — they share `bounces[src_rank]`. Two host threads
        // racing here would clobber each other's pinned buffer.
        // Held across DtoH+sync+HtoD+sync; total ~100 µs per stage
        // transition. See `peer_copy_lock` field doc.
        let _bounce_guard = self.peer_copy_lock[src_rank].lock().map_err(|_| {
            DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!("peer_copy_lock[{src_rank}] poisoned"),
            }
        })?;

        let buf = self.ensure_bounce(src_rank, bytes)?;

        // 1. DtoH on src rank's default stream. We bind before the async
        // launch so `hipMemcpyAsync` routes to the right device's
        // command queue.
        let src_dev = &self.devices[src_rank];
        src_dev.bind()?;
        // SAFETY: `buf` is a live pinned-host allocation sized `bytes` (from
        // `ensure_bounce`). `src_ptr` is valid for `bytes` on `src_dev` per the
        // outer fn's caller contract. `src_dev.default_stream().raw()` is a
        // live stream handle owned by `src_dev`.
        let rc = unsafe {
            hipMemcpyAsync(
                buf,
                src_ptr.as_usize() as *const c_void,
                bytes,
                hipMemcpyKind::DeviceToHost,
                src_dev.default_stream().raw(),
            )
        };
        check(rc, "peer_copy DtoH")?;
        // SAFETY: live stream handle as above.
        let rc = unsafe { hipStreamSynchronize(src_dev.default_stream().raw()) };
        check(rc, "peer_copy DtoH sync")?;

        // 2. HtoD on dst rank's default stream.
        let dst_dev = &self.devices[dst_rank];
        dst_dev.bind()?;
        // SAFETY: `buf` still holds the transferred bytes (DtoH sync above
        // guarantees the pinned buffer is fully populated). `dst_ptr` is
        // valid for `bytes` on `dst_dev` per the outer fn's caller contract.
        // `dst_dev.default_stream().raw()` is a live stream handle.
        let rc = unsafe {
            hipMemcpyAsync(
                dst_ptr.as_usize() as *mut c_void,
                buf.cast_const(),
                bytes,
                hipMemcpyKind::HostToDevice,
                dst_dev.default_stream().raw(),
            )
        };
        check(rc, "peer_copy HtoD")?;
        // followup** — sync the HtoD before returning so the
        // pinned bounce buffer (shared per src rank) is no longer in
        // flight. Without this sync, callers that loop this primitive
        // for fan-out (PP-of-TP hand-off: stage s rank 0 → all
        // tp_size ranks of stage s+1) race the next iteration's DtoH
        // against the previous iteration's still-pending HtoD over the
        // same bounce buffer. Symptom: non-deterministic decode at
        // temp=0 starting from token 2 (token 1 of the very first
        // hand-off is correct because nothing competes for the
        // buffer yet). PP-only is unaffected (single dst per hand-off
        // ⇒ the loop body runs once per stage). Cost ~50–200 µs per
        // hand-off; acceptable on the cold path. SAFETY: live stream
        // handle as above.
        let rc = unsafe { hipStreamSynchronize(dst_dev.default_stream().raw()) };
        check(rc, "peer_copy HtoD sync")?;

        Ok(())
    }

    /// Async PCIe peer copy: submits DtoH + HtoD on the rank default
    /// streams; dst waits on src's DtoH via a HIP event (no CPU sync).
    /// Callers must guarantee — via a downstream sync on the dst stream
    /// before the next call — that the previous HtoD has read the
    /// shared `bounces[src_rank]` slab before the next DtoH overwrites it.
    /// # Safety
    /// As per [`Self::peer_copy_via_host`].
    pub unsafe fn peer_copy_via_host_event(
        &self,
        dst_ptr: DevicePtr,
        dst_rank: usize,
        src_ptr: DevicePtr,
        src_rank: usize,
        bytes: usize,
    ) -> DeviceResult<()> {
        if bytes == 0 {
            return Ok(());
        }
        if src_rank >= self.devices.len() || dst_rank >= self.devices.len() {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!(
                    "peer_copy_via_host_event: rank out of range (src={src_rank}, dst={dst_rank}, N={})",
                    self.devices.len()
                ),
            });
        }
        if src_rank == dst_rank {
            // Same-device DtoD on dst's default stream — fully async.
            let device = &self.devices[dst_rank];
            device.bind()?;
            // SAFETY: per caller contract, `src_ptr`/`dst_ptr` valid for `bytes`.
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    flambeau_core::CopyDirection::DeviceToDevice,
                    dst_ptr,
                    src_ptr,
                    bytes,
                )?;
            }
            return Ok(());
        }

        // Shares the per-src lock + bounce buffer with `peer_copy_via_host`.
        let _bounce_guard = self.peer_copy_lock[src_rank].lock().map_err(|_| {
            DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!("peer_copy_lock[{src_rank}] poisoned"),
            }
        })?;

        let buf = self.ensure_bounce(src_rank, bytes)?;

        let src_dev = &self.devices[src_rank];
        src_dev.bind()?;
        let src_stream = src_dev.default_stream();
        // SAFETY: `buf` is a pinned-host region of >= `bytes` bytes
        // (`ensure_bounce`); `src_ptr` is valid for `bytes` per caller contract.
        let rc = unsafe {
            hipMemcpyAsync(
                buf,
                src_ptr.as_usize() as *const c_void,
                bytes,
                hipMemcpyKind::DeviceToHost,
                src_stream.raw(),
            )
        };
        check(rc, "peer_copy_event DtoH")?;

        let bridge = crate::HipEvent::new(src_dev.id())?;
        bridge.record(src_stream)?;

        let dst_dev = &self.devices[dst_rank];
        dst_dev.bind()?;
        let dst_stream = dst_dev.default_stream();
        bridge.stream_wait(dst_stream)?;
        // SAFETY: `bridge` orders this HtoD after the DtoH that populates `buf`.
        let rc = unsafe {
            hipMemcpyAsync(
                dst_ptr.as_usize() as *mut c_void,
                buf.cast_const(),
                bytes,
                hipMemcpyKind::HostToDevice,
                dst_stream.raw(),
            )
        };
        check(rc, "peer_copy_event HtoD")?;

        Ok(())
    }

    /// 5.b — fully-asynchronous PCIe peer copy.
    /// Unlike [`Self::peer_copy_via_host`] which blocks the CPU between DtoH
    /// and HtoD via `hipStreamSynchronize`, this variant records a HIP event
    /// after the DtoH and has the destination stream wait on it driver-side.
    /// The caller's `src_stream` and `dst_stream` continue receiving work
    /// without host round-trips — essential for the 5.d async ubatch
    /// pipeline where rank k's next ubatch should start before rank k+1's
    /// current ubatch finishes.
    /// `done_event` is optional: if `Some`, recorded on `dst_stream` after
    /// the HtoD completes (so downstream dependents can wait without a sync).
    /// Same-rank and rank-out-of-range paths mirror the blocking variant.
    /// # Safety
    /// - `src_ptr` / `dst_ptr` must be valid for `bytes` on their respective
    /// devices.
    /// - `src_stream` must be on `src_rank`'s device; `dst_stream` on
    /// `dst_rank`'s device.
    /// - No other work may concurrently alias the pinned bounce buffer
    /// bytes for `src_rank` between the DtoH and HtoD on different streams.
    /// In practice this means: do not issue two overlapping async peer
    /// copies from the SAME source rank on different lanes without
    /// per-lane bounce buffers — the current impl has one bounce per rank.
    /// 5.d works around this by pacing: each ubatch stage completes
    /// its DtoH before the next stage starts its DtoH on the same rank.
    pub unsafe fn peer_copy_via_host_async(
        &self,
        dst_ptr: DevicePtr,
        dst_rank: usize,
        src_ptr: DevicePtr,
        src_rank: usize,
        bytes: usize,
        src_stream: &HipStream,
        dst_stream: &HipStream,
        bridge_event: &crate::HipEvent,
        done_event: Option<&crate::HipEvent>,
    ) -> DeviceResult<()> {
        // SAFETY: forwards to the laned variant with lane=0 (legacy
        // per-rank bounce). Callers that need concurrent async peer-copies
        // across lanes should use `peer_copy_via_host_async_laned`.
        unsafe {
            self.peer_copy_via_host_async_laned(
                dst_ptr, dst_rank, src_ptr, src_rank, bytes,
                src_stream, dst_stream, bridge_event, done_event, None,
            )
        }
    }

    /// 5.g — async peer copy with an optional `lane` for per-lane
    /// bounce buffer selection. When `lane = Some(L)`, the DtoH writes to
    /// `lane_bounces[src_rank][L]` (must have been reserved via
    /// [`Self::reserve_lane_bounces`]). When `None`, falls back to the
    /// shared per-rank bounce (identical to the unlaned variant).
    /// Per-lane bounces let two concurrent async peer-copies from the
    /// same source rank truly overlap — each lane has its own pinned
    /// slab so the driver's memcpy DAG doesn't force serialisation on
    /// shared host memory.
    /// # Safety
    /// Same as the unlaned variant. Additionally: when `lane = Some(L)`,
    /// no other in-flight async copy may alias lane L's bounce on this
    /// source rank.
    #[expect(clippy::too_many_arguments, reason = "full peer-copy contract")]
    pub unsafe fn peer_copy_via_host_async_laned(
        &self,
        dst_ptr: DevicePtr,
        dst_rank: usize,
        src_ptr: DevicePtr,
        src_rank: usize,
        bytes: usize,
        src_stream: &HipStream,
        dst_stream: &HipStream,
        bridge_event: &crate::HipEvent,
        done_event: Option<&crate::HipEvent>,
        lane: Option<usize>,
    ) -> DeviceResult<()> {
        if bytes == 0 {
            return Ok(());
        }
        if src_rank >= self.devices.len() || dst_rank >= self.devices.len() {
            return Err(DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: format!(
                    "peer_copy_via_host_async: rank out of range (src={src_rank}, dst={dst_rank}, N={})",
                    self.devices.len()
                ),
            });
        }
        if src_rank == dst_rank {
            // Same-rank DtoD on dst_stream, no cross-device events needed.
            let device = &self.devices[dst_rank];
            device.bind()?;
            // SAFETY: caller's contract on this `unsafe fn`.
            unsafe {
                device.memcpy_async(
                    dst_stream,
                    flambeau_core::CopyDirection::DeviceToDevice,
                    dst_ptr,
                    src_ptr,
                    bytes,
                )?;
            }
            if let Some(done) = done_event {
                done.record(dst_stream)?;
            }
            return Ok(());
        }

        let buf = match lane {
            Some(l) => {
                let guard = self.lane_bounces[src_rank].lock().map_err(|_| {
                    DeviceError::Backend {
                        backend: "hip",
                        code: -1,
                        message: "lane_bounces mutex poisoned".into(),
                    }
                })?;
                if l >= guard.len() {
                    return Err(DeviceError::Backend {
                        backend: "hip",
                        code: -1,
                        message: format!(
                            "peer_copy_via_host_async_laned: lane {l} not reserved on rank {src_rank} (have {}); call reserve_lane_bounces first",
                            guard.len()
                        ),
                    });
                }
                self.ensure_bounce_in(&guard[l], src_rank, bytes)?
            }
            None => self.ensure_bounce(src_rank, bytes)?,
        };

        // 1. DtoH on src_stream.
        let src_dev = &self.devices[src_rank];
        src_dev.bind()?;
        // SAFETY: per caller contract + `ensure_bounce` guarantees `buf` is a
        // live pinned-host region of `bytes` bytes.
        let rc = unsafe {
            hipMemcpyAsync(
                buf,
                src_ptr.as_usize() as *const c_void,
                bytes,
                hipMemcpyKind::DeviceToHost,
                src_stream.raw(),
            )
        };
        check(rc, "peer_copy_async DtoH")?;
        // 2. Record bridge event on src_stream.
        bridge_event.record(src_stream)?;

        // 3. dst_stream waits on the bridge event (driver-side, no CPU sync).
        let dst_dev = &self.devices[dst_rank];
        dst_dev.bind()?;
        bridge_event.stream_wait(dst_stream)?;

        // 4. HtoD on dst_stream.
        // SAFETY: bridge_event guarantees the pinned buf is populated before
        // this HtoD starts (driver DAG edge).
        let rc = unsafe {
            hipMemcpyAsync(
                dst_ptr.as_usize() as *mut c_void,
                buf.cast_const(),
                bytes,
                hipMemcpyKind::HostToDevice,
                dst_stream.raw(),
            )
        };
        check(rc, "peer_copy_async HtoD")?;

        // 5. Optional done event for downstream waiters.
        if let Some(done) = done_event {
            done.record(dst_stream)?;
        }

        Ok(())
    }
}

impl Drop for HipCluster {
    fn drop(&mut self) {
        for bounce in &self.bounces {
            // Relaxed is fine here: we're on the sole remaining thread
            // holding this cluster (Drop implies no outstanding borrows).
            let p = bounce.ptr.load(Ordering::Relaxed);
            if !p.is_null() {
                let bytes = bounce.bytes.load(Ordering::Relaxed);
                tracing::warn!(
                    target: "flambeau_backend_hip::cluster",
                    bytes,
                    "HipCluster dropped without dispose(); pinned host buffer leaked"
                );
            }
        }
    }
}

fn check(rc: c_int, tag: &str) -> DeviceResult<()> {
    if rc == HIP_SUCCESS {
        Ok(())
    } else {
        Err(DeviceError::Backend {
            backend: "hip",
            code: rc,
            message: format!("{tag}: {}", error_string(rc)),
        })
    }
}

/// Probe `hipDeviceCanAccessPeer` for every off-diagonal `(src, dst)` rank
/// pair and call `hipDeviceEnablePeerAccess` where reachable. Returns the
/// resulting N×N matrix.
/// Failures are *not* propagated: a pair that can't be enabled is
/// recorded as `false` and the cluster continues to construct (the
/// host-bounce AllReduce stays available for that pair). A `tracing::warn`
/// is emitted per blocked edge so the operator can correlate against
/// BIOS / motherboard topology.
fn probe_and_enable_peer_access(devices: &[HipDevice]) -> DeviceResult<Vec<Vec<bool>>> {
    let n = devices.len();
    let mut matrix = vec![vec![false; n]; n];
    for i in 0..n {
        matrix[i][i] = true;
    }
    if n < 2 {
        return Ok(matrix);
    }
    for src in 0..n {
        // Bind src so hipDeviceEnablePeerAccess targets the right device —
        // it operates on the *current* HIP device, granting it access to
        // the supplied peer.
        devices[src].bind()?;
        for dst in 0..n {
            if src == dst {
                continue;
            }
            let mut can: c_int = 0;
            // SAFETY: `hipDeviceCanAccessPeer` writes a single c_int
            // through the out-pointer and reads the two device ordinals
            // by value. `&raw mut can` is a valid pointer for one i32
            // write; the device ids come from already-opened HipDevices.
            let rc = unsafe {
                hipDeviceCanAccessPeer(&raw mut can, devices[src].id(), devices[dst].id())
            };
            if rc != HIP_SUCCESS {
                tracing::warn!(
                    target: "flambeau_backend_hip::cluster",
                    src_rank = src,
                    dst_rank = dst,
                    src_device = devices[src].id(),
                    dst_device = devices[dst].id(),
                    error = error_string(rc),
                    "hipDeviceCanAccessPeer failed; treating edge as unreachable"
                );
                continue;
            }
            if can != 1 {
                tracing::warn!(
                    target: "flambeau_backend_hip::cluster",
                    src_rank = src,
                    dst_rank = dst,
                    "hipDeviceCanAccessPeer reported 0; BAR1 P2P not available on this edge \
                     (check BIOS Above-4G-Decoding + Resizable-BAR)"
                );
                continue;
            }
            // Reserved per HIP spec — must be 0.
            const ENABLE_PEER_FLAGS: c_uint = 0;
            // SAFETY: `hipDeviceEnablePeerAccess` operates on the
            // currently-bound device (set above) and reads the peer id
            // by value. No memory is dereferenced through these args.
            let rc = unsafe { hipDeviceEnablePeerAccess(devices[dst].id(), ENABLE_PEER_FLAGS) };
            if rc == HIP_SUCCESS || rc == HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED {
                matrix[src][dst] = true;
            } else {
                tracing::warn!(
                    target: "flambeau_backend_hip::cluster",
                    src_rank = src,
                    dst_rank = dst,
                    error = error_string(rc),
                    "hipDeviceEnablePeerAccess failed; falling back to host-bounce on this edge"
                );
            }
        }
    }
    Ok(matrix)
}
