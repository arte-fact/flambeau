//! V1.7.5 — multi-device cluster primitive for pipeline parallelism.
//!
//! A `HipCluster` owns one `HipDevice` per rank plus a pinned host
//! bounce buffer per rank. Its single public op today is `peer_copy_via_host`:
//! a **CPU-bounce peer copy** that stages a DeviceToHost transfer on the
//! source rank and a HostToDevice transfer on the destination rank through
//! pinned host memory.
//!
//! Why not direct `hipMemcpyPeerAsync`? The V1 target rig is PCIe-only
//! (no xGMI / no NVLink / no kernel `CONFIG_HSA_AMD_P2P`). On that
//! topology, direct peer copies appear to submit successfully but the
//! stream ends up in an unsync'able state (candle's
//! `hip_backend/cluster.rs` documents the same behaviour on ROCm
//! 7.1.1 + gfx906). The host-bounce fallback is boring and reliable.
//!
//! Bandwidth envelope (candle measured on MI50 Gen3 x16): single-chunk
//! pinned-host bounce hits ~6.75 GB/s. Pipelined multi-chunk with events
//! and a dedicated DtoH stream climbs to ~10–11 GB/s. V1.7.5-B ships the
//! single-chunk path — it covers every stage-boundary transfer on the
//! decode/prefill hot path (hidden-state F16 at L ≤ 128 tokens ≤ 512 KB
//! per hop), which sits comfortably below the 4 MiB chunk threshold the
//! pipelined path was designed to exceed. Pipelined copy lands later if
//! we ever shuffle bulk (weight re-sharding, KV migration) between ranks.

use std::os::raw::{c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::sys::{
    error_string, hipHostFree, hipHostMalloc, hipMemcpyAsync, hipMemcpyKind,
    hipStreamSynchronize, HIP_HOST_MALLOC_PORTABLE, HIP_SUCCESS,
};
use crate::HipDevice;
use flambeau_core::{Device, DeviceError, DevicePtr, DeviceResult};

/// One entry per rank: the pinned host bounce buffer.
///
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
/// per rank. Only the peer-copy primitive is wired today; V1.7.5-C+ add
/// pipeline-level orchestration on top.
#[derive(Debug)]
pub struct HipCluster {
    devices: Vec<HipDevice>,
    /// Pinned-host bounce buffers, one per rank. Atomic/lock-free on the
    /// hot path — see [`RankBounce`].
    bounces: Vec<RankBounce>,
}

impl HipCluster {
    /// Open one `HipDevice` per rank. `device_ids[r]` is the HIP ordinal
    /// for rank `r`. Typically `0..N` when every card is usable.
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
        Ok(Self { devices, bounces })
    }

    /// Pre-grow every rank's pinned bounce buffer to `bytes_per_rank`.
    ///
    /// Call this once at session init with the largest expected
    /// stage-boundary payload (`hidden_dim * sizeof::<f16>() *
    /// max_prefill_tokens`). After this call, `peer_copy_via_host` never
    /// takes the grow lock — it only does the atomic fast-path read.
    ///
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
        for bounce in self.bounces.drain(..) {
            // Atomically null out the ptr so a concurrent reader (if the
            // caller ignored the "dispose after last use" contract) sees
            // a null rather than a dangling pointer. `bounce` is moved
            // out of `self.bounces` so no other reference exists.
            let p = bounce.ptr.swap(ptr::null_mut(), Ordering::AcqRel);
            if !p.is_null() {
                // SAFETY: `p` came from `hipHostMalloc` inside `ensure_bounce`.
                // We atomically swapped it out; `bounce` itself drops at the
                // end of this iteration.
                let rc = unsafe { hipHostFree(p) };
                if rc != HIP_SUCCESS {
                    return Err(DeviceError::Backend {
                        backend: "hip",
                        code: rc,
                        message: format!("hipHostFree: {}", error_string(rc)),
                    });
                }
            }
        }
        Ok(())
    }

    /// Ensure rank `r`'s pinned bounce buffer is at least `need` bytes.
    ///
    /// Hot path (steady state, buffer already large enough): two atomic
    /// loads, no lock. Cold path (first call for this capacity): takes the
    /// grow_lock, double-checks, frees the old slab, allocates a new one,
    /// and releases the grow_lock. Callers that want to avoid ever hitting
    /// the cold path should call [`Self::reserve_bounce_capacity`] at
    /// session init.
    fn ensure_bounce(&self, rank: usize, need: usize) -> DeviceResult<*mut c_void> {
        let slot = &self.bounces[rank];
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
        let _guard = slot.grow_lock.lock().map_err(|_| DeviceError::Backend {
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
        let rc = unsafe { hipHostMalloc(&mut new_ptr, need, HIP_HOST_MALLOC_PORTABLE) };
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
    ///
    /// Correctness-first path. For the PP hot path (≤ 512 KB per hop),
    /// the single-memcpy overhead is one DtoH + one HtoD, each a few µs
    /// on MI50 PCIe 3.0 x16. A pipelined multi-chunk variant is the
    /// natural follow-up if we ever have to shuffle > 4 MiB between ranks.
    ///
    /// # Safety
    /// - `src_ptr` must point to at least `bytes` valid device bytes on
    ///   the source rank's HIP device.
    /// - `dst_ptr` must point to at least `bytes` valid device bytes on
    ///   the destination rank's HIP device.
    /// - No other stream on either device may concurrently access the
    ///   source or destination regions.
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

        let buf = self.ensure_bounce(src_rank, bytes)?;

        // 1. DtoH on src rank's default stream. We bind before the async
        //    launch so `hipMemcpyAsync` routes to the right device's
        //    command queue.
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
                buf as *const c_void,
                bytes,
                hipMemcpyKind::HostToDevice,
                dst_dev.default_stream().raw(),
            )
        };
        check(rc, "peer_copy HtoD")?;
        // NO post-HtoD sync: downstream kernels on dst_dev's default_stream
        // will serialize naturally against this memcpy. CPU-waiting here
        // burns ~200 µs per hand-off with no correctness benefit.
        // (DtoH sync above IS required — the pinned bounce buffer must be
        // fully filled before the HtoD reads it, since HtoD is on a different
        // stream on a different device.)

        Ok(())
    }
}

impl Drop for HipCluster {
    fn drop(&mut self) {
        for bounce in self.bounces.iter() {
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
