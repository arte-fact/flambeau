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
use std::sync::Mutex;

use crate::sys::{
    error_string, hipHostFree, hipHostMalloc, hipMemcpyAsync, hipMemcpyKind,
    hipStreamSynchronize, HIP_HOST_MALLOC_PORTABLE, HIP_SUCCESS,
};
use crate::HipDevice;
use flambeau_core::{Device, DeviceError, DevicePtr, DeviceResult};

/// One entry per rank: the pinned host bounce buffer (grown on demand)
/// wrapped in a mutex so two concurrent peer copies can't race on the
/// same allocation.
struct RankBounce {
    ptr: *mut c_void,
    bytes: usize,
}

impl std::fmt::Debug for RankBounce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RankBounce")
            .field("ptr", &(self.ptr as usize))
            .field("bytes", &self.bytes)
            .finish()
    }
}

// SAFETY: `RankBounce` holds a raw pointer returned by `hipHostMalloc`.
// It's only mutated under the cluster's per-rank mutex, and reads during
// memcpy are driver-synchronised.
unsafe impl Send for RankBounce {}

impl RankBounce {
    const fn empty() -> Self {
        Self {
            ptr: ptr::null_mut(),
            bytes: 0,
        }
    }
}

/// Multi-device cluster holding one `HipDevice` + one pinned bounce slab
/// per rank. Only the peer-copy primitive is wired today; V1.7.5-C+ add
/// pipeline-level orchestration on top.
#[derive(Debug)]
pub struct HipCluster {
    devices: Vec<HipDevice>,
    /// One default stream per rank. The `HipDevice` already owns a default
    /// stream; we keep a borrowing view via index for clarity.
    bounces: Vec<Mutex<RankBounce>>,
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
        let bounces = (0..device_ids.len())
            .map(|_| Mutex::new(RankBounce::empty()))
            .collect();
        Ok(Self { devices, bounces })
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
            let b = bounce.into_inner().map_err(|_| DeviceError::Backend {
                backend: "hip",
                code: -1,
                message: "HipCluster bounce mutex poisoned during dispose".into(),
            })?;
            if !b.ptr.is_null() {
                // SAFETY: `b.ptr` came from `hipHostMalloc` above. `b` is a
                // local from `into_inner` so it drops after this block; no
                // need to null-out the fields.
                let rc = unsafe { hipHostFree(b.ptr) };
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
    /// Grows (re-allocs) if needed; returns the buffer pointer.
    fn ensure_bounce(&self, rank: usize, need: usize) -> DeviceResult<*mut c_void> {
        let mut slot = self.bounces[rank].lock().map_err(|_| DeviceError::Backend {
            backend: "hip",
            code: -1,
            message: "HipCluster bounce mutex poisoned".into(),
        })?;
        if slot.bytes >= need && !slot.ptr.is_null() {
            return Ok(slot.ptr);
        }
        // Free the old slab (if any) and allocate a fresh one. V1.7.5-B's
        // hot-path payloads are all small (≤ 512 KB), so a single grow to
        // the first observed size is the common case.
        if !slot.ptr.is_null() {
            // SAFETY: `slot.ptr` came from a prior `hipHostMalloc` in this same
            // function. The `slot` mutex guard serialises access so nobody else
            // is reading/writing the buffer; we null the field immediately after.
            let rc = unsafe { hipHostFree(slot.ptr) };
            if rc != HIP_SUCCESS {
                return Err(DeviceError::Backend {
                    backend: "hip",
                    code: -1,
                    message: format!("hipHostFree (grow): {}", error_string(rc)),
                });
            }
            slot.ptr = ptr::null_mut();
            slot.bytes = 0;
        }
        // Bind the source device before allocating pinned memory — on
        // ROCm the current device at alloc time determines the pinned
        // page's NUMA affinity.
        self.devices[rank].bind()?;
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: `hipHostMalloc` writes a host pointer through the out-pointer
        // and reads nothing from it. `&mut ptr` is valid for writes of
        // `sizeof(void*)`. Returned pointer ownership is transferred into `slot`.
        let rc = unsafe { hipHostMalloc(&mut ptr, need, HIP_HOST_MALLOC_PORTABLE) };
        if rc != HIP_SUCCESS {
            return Err(DeviceError::Alloc {
                backend: "hip",
                device: self.devices[rank].id(),
                bytes: need,
                reason: format!("hipHostMalloc: {}", error_string(rc)),
            });
        }
        slot.ptr = ptr;
        slot.bytes = need;
        Ok(ptr)
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
            let b = match bounce.lock() {
                Ok(g) => g,
                Err(_) => continue, // poisoned; best-effort
            };
            if !b.ptr.is_null() {
                tracing::warn!(
                    target: "flambeau_backend_hip::cluster",
                    bytes = b.bytes,
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
