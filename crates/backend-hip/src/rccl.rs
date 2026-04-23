//! RCCL-backed `Mesh<N>` + collective-op implementations.
//!
//! Design: a `HipMesh<N>` is created from a single process via
//! `ncclCommInitAll`, which atomically builds `N` communicators over the
//! selected devices. Each `HipRankHandle` borrows one comm and binds HIP
//! ops to its device. Tests drive ranks from `N` threads; every thread runs
//! `HipDevice::bind()` + its handle's collectives on its own stream.
//!
//! Buffers for collectives must already live on the rank's device memory —
//! RCCL is a device-side library. Hosts copy to device before the collective
//! and read back after. The runtime-level `AllReduce` trait operates on byte
//! slices; we interpret those as **device pointers**: the caller is
//! responsible for having allocated device memory and copied input into it.
//!
//! To keep the runtime trait unchanged we expose two surfaces:
//! - `HipRankHandle::all_reduce_device(d_ptr, ...)` — the "real" HIP path.
//! - `HipRankHandle::all_reduce_host(host_bytes, ...)` — convenience that
//!   copies H→D, calls RCCL, copies D→H. Used by the cert harness.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "RCCL FFI — every unsafe block wraps an `nccl*` call with comm / stream / \
              buffer pointers validated at HipMesh construction and owned for the \
              communicators lifetime."
)]

use std::os::raw::c_int;
use std::ptr;
use std::sync::Arc;

use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_runtime::{
    collective::{CollectiveError, CollectiveResult},
    CollectiveCfg, CollectiveDType, Mesh, RankId, ReduceOp,
};

use crate::device::HipDevice;
use crate::rccl_sys::{
    nccl_error_string, ncclAllGather, ncclAllReduce, ncclBroadcast, ncclCommDestroy,
    ncclCommInitRank, ncclComm_t, ncclDataType_t, ncclGetUniqueId, ncclGroupEnd, ncclGroupStart,
    ncclRecv, ncclRedOp_t, ncclSend, ncclUniqueId, NCCL_SUCCESS,
};

fn rccl_check(code: c_int, ctx: &'static str) -> CollectiveResult<()> {
    if code == NCCL_SUCCESS {
        Ok(())
    } else {
        Err(CollectiveError::Backend {
            backend: "hip-rccl",
            ctx,
            code,
            message: nccl_error_string(code),
        })
    }
}

fn to_nccl_dtype(d: CollectiveDType) -> ncclDataType_t {
    match d {
        CollectiveDType::F32 => ncclDataType_t::Float32,
        CollectiveDType::F16 => ncclDataType_t::Float16,
    }
}

fn to_nccl_op(op: ReduceOp) -> ncclRedOp_t {
    match op {
        ReduceOp::Sum => ncclRedOp_t::Sum,
        ReduceOp::Max => ncclRedOp_t::Max,
        ReduceOp::Min => ncclRedOp_t::Min,
    }
}

/// A multi-GPU mesh over HIP devices, backed by RCCL.
///
/// Built rank-by-rank via [`HipMesh::builder`] → `connect_rank` on each
/// driver thread. This matches candle's proven pattern: each rank calls
/// `ncclCommInitRank` from its own thread after `hipSetDevice(rank)`, and
/// a shared `ncclUniqueId` rendez-vous's them. `ncclCommInitAll` from a
/// single thread segfaults on gfx906 in ROCm 7.1+ with our usage pattern;
/// we avoid it entirely.
pub struct HipMesh {
    comms: Vec<std::sync::OnceLock<ncclComm_t>>,
    devices: Vec<i32>,
    unique_id: ncclUniqueId,
}

unsafe impl Send for HipMesh {}
unsafe impl Sync for HipMesh {}

impl HipMesh {
    /// Build an empty mesh with a freshly-generated `ncclUniqueId`. Each
    /// rank must call [`HipRankHandle::connect`] from its own driver thread
    /// before collectives will work on that rank.
    ///
    /// RCCL's `ncclCommInitRank` **blocks** until every rank has joined, so
    /// all N driver threads must race to call it around the same time.
    pub fn new(devices: &[i32]) -> CollectiveResult<Arc<Self>> {
        assert!(!devices.is_empty());
        let mut unique_id = ncclUniqueId::default();
        let code = unsafe { ncclGetUniqueId(&mut unique_id as *mut _) };
        rccl_check(code, "ncclGetUniqueId")?;
        Ok(Arc::new(Self {
            comms: (0..devices.len()).map(|_| std::sync::OnceLock::new()).collect(),
            devices: devices.to_vec(),
            unique_id,
        }))
    }

    pub fn rank_count(&self) -> u32 {
        self.comms.len() as u32
    }

    pub fn device_id(&self, rank: RankId) -> i32 {
        self.devices[rank.0 as usize]
    }

    /// Hand out a per-rank handle. Each rank's operations must be issued from
    /// a thread that has `hipSetDevice(device_id)` in effect.
    pub fn rank_handle(self: &Arc<Self>, rank: RankId) -> HipRankHandle {
        assert!(
            (rank.0 as usize) < self.comms.len(),
            "rank {} exceeds mesh size {}",
            rank.0,
            self.comms.len()
        );
        HipRankHandle {
            rank,
            mesh: Arc::clone(self),
        }
    }
}

impl Mesh for HipMesh {
    fn rank_count(&self) -> u32 {
        self.comms.len() as u32
    }
    fn backend(&self) -> &'static str {
        "hip-rccl"
    }
}

impl Drop for HipMesh {
    fn drop(&mut self) {
        for slot in &self.comms {
            if let Some(&c) = slot.get() {
                if !c.is_null() {
                    // Best-effort; destroy errors at drop time are not actionable.
                    let _ = unsafe { ncclCommDestroy(c) };
                }
            }
        }
    }
}

/// Per-rank handle against a `HipMesh`. Clonable.
#[derive(Clone)]
pub struct HipRankHandle {
    pub rank: RankId,
    mesh: Arc<HipMesh>,
}

impl HipRankHandle {
    /// Bind the calling thread to this rank's device and call
    /// `ncclCommInitRank`. Blocks until every rank in the mesh has done the
    /// same. Must be called exactly once per rank, from that rank's driver
    /// thread, before any collective is issued.
    pub fn connect(&self) -> CollectiveResult<()> {
        let device_id = self.mesh.device_id(self.rank);
        crate::device::bind(device_id).map_err(|e| CollectiveError::Device {
            backend: "hip",
            ctx: "connect:bind",
            message: format!("{e}"),
        })?;
        let mut comm: ncclComm_t = ptr::null_mut();
        let code = unsafe {
            ncclCommInitRank(
                &mut comm as *mut _,
                self.mesh.comms.len() as c_int,
                self.mesh.unique_id,
                self.rank.0 as c_int,
            )
        };
        rccl_check(code, "ncclCommInitRank")?;
        self.mesh.comms[self.rank.0 as usize]
            .set(comm)
            .map_err(|_comm| CollectiveError::Device {
                backend: "hip",
                ctx: "connect:set",
                message: "rank already initialised".into(),
            })?;
        Ok(())
    }

    pub fn rank_count(&self) -> u32 {
        self.mesh.rank_count()
    }

    pub fn device_id(&self) -> i32 {
        self.mesh.device_id(self.rank)
    }

    fn comm(&self) -> ncclComm_t {
        *self
            .mesh
            .comms[self.rank.0 as usize]
            .get()
            .expect("comm not initialised — HipMesh::new should have connected all ranks")
    }

    // ---- device-resident collectives (the "real" hot path) ----

    /// AllReduce on device pointers. Buffer sizes are expressed in *elements*
    /// via `cfg`; `d_buf` must point to at least `cfg.buffer_bytes()` of the
    /// rank's own device memory.
    ///
    /// # Safety
    /// `d_buf` must be a valid device pointer on this rank's device, and the
    /// rank's HIP context must be bound (`hipSetDevice`) on the calling thread.
    pub unsafe fn all_reduce_device(
        &self,
        d_buf: DevicePtr,
        cfg: &CollectiveCfg,
        stream: &<HipDevice as Device>::Stream,
    ) -> CollectiveResult<()> {
        let code = unsafe {
            ncclAllReduce(
                d_buf.as_usize() as *const _,
                d_buf.as_usize() as *mut _,
                cfg.elem_count,
                to_nccl_dtype(cfg.dtype),
                to_nccl_op(cfg.op),
                self.comm(),
                stream_raw(stream),
            )
        };
        rccl_check(code, "ncclAllReduce")
    }

    /// # Safety
    /// `d_send` and `d_recv` must be valid device pointers on this rank.
    /// `d_recv` must have `rank_count * cfg.buffer_bytes()` of space.
    pub unsafe fn all_gather_device(
        &self,
        d_send: DevicePtr,
        d_recv: DevicePtr,
        cfg: &CollectiveCfg,
        stream: &<HipDevice as Device>::Stream,
    ) -> CollectiveResult<()> {
        let code = unsafe {
            ncclAllGather(
                d_send.as_usize() as *const _,
                d_recv.as_usize() as *mut _,
                cfg.elem_count,
                to_nccl_dtype(cfg.dtype),
                self.comm(),
                stream_raw(stream),
            )
        };
        rccl_check(code, "ncclAllGather")
    }

    /// # Safety
    /// `d_buf` must be a valid device pointer on this rank with space for
    /// `cfg.buffer_bytes()` bytes.
    pub unsafe fn broadcast_device(
        &self,
        d_buf: DevicePtr,
        root: RankId,
        cfg: &CollectiveCfg,
        stream: &<HipDevice as Device>::Stream,
    ) -> CollectiveResult<()> {
        let code = unsafe {
            ncclBroadcast(
                d_buf.as_usize() as *const _,
                d_buf.as_usize() as *mut _,
                cfg.elem_count,
                to_nccl_dtype(cfg.dtype),
                root.0 as c_int,
                self.comm(),
                stream_raw(stream),
            )
        };
        rccl_check(code, "ncclBroadcast")
    }

    /// AllToAll — send[r * shard..(r+1)*shard] goes to rank r, recv[s * shard..(s+1)*shard]
    /// comes from rank s. Implemented as N² ncclSend/ncclRecv inside a group.
    ///
    /// # Safety
    /// `d_send` and `d_recv` must be valid device pointers on this rank, each
    /// with `rank_count * cfg.buffer_bytes()` bytes of space.
    pub unsafe fn all_to_all_device(
        &self,
        d_send: DevicePtr,
        d_recv: DevicePtr,
        cfg: &CollectiveCfg,
        stream: &<HipDevice as Device>::Stream,
    ) -> CollectiveResult<()> {
        let n = self.rank_count() as usize;
        let shard_bytes = cfg.buffer_bytes();
        let dtype = to_nccl_dtype(cfg.dtype);
        let stream_raw = stream_raw(stream);
        let comm = self.comm();

        rccl_check(unsafe { ncclGroupStart() }, "ncclGroupStart")?;
        for peer in 0..n {
            let peer_c = peer as c_int;
            let send_ptr = d_send.offset_bytes(peer * shard_bytes).as_usize();
            let recv_ptr = d_recv.offset_bytes(peer * shard_bytes).as_usize();
            let code = unsafe {
                ncclSend(
                    send_ptr as *const _,
                    cfg.elem_count,
                    dtype,
                    peer_c,
                    comm,
                    stream_raw,
                )
            };
            rccl_check(code, "ncclSend")?;
            let code = unsafe {
                ncclRecv(
                    recv_ptr as *mut _,
                    cfg.elem_count,
                    dtype,
                    peer_c,
                    comm,
                    stream_raw,
                )
            };
            rccl_check(code, "ncclRecv")?;
        }
        rccl_check(unsafe { ncclGroupEnd() }, "ncclGroupEnd")?;
        Ok(())
    }

    // ---- host-bounce convenience wrappers (cert harness) ----

    /// Upload → RCCL AllReduce → download. Convenience for tests / the cert
    /// harness. Not on the hot path.
    pub fn all_reduce_host(
        &self,
        dev: &HipDevice,
        buf: &mut [u8],
        cfg: &CollectiveCfg,
    ) -> CollectiveResult<()> {
        let ctx = device_err_ctx("all_reduce_host");
        if buf.len() != cfg.buffer_bytes() {
            return Err(CollectiveError::BufferLen {
                expected_elems: cfg.elem_count,
                dtype: cfg.dtype,
                got: buf.len(),
            });
        }
        dev.bind().map_err(&ctx)?;
        let d_buf = dev.alloc(buf.len()).map_err(&ctx)?;
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::HostToDevice,
                d_buf,
                DevicePtr(buf.as_ptr() as usize),
                buf.len(),
            )
            .map_err(&ctx)?;
        }
        dev.default_stream().synchronize().map_err(&ctx)?;

        unsafe {
            self.all_reduce_device(d_buf, cfg, dev.default_stream())?;
        }
        dev.default_stream().synchronize().map_err(&ctx)?;

        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(buf.as_mut_ptr() as usize),
                d_buf,
                buf.len(),
            )
            .map_err(&ctx)?;
        }
        dev.default_stream().synchronize().map_err(&ctx)?;
        unsafe { dev.dealloc(d_buf, buf.len()).map_err(&ctx)? };
        Ok(())
    }
}

fn device_err_ctx(ctx: &'static str) -> impl Fn(flambeau_core::DeviceError) -> CollectiveError {
    move |e| CollectiveError::Device {
        backend: "hip",
        ctx,
        message: format!("{e}"),
    }
}

fn stream_raw(s: &<HipDevice as Device>::Stream) -> crate::sys::hipStream_t {
    s.raw_handle() as *mut std::os::raw::c_void
}
