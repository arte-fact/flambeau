//! `CudaCluster` — the CUDA rank set behind the [`flambeau_runtime::Cluster`]
//! seam: a list of `CudaDevice`s plus a peer-access matrix. Under the `nccl`
//! feature it also owns one `ncclComm` per rank and exposes the
//! [`DeviceAllReduce`] collective (`ncclAllReduce`, in place, Sum). Single
//! process, single node — `ncclCommInitAll` over the device ordinals; no BAR1
//! P2P (NCCL owns the PCIe/NVLink path).

use flambeau_core::DeviceResult;

use crate::device::{check, CudaDevice};
use crate::sys::cuDeviceCanAccessPeer;

/// The CUDA rank set. `devices[i]` drives rank `i`.
pub struct CudaCluster {
    devices: Vec<CudaDevice>,
    peer_access: Vec<Vec<bool>>,
    #[cfg(feature = "nccl")]
    comms: Vec<crate::nccl_sys::NcclComm>,
}

// SAFETY: `ncclComm` handles are driver-owned and used per-rank; the cluster is
// shared read-only after construction (collectives take `&self`).
unsafe impl Send for CudaCluster {}
unsafe impl Sync for CudaCluster {}

impl CudaCluster {
    /// Build a cluster over the given device ordinals. Probes the peer-access
    /// matrix and (under `nccl`) initialises one communicator per rank.
    pub fn new(device_ids: &[i32]) -> DeviceResult<Self> {
        assert!(!device_ids.is_empty(), "CudaCluster needs at least one device");
        let mut devices = Vec::with_capacity(device_ids.len());
        for &id in device_ids {
            devices.push(CudaDevice::new(id)?);
        }
        let peer_access = probe_peer_access(&devices)?;

        #[cfg(feature = "nccl")]
        let comms = init_nccl_comms(device_ids)?;

        Ok(CudaCluster {
            devices,
            peer_access,
            #[cfg(feature = "nccl")]
            comms,
        })
    }

    /// `peer_access[i][j]` — `true` iff rank `i`'s device can directly
    /// dereference rank `j`'s allocations (diagonal is `true`).
    pub fn peer_access_matrix(&self) -> &[Vec<bool>] {
        &self.peer_access
    }
}

fn probe_peer_access(devices: &[CudaDevice]) -> DeviceResult<Vec<Vec<bool>>> {
    let n = devices.len();
    let mut matrix = vec![vec![false; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                matrix[i][j] = true;
                continue;
            }
            let mut can: std::os::raw::c_int = 0;
            // SAFETY: both CUdevice handles are live for the cluster's lifetime.
            check(
                unsafe { cuDeviceCanAccessPeer(&raw mut can, devices[i].cu_device(), devices[j].cu_device()) },
                "cuDeviceCanAccessPeer",
            )?;
            matrix[i][j] = can != 0;
        }
    }
    Ok(matrix)
}

impl flambeau_runtime::Cluster for CudaCluster {
    type Device = CudaDevice;

    fn ranks(&self) -> usize {
        self.devices.len()
    }

    fn device(&self, rank: usize) -> &CudaDevice {
        &self.devices[rank]
    }

    fn peer_access_full(&self) -> bool {
        let n = self.devices.len();
        (0..n).all(|i| (0..n).all(|j| i == j || self.peer_access[i][j]))
    }
}

#[cfg(feature = "nccl")]
fn init_nccl_comms(device_ids: &[i32]) -> DeviceResult<Vec<crate::nccl_sys::NcclComm>> {
    use crate::nccl_sys::{ncclCommInitAll, NcclComm, NCCL_SUCCESS};
    let n = device_ids.len();
    let mut comms: Vec<NcclComm> = vec![std::ptr::null_mut(); n];
    let devlist: Vec<std::os::raw::c_int> = device_ids.iter().map(|&d| d as std::os::raw::c_int).collect();
    // SAFETY: comms has length n; devlist has length n of valid ordinals.
    let rc = unsafe { ncclCommInitAll(comms.as_mut_ptr(), n as std::os::raw::c_int, devlist.as_ptr()) };
    if rc != NCCL_SUCCESS {
        return Err(flambeau_core::DeviceError::Backend {
            backend: "cuda",
            code: rc,
            message: format!("ncclCommInitAll: {}", crate::nccl_sys::nccl_error_text(rc)),
        });
    }
    Ok(comms)
}

#[cfg(feature = "nccl")]
impl Drop for CudaCluster {
    fn drop(&mut self) {
        for &comm in &self.comms {
            if !comm.is_null() {
                // SAFETY: each comm came from ncclCommInitAll and is destroyed once.
                unsafe { crate::nccl_sys::ncclCommDestroy(comm) };
            }
        }
    }
}

#[cfg(feature = "nccl")]
pub use nccl_ar::CudaArRank;

#[cfg(feature = "nccl")]
mod nccl_ar {
    use super::CudaCluster;
    use crate::device::CudaDevice;
    use crate::nccl_sys::{ncclAllReduce, ncclGroupEnd, ncclGroupStart, NcclComm, NCCL_FLOAT16, NCCL_FLOAT32, NCCL_SUCCESS, NCCL_SUM};
    use flambeau_core::device::Device;
    use flambeau_core::DevicePtr;
    use flambeau_runtime::collective::{CollectiveError, CollectiveResult, DeviceAllReduce};
    use flambeau_runtime::CollectiveDType;

    impl CudaCluster {
        /// A per-rank collective handle bound to rank `rank`'s communicator.
        pub fn rank_handle(&self, rank: usize) -> CudaArRank<'_> {
            CudaArRank { comm: self.comms[rank], _cluster: std::marker::PhantomData }
        }

        /// Run `f` inside an `ncclGroupStart`/`ncclGroupEnd` pair — required when
        /// issuing collectives for >1 rank from a single thread so they do not
        /// deadlock (single-process multi-GPU / multi-rank-per-GPU idiom).
        pub fn grouped<T>(&self, f: impl FnOnce() -> CollectiveResult<T>) -> CollectiveResult<T> {
            nccl_check(unsafe { ncclGroupStart() }, "ncclGroupStart")?;
            let r = f();
            nccl_check(unsafe { ncclGroupEnd() }, "ncclGroupEnd")?;
            r
        }
    }

    fn nccl_check(rc: std::os::raw::c_int, ctx: &'static str) -> CollectiveResult<()> {
        if rc == NCCL_SUCCESS {
            Ok(())
        } else {
            Err(CollectiveError::Backend {
                backend: "cuda",
                ctx,
                code: rc,
                message: crate::nccl_sys::nccl_error_text(rc),
            })
        }
    }

    /// Per-rank NCCL collective handle. Borrows its cluster's communicator.
    #[derive(Copy, Clone)]
    pub struct CudaArRank<'a> {
        comm: NcclComm,
        _cluster: std::marker::PhantomData<&'a CudaCluster>,
    }

    impl DeviceAllReduce for CudaArRank<'_> {
        type Device = CudaDevice;

        fn all_reduce_sum(
            &self,
            buf: DevicePtr,
            n_elems: usize,
            dtype: CollectiveDType,
            device: &CudaDevice,
            stream: &<CudaDevice as Device>::Stream,
        ) -> CollectiveResult<()> {
            device.bind().map_err(|e| CollectiveError::Device {
                backend: "cuda",
                ctx: "bind",
                message: e.to_string(),
            })?;
            let dt = match dtype {
                CollectiveDType::F32 => NCCL_FLOAT32,
                CollectiveDType::F16 => NCCL_FLOAT16,
            };
            let ptr = buf.0 as *mut std::ffi::c_void;
            // In-place AllReduce: NCCL permits sendbuff == recvbuff.
            nccl_check(
                unsafe { ncclAllReduce(ptr, ptr, n_elems, dt, NCCL_SUM, self.comm, stream.raw() as *mut std::ffi::c_void) },
                "ncclAllReduce",
            )
        }
    }
}
