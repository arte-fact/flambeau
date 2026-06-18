//! flambeau-backend-cuda — CUDA driver-API backend.
//!
//! `CudaDevice` / `CudaStream` / `CudaEvent` implement the `core`
//! `Device` / `Stream` / `Event` seams; `CudaModule` / `CudaKernel` load and
//! launch `.cubin` modules. `CudaCluster` implements the `runtime` `Cluster`
//! seam (+ NCCL `DeviceAllReduce` under the `nccl` feature). The op registry
//! (`CudaOps`) and graph executor are not yet implemented.
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod cluster;
pub mod device;
pub mod module;
#[cfg(feature = "nccl")]
pub mod nccl_sys;
pub mod sys;

pub use cluster::CudaCluster;
pub use device::{device_count, CudaDevice, CudaEvent, CudaStream};
pub use module::{CudaKernel, CudaModule, FuncAttributes, KernelArgs, LaunchCfg};
