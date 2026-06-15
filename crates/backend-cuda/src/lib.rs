//! flambeau-backend-cuda — CUDA driver-API backend.
//!
//! `CudaDevice` / `CudaStream` / `CudaEvent` implement the `core`
//! `Device` / `Stream` / `Event` seams; `CudaModule` / `CudaKernel` load and
//! launch `.cubin` modules. The op registry (`CudaOps`), cluster + NCCL
//! collectives, and the graph executor are not yet implemented.
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod device;
pub mod module;
pub mod sys;

pub use device::{device_count, CudaDevice, CudaEvent, CudaStream};
pub use module::{CudaKernel, CudaModule, FuncAttributes, KernelArgs, LaunchCfg};
