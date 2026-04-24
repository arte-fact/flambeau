//! flambeau-backend-hip — `HipDevice`, `HipStream`, minimal HIP runtime FFI,
//! and (behind the `rccl` feature) RCCL-backed collective impls registered
//! against `flambeau-runtime`'s op traits.
//!
//! V1.2 scope: device + stream + allocation + H↔D copy + RCCL AllReduce /
//! AllGather / AllToAll / Broadcast. V1.3+ adds the `KernelImpl<..., HipDevice>`
//! registrations alongside their `.cu` counterparts in `flambeau-kernels-hip`.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod cluster;
pub mod device;
pub mod impls;
pub mod module;
pub mod sys;

pub use cluster::HipCluster;
pub use impls::{dispatch_qmatmul, QMATMUL_GFX906};
pub use module::{FuncAttributes, HipKernel, HipModule, KernelArgs, LaunchCfg};

#[cfg(feature = "rccl")]
pub mod rccl;

#[cfg(feature = "rccl")]
pub mod rccl_sys;

pub use device::{bind, current_device, device_count, HipDevice, HipEvent, HipStream};

#[cfg(feature = "rccl")]
pub use rccl::{HipMesh, HipRankHandle};
