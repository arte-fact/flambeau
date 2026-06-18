//! CUDA-specialised op surface — the per-backend aliases over the
//! backend-generic launch layer (`crate::launch`) and registry.
//!
//! Mirrors `hip::` exactly: the op-launch bodies and the registry are shared and
//! generic over [`KernelBackend`](flambeau_core::KernelBackend); this module only
//! pins them to the CUDA launch plane (`CudaKernels`). No launch glue is copied
//! (CLAUDE.md rule 14).

use flambeau_backend_cuda::CudaKernels;
pub use flambeau_backend_cuda::{CudaDevice, CudaStream};

// Same backend-generic op bodies the HIP surface uses.
pub use crate::launch::{
    attention, cast, conv, mlp, moe, norm, pe, qmatmul, recurrent, router, sampling, softcap,
    softmax,
};

pub use crate::registry::{Registry, RegistryError as OpsRegistryError, KERNEL_STEMS};

/// CUDA single-session op registry — [`Registry`] specialised to the CUDA launch
/// backend. Op bodies hold `&CudaOpsRegistry`.
pub type CudaOpsRegistry = Registry<CudaKernels>;

/// `Ops` for the CUDA backend — the generic [`OpsImpl`](crate::OpsImpl) bound to
/// the CUDA launch plane. Same body as `HipOps`, different launch-plane marker.
pub type CudaOps<'a> = crate::ops_impl::OpsImpl<'a, CudaKernels>;
