//! Backend extension for the sweep / autotune harness.
//!
//! The launch plane (module load + kernel launch) now lives in
//! [`flambeau_core::KernelBackend`]; [`BenchBackend`] extends it with the two
//! harness-only operations — device construction and the PMC attribute query.
//! The `Hip` / `Cuda` markers are the backend crates' `KernelBackend` impls.

use flambeau_core::{DeviceResult, KernelBackend};
// Re-exported so sweep modules keep importing `crate::backend::LaunchArgs`.
pub use flambeau_core::LaunchArgs;

use crate::cert::PmcSnapshot;

/// A GPU backend the sweep / autotune harness can drive end-to-end: the
/// portable launch plane ([`KernelBackend`]) plus device construction and the
/// per-kernel attribute query mapped into the cert PMC fields.
pub trait BenchBackend: KernelBackend {
    /// Construct and bind device `id`.
    /// # Errors
    /// Propagates the backend device-open / bind failure.
    fn new_device(id: i32) -> DeviceResult<Self::Device>;

    /// Static per-kernel attributes mapped into the cert PMC fields. Register
    /// count → `vgpr_count`; runtime counters stay `None` here (HIP fills them
    /// via rocprofv3, CUDA leaves them null).
    /// # Errors
    /// Propagates the backend attribute-query failure.
    fn attributes(kernel: &Self::Kernel<'_>) -> DeviceResult<PmcSnapshot>;
}

#[cfg(feature = "hip")]
mod hip_backend {
    use super::{BenchBackend, PmcSnapshot};
    use flambeau_backend_hip::{HipDevice, HipKernel, HipKernels};
    use flambeau_core::DeviceResult;

    impl BenchBackend for HipKernels {
        fn new_device(id: i32) -> DeviceResult<HipDevice> {
            let dev = HipDevice::new(id)?;
            dev.bind()?;
            Ok(dev)
        }
        fn attributes(kernel: &HipKernel<'_>) -> DeviceResult<PmcSnapshot> {
            let a = kernel.attributes()?;
            Ok(PmcSnapshot {
                vgpr_count: Some(a.num_regs),
                sgpr_count: None,
                waves_per_simd: Some(a.gfx906_waves_per_simd()),
                mem_busy_pct: None,
                valu_busy_pct: None,
            })
        }
    }
}
#[cfg(feature = "hip")]
pub use flambeau_backend_hip::HipKernels as Hip;

#[cfg(feature = "cuda")]
mod cuda_backend {
    use super::{BenchBackend, PmcSnapshot};
    use flambeau_backend_cuda::{CudaDevice, CudaKernel, CudaKernels};
    use flambeau_core::DeviceResult;

    impl BenchBackend for CudaKernels {
        fn new_device(id: i32) -> DeviceResult<CudaDevice> {
            CudaDevice::new(id)
        }
        fn attributes(kernel: &CudaKernel<'_>) -> DeviceResult<PmcSnapshot> {
            let a = kernel.attributes()?;
            Ok(PmcSnapshot {
                vgpr_count: u32::try_from(a.num_regs).ok(),
                sgpr_count: None,
                waves_per_simd: None,
                mem_busy_pct: None,
                valu_busy_pct: None,
            })
        }
    }
}
#[cfg(feature = "cuda")]
pub use flambeau_backend_cuda::CudaKernels as Cuda;
