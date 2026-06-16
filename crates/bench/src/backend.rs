//! Backend seam for the sweep / autotune harness.
//!
//! `core::Device` abstracts the data-plane (alloc / stream / memcpy), but the
//! launch-plane — module load + kernel launch — is per-backend with no shared
//! trait. [`BenchBackend`] is that missing seam, kept local to bench so the
//! runtime keeps its concrete `HipModule` / `CudaModule` (promoting this into
//! `core::Device` is deferred to P5, when the CUDA forward path needs it). One
//! impl per backend wraps the existing concrete types.

use flambeau_core::{Device, DeviceResult, LaunchCfg};

use crate::cert::PmcSnapshot;

/// Push kernel arguments by reference without naming a backend's concrete
/// `KernelArgs`. Implemented for each backend's `KernelArgs` — a local trait on
/// a foreign type, which the orphan rule permits.
pub trait LaunchArgs<'a>: Default {
    /// Push an argument by reference; the referent must outlive the launch.
    fn push_arg<T>(&mut self, v: &'a T);
}

/// A GPU backend the sweep / autotune harness can drive end-to-end.
pub trait BenchBackend: 'static {
    type Device: Device;
    type Module;
    type Kernel<'m>;
    type Args<'a>: LaunchArgs<'a>;

    /// Construct and bind device `id`.
    /// # Errors
    /// Propagates the backend device-open / bind failure.
    fn new_device(id: i32) -> DeviceResult<Self::Device>;

    /// Compiled kernel image (`.hsaco` / `.cubin`) for catalogue `stem`, if the
    /// kernel was compiled into this build.
    fn image(stem: &str) -> Option<&'static [u8]>;

    /// Load `image` as a module on `dev`.
    /// # Errors
    /// Propagates the backend module-load failure.
    fn load(dev: &Self::Device, image: &[u8]) -> DeviceResult<Self::Module>;

    /// Resolve the `extern "C"` entry `name` in `module`.
    /// # Errors
    /// Propagates the backend symbol-resolution failure.
    fn kernel<'m>(module: &'m Self::Module, name: &'static str) -> DeviceResult<Self::Kernel<'m>>;

    /// Launch `kernel` with `cfg` + `args` on `stream`.
    /// # Errors
    /// Propagates the backend launch failure.
    fn launch(
        kernel: &Self::Kernel<'_>,
        stream: &<Self::Device as Device>::Stream,
        cfg: LaunchCfg,
        args: Self::Args<'_>,
    ) -> DeviceResult<()>;

    /// Static per-kernel attributes mapped into the cert PMC fields. Register
    /// count → `vgpr_count`; runtime counters (`mem_busy`/`valu_busy`) stay
    /// `None` here — HIP fills them via rocprofv3, CUDA leaves them null.
    /// # Errors
    /// Propagates the backend attribute-query failure.
    fn attributes(kernel: &Self::Kernel<'_>) -> DeviceResult<PmcSnapshot>;
}

#[cfg(feature = "hip")]
mod hip_backend {
    use super::{BenchBackend, LaunchArgs, PmcSnapshot};
    use flambeau_backend_hip::{HipDevice, HipKernel, HipModule, KernelArgs};
    use flambeau_core::{Device, DeviceResult, LaunchCfg};

    impl<'a> LaunchArgs<'a> for KernelArgs<'a> {
        fn push_arg<T>(&mut self, v: &'a T) {
            self.push(v);
        }
    }

    /// HIP / gfx906 backend.
    pub struct Hip;

    impl BenchBackend for Hip {
        type Device = HipDevice;
        type Module = HipModule;
        type Kernel<'m> = HipKernel<'m>;
        type Args<'a> = KernelArgs<'a>;

        fn new_device(id: i32) -> DeviceResult<HipDevice> {
            let dev = HipDevice::new(id)?;
            dev.bind()?;
            Ok(dev)
        }
        fn image(stem: &str) -> Option<&'static [u8]> {
            flambeau_kernels_hip::hsaco(stem)
        }
        fn load(dev: &HipDevice, image: &[u8]) -> DeviceResult<HipModule> {
            HipModule::load(dev.id(), image)
        }
        fn kernel<'m>(module: &'m HipModule, name: &'static str) -> DeviceResult<HipKernel<'m>> {
            module.kernel(name)
        }
        fn launch(
            kernel: &HipKernel<'_>,
            stream: &<HipDevice as Device>::Stream,
            cfg: LaunchCfg,
            args: KernelArgs<'_>,
        ) -> DeviceResult<()> {
            // SAFETY: sweeps keep arg storage + device pointers alive until the
            // stream is synchronized after the launch (harness contract).
            unsafe { kernel.launch(stream, cfg, args) }
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
pub use hip_backend::Hip;

#[cfg(feature = "cuda")]
mod cuda_backend {
    use super::{BenchBackend, LaunchArgs, PmcSnapshot};
    use flambeau_backend_cuda::{CudaDevice, CudaKernel, CudaModule, KernelArgs};
    use flambeau_core::{Device, DeviceResult, LaunchCfg};

    impl<'a> LaunchArgs<'a> for KernelArgs<'a> {
        fn push_arg<T>(&mut self, v: &'a T) {
            self.push(v);
        }
    }

    /// CUDA / sm_86 backend.
    pub struct Cuda;

    impl BenchBackend for Cuda {
        type Device = CudaDevice;
        type Module = CudaModule;
        type Kernel<'m> = CudaKernel<'m>;
        type Args<'a> = KernelArgs<'a>;

        fn new_device(id: i32) -> DeviceResult<CudaDevice> {
            CudaDevice::new(id)
        }
        fn image(stem: &str) -> Option<&'static [u8]> {
            flambeau_kernels_cuda::cubin(stem)
        }
        fn load(dev: &CudaDevice, image: &[u8]) -> DeviceResult<CudaModule> {
            CudaModule::load(dev.id(), image)
        }
        fn kernel<'m>(module: &'m CudaModule, name: &'static str) -> DeviceResult<CudaKernel<'m>> {
            module.kernel(name)
        }
        fn launch(
            kernel: &CudaKernel<'_>,
            stream: &<CudaDevice as Device>::Stream,
            cfg: LaunchCfg,
            args: KernelArgs<'_>,
        ) -> DeviceResult<()> {
            kernel.launch(stream, cfg, args)
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
pub use cuda_backend::Cuda;
