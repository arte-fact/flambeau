//! `CudaKernels` — the CUDA impl of the `core::KernelBackend` launch seam. Wraps
//! `CudaModule` / `CudaKernel` / `KernelArgs` so the op-launch layer and the
//! sweep harness drive CUDA through the portable trait.

use flambeau_core::{Device, DeviceResult, KernelBackend, LaunchArgs, LaunchCfg};

use crate::device::CudaStream;
use crate::module::{CudaKernel, CudaModule, KernelArgs};
use crate::CudaDevice;

impl<'a> LaunchArgs<'a> for KernelArgs<'a> {
    fn push_arg<T>(&mut self, v: &'a T) {
        self.push(v);
    }
}

/// CUDA launch-plane marker (sm_80+ family).
pub struct CudaKernels;

impl KernelBackend for CudaKernels {
    type Device = CudaDevice;
    type Module = CudaModule;
    type Kernel<'m> = CudaKernel<'m>;
    type Args<'a> = KernelArgs<'a>;

    fn bind(dev: &CudaDevice) -> DeviceResult<()> {
        dev.bind()
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
        stream: &CudaStream,
        cfg: LaunchCfg,
        args: KernelArgs<'_>,
    ) -> DeviceResult<()> {
        kernel.launch(stream, cfg, args)
    }
}
