//! `HipKernels` — the HIP impl of the `core::KernelBackend` launch seam. Wraps
//! the existing `HipModule` / `HipKernel` / `KernelArgs` so the op-launch layer
//! and the sweep harness drive HIP through the portable trait.

use flambeau_core::{Device, DeviceResult, KernelBackend, KernelDescriptor, LaunchArgs, LaunchCfg, QMatMulCfg};

use crate::{HipDevice, HipKernel, HipModule, HipStream, KernelArgs};

impl<'a> LaunchArgs<'a> for KernelArgs<'a> {
    fn push_arg<T>(&mut self, v: &'a T) {
        self.push(v);
    }
}

/// HIP launch-plane marker (gfx906 family).
pub struct HipKernels;

impl KernelBackend for HipKernels {
    type Device = HipDevice;
    type Module = HipModule;
    type Kernel<'m> = HipKernel<'m>;
    type Args<'a> = KernelArgs<'a>;

    fn bind(dev: &HipDevice) -> DeviceResult<()> {
        dev.bind()
    }

    fn dispatch_qmatmul(cfg: &QMatMulCfg) -> Option<&'static KernelDescriptor> {
        crate::impls::dispatch_qmatmul(cfg)
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
        stream: &HipStream,
        cfg: LaunchCfg,
        args: KernelArgs<'_>,
    ) -> DeviceResult<()> {
        // SAFETY: callers keep arg storage + device pointers alive until the
        // stream is synchronized after the launch (the op / sweep contract).
        unsafe { kernel.launch(stream, cfg, args) }
    }
}
