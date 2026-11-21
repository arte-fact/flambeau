//! `HipKernels` — the HIP impl of the `core::KernelBackend` launch seam. Wraps
//! the existing `HipModule` / `HipKernel` / `KernelArgs` so the op-launch layer
//! and the sweep harness drive HIP through the portable trait.

use flambeau_core::{Device, DeviceResult, KernelBackend, KernelDescriptor, LaunchArgs, LaunchCfg, QMatMulCfg};

use crate::{HipDevice, HipKernel, HipModule, HipStream, KernelArgs};

impl<'a> LaunchArgs<'a> for KernelArgs<'a> {
    fn push_arg<T>(&mut self, v: &'a T) {
        self.push(v);
    }

    fn push_slot<T>(&mut self, v: &'a T, slot: flambeau_core::ScalarSlot) {
        // The inherent `KernelArgs::push_slot` records the (slot, arg_index) tag
        // for graph-capture rebind; UFCS picks it over this trait method.
        KernelArgs::push_slot(self, v, slot);
    }
}

/// HIP launch-plane marker (gfx906 family).
pub struct HipKernels;

impl KernelBackend for HipKernels {
    type Device = HipDevice;
    type Module = HipModule;
    type Kernel<'m> = HipKernel<'m>;
    type Args<'a> = KernelArgs<'a>;

    const WARP_SIZE: u32 = 64;

    fn bind(dev: &HipDevice) -> DeviceResult<()> {
        dev.bind()
    }

    fn dispatch_qmatmul(cfg: &QMatMulCfg) -> Option<&'static KernelDescriptor> {
        crate::impls::dispatch_qmatmul(cfg)
    }

    fn dispatch_attention_decode(kv_dtype: flambeau_core::QDtype, m: usize) -> Option<&'static KernelDescriptor> {
        crate::impls::dispatch_attention_decode(kv_dtype, m)
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
