//! The launch-plane seam: module load + entry resolve + launch, abstracted over
//! backend. [`Device`] covers the data plane (alloc / stream / memcpy); this
//! covers the launch plane, so the op-launch layer and the sweep harness name
//! no concrete `HipModule` / `CudaModule`. One impl per backend (`HipKernels`,
//! `CudaKernels`) wraps the existing concrete types.

use crate::{Device, DeviceResult, KernelDescriptor, LaunchCfg, QMatMulCfg, ScalarSlot};

/// Push kernel arguments by reference without naming a backend's concrete args
/// type. Implemented for each backend's `KernelArgs` — a local trait on a
/// foreign type, which the orphan rule permits in the backend crate.
pub trait LaunchArgs<'a>: Default {
    /// Push an argument by reference; the referent must outlive the launch.
    fn push_arg<T>(&mut self, v: &'a T);

    /// Push an argument and tag it with a graph-capture [`ScalarSlot`] for
    /// per-replay rebind. The default ignores the slot (plain push) — correct
    /// for backends without a graph slot-map; HIP overrides it to record the
    /// tag so `HipGraphExec` can rebind the arg between replays.
    fn push_slot<T>(&mut self, v: &'a T, _slot: ScalarSlot) {
        self.push_arg(v);
    }
}

/// A GPU backend's launch plane: load a module, resolve an entry, launch it.
/// The associated `Device` ties it to the matching data-plane backend.
pub trait KernelBackend: 'static {
    type Device: Device;
    type Module;
    type Kernel<'m>;
    type Args<'a>: LaunchArgs<'a>;

    /// Bind `dev` as the calling thread's current device, so subsequent module
    /// loads / launches land on it (matters when several registries are built
    /// back-to-back on a multi-GPU cluster).
    /// # Errors
    /// Propagates the backend context-bind failure.
    fn bind(dev: &Self::Device) -> DeviceResult<()>;

    /// Pick the `QMatMul` kernel variant for `cfg` from this backend's
    /// committed dispatch table (`first_match` over `QMATMUL_GFX906` /
    /// `QMATMUL_SM86`). The arch-specific variant choice lives here, not in the
    /// op body — so the generic `qmatmul` op names no per-arch table.
    fn dispatch_qmatmul(cfg: &QMatMulCfg) -> Option<&'static KernelDescriptor>;

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
}
