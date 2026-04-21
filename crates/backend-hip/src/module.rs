//! `HipModule` / `HipKernel` — safe wrappers around `hipModuleLoadData` +
//! `hipModuleGetFunction` + `hipModuleLaunchKernel`.
//!
//! A `HipModule` owns one loaded `.hsaco` code object per device. Kernel
//! handles hang off the module; they don't hold their own resources and are
//! cheap to derive.
//!
//! Design notes:
//! - Device context. Modules are loaded into the current HIP context's
//!   device. Call `HipDevice::bind()` on the owning device before constructing
//!   a module and before every launch — same rule as `HipDevice` itself.
//! - Args are type-erased at the ABI. `hipModuleLaunchKernel` takes a
//!   `void**` of argument pointers; we expose a tiny builder that takes
//!   anything `Copy` via `&T`, then addresses of those slots form the array.
//! - Stream ownership. `launch` takes `&HipStream`; the stream must live on
//!   the same device as the module.

use std::ffi::CString;
use std::marker::PhantomData;
use std::os::raw::c_uint;
use std::ptr;

use flambeau_core::{DeviceError, DeviceResult};

use crate::sys::{
    error_string, hipFuncGetAttribute, hipFunction_t, hipModuleGetFunction, hipModuleLaunchKernel,
    hipModuleLoadData, hipModuleUnload, hipModule_t, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, HIP_FUNC_ATTRIBUTE_NUM_REGS,
    HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, HIP_SUCCESS,
};
use crate::HipStream;
use flambeau_core::Stream;

const BACKEND: &str = "hip";

fn check(code: i32, ctx: &'static str) -> DeviceResult<()> {
    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(DeviceError::Backend {
            backend: BACKEND,
            code,
            message: format!("{ctx}: {}", error_string(code)),
        })
    }
}

/// A loaded HIP module (one `.hsaco` worth of kernels) tied to a device.
pub struct HipModule {
    raw: hipModule_t,
    device_id: i32,
}

unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}

impl HipModule {
    /// Load `image` (a slice of ELF bytes as produced by our `kernels-hip`
    /// crate) onto `device_id`. Caller must have `HipDevice::bind()` in
    /// effect for the current thread.
    pub fn load(device_id: i32, image: &[u8]) -> DeviceResult<Self> {
        let mut m: hipModule_t = ptr::null_mut();
        let code =
            unsafe { hipModuleLoadData(&mut m as *mut _, image.as_ptr() as *const _) };
        check(code, "hipModuleLoadData")?;
        Ok(Self {
            raw: m,
            device_id,
        })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Resolve a kernel by its extern-C symbol name.
    pub fn kernel(&self, name: &str) -> DeviceResult<HipKernel<'_>> {
        let cname = CString::new(name).map_err(|_| DeviceError::Backend {
            backend: BACKEND,
            code: -1,
            message: format!("kernel name contains NUL: {name:?}"),
        })?;
        let mut f: hipFunction_t = ptr::null_mut();
        let code = unsafe {
            hipModuleGetFunction(&mut f as *mut _, self.raw, cname.as_ptr())
        };
        check(code, "hipModuleGetFunction")?;
        Ok(HipKernel {
            raw: f,
            name: name.to_string(),
            _module: PhantomData,
        })
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { hipModuleUnload(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

/// A kernel handle derived from a `HipModule`. Borrows from the module so
/// the module stays alive as long as any kernel handle does.
pub struct HipKernel<'m> {
    raw: hipFunction_t,
    pub name: String,
    _module: PhantomData<&'m HipModule>,
}

/// Launch configuration — grid + block dimensions + dynamic shared mem.
#[derive(Debug, Clone, Copy)]
pub struct LaunchCfg {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_bytes: u32,
}

impl LaunchCfg {
    pub fn one_d(grid_x: u32, block_x: u32) -> Self {
        Self {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_bytes: 0,
        }
    }
}

/// Kernel argument buffer. Pushes pointer to each arg's storage; the
/// underlying storage must outlive the launch.
#[derive(Default)]
pub struct KernelArgs<'a> {
    ptrs: Vec<*mut std::os::raw::c_void>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> KernelArgs<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push an argument by reference. The reference must live until after
    /// the kernel has launched (note: HIP's launch captures the pointers;
    /// the launch itself is synchronous on the *host* side of copying, but
    /// the kernel reads them asynchronously, so arg storage must outlive
    /// the stream's consumption of the launch).
    pub fn push<T>(&mut self, v: &'a T) {
        self.ptrs.push(v as *const T as *mut _);
    }

    fn as_raw(&mut self) -> *mut *mut std::os::raw::c_void {
        self.ptrs.as_mut_ptr()
    }

    /// Same as `as_raw` but exposed publicly so callers that reuse a
    /// `KernelArgs` across many launches can pass the pointer to
    /// [`HipKernel::launch_raw`] directly. See V2.1 docs: this saves
    /// one Vec allocation + N `push()` calls per launch when the arg
    /// layout is stable (common case: same kernel, same shape, different
    /// device pointers mutated in-place by the caller).
    pub fn raw_ptrs(&mut self) -> *mut *mut std::os::raw::c_void {
        self.ptrs.as_mut_ptr()
    }

    /// Overwrite slot `idx` with a new `&T`. Caller must ensure `idx` is
    /// less than the number of previously-pushed slots. Used by reuse-pool
    /// callers that construct once and update pointers per launch.
    pub fn set<T>(&mut self, idx: usize, v: &'a T) {
        debug_assert!(idx < self.ptrs.len(), "set({idx}) out of bounds");
        self.ptrs[idx] = v as *const T as *mut _;
    }

    pub fn len(&self) -> usize {
        self.ptrs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ptrs.is_empty()
    }
}

/// Per-kernel static attributes queried via `hipFuncGetAttribute` — the
/// first-order PMC inputs (VGPR count, shared-mem footprint, register /
/// occupancy ceiling).
#[derive(Debug, Clone, Copy)]
pub struct FuncAttributes {
    pub num_regs: u32,
    pub shared_size_bytes: u32,
    pub local_size_bytes: u32,
    pub max_threads_per_block: u32,
}

impl FuncAttributes {
    /// Occupancy ceiling on gfx906: 256 VGPR / SIMD, 10 waves / SIMD
    /// (architectural limit). `waves_per_simd = min(10, 256 / num_regs)`,
    /// rounded down. Matches AMD ISA §3.7.4.
    pub fn gfx906_waves_per_simd(&self) -> u32 {
        if self.num_regs == 0 {
            return 10;
        }
        (256 / self.num_regs).min(10)
    }
}

impl<'m> HipKernel<'m> {
    /// Query the kernel's static attributes — VGPR count, shared-mem
    /// footprint, etc. Cheap in-process call; no kernel launch.
    pub fn attributes(&self) -> DeviceResult<FuncAttributes> {
        fn q(raw: hipFunction_t, attr: std::os::raw::c_int, ctx: &'static str) -> DeviceResult<i32> {
            let mut v: std::os::raw::c_int = 0;
            let code = unsafe { hipFuncGetAttribute(&mut v as *mut _, attr, raw) };
            check(code, ctx)?;
            Ok(v as i32)
        }
        let num_regs = q(self.raw, HIP_FUNC_ATTRIBUTE_NUM_REGS, "hipFuncGetAttribute NUM_REGS")?
            .max(0) as u32;
        let shared = q(self.raw, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, "hipFuncGetAttribute SHARED")?
            .max(0) as u32;
        let local = q(self.raw, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, "hipFuncGetAttribute LOCAL")?
            .max(0) as u32;
        let max_tpb = q(
            self.raw,
            HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            "hipFuncGetAttribute MAX_TPB",
        )?
        .max(0) as u32;
        Ok(FuncAttributes {
            num_regs,
            shared_size_bytes: shared,
            local_size_bytes: local,
            max_threads_per_block: max_tpb,
        })
    }

    /// Launch the kernel on `stream` with `cfg` and the arguments packed in
    /// `args`.
    ///
    /// # Safety
    /// - Argument storage referenced by `args` must live until the stream
    ///   consumes the launch (typically: until the subsequent `synchronize`).
    /// - `cfg.grid` × `cfg.block` must not exceed device limits.
    /// - Any device pointers in `args` must be valid on this kernel's device.
    pub unsafe fn launch(
        &self,
        stream: &HipStream,
        cfg: LaunchCfg,
        mut args: KernelArgs<'_>,
    ) -> DeviceResult<()> {
        let code = unsafe {
            hipModuleLaunchKernel(
                self.raw,
                cfg.grid.0 as c_uint,
                cfg.grid.1 as c_uint,
                cfg.grid.2 as c_uint,
                cfg.block.0 as c_uint,
                cfg.block.1 as c_uint,
                cfg.block.2 as c_uint,
                cfg.shared_bytes as c_uint,
                stream.raw_handle() as crate::sys::hipStream_t,
                args.as_raw(),
                ptr::null_mut(),
            )
        };
        check(code, "hipModuleLaunchKernel")
    }

    /// Lower-latency launch: caller pre-built the arg pointer array.
    /// Used by V2.1's reuse-pool pattern — callers that launch the same
    /// kernel many times with a stable arg layout can construct a
    /// `KernelArgs` once, call `.raw_ptrs()` once, and invoke this in
    /// a tight loop without re-running the builder each iteration.
    ///
    /// # Safety
    /// - `args_ptr` must point to a valid array of at least the number
    ///   of arguments this kernel expects.
    /// - Each entry must point to storage matching the kernel's signature
    ///   at the corresponding position, and that storage must remain live
    ///   until the stream consumes the launch.
    pub unsafe fn launch_raw(
        &self,
        stream: &HipStream,
        cfg: LaunchCfg,
        args_ptr: *mut *mut std::os::raw::c_void,
    ) -> DeviceResult<()> {
        let code = unsafe {
            hipModuleLaunchKernel(
                self.raw,
                cfg.grid.0 as c_uint,
                cfg.grid.1 as c_uint,
                cfg.grid.2 as c_uint,
                cfg.block.0 as c_uint,
                cfg.block.1 as c_uint,
                cfg.block.2 as c_uint,
                cfg.shared_bytes as c_uint,
                stream.raw_handle() as crate::sys::hipStream_t,
                args_ptr,
                ptr::null_mut(),
            )
        };
        check(code, "hipModuleLaunchKernel")
    }
}
