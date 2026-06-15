//! `HipModule` / `HipKernel` — safe wrappers around `hipModuleLoadData` +
//! `hipModuleGetFunction` + `hipModuleLaunchKernel`.
//! A `HipModule` owns one loaded `.hsaco` code object per device. Kernel
//! handles hang off the module; they don't hold their own resources and are
//! cheap to derive.
//! Design notes:
//! - Device context. Modules are loaded into the current HIP context's
//!   device. Call `HipDevice::bind()` on the owning device before constructing
//!   a module and before every launch — same rule as `HipDevice` itself.
//! - Args are type-erased at the ABI. `hipModuleLaunchKernel` takes a
//!   `void**` of argument pointers; we expose a tiny builder that takes
//!   anything `Copy` via `&T`, then addresses of those slots form the array.
//! - Stream ownership. `launch` takes `&HipStream`; the stream must live on
//!   the same device as the module.

use std::collections::HashMap;
use std::ffi::CString;
use std::marker::PhantomData;
use std::ptr;
use std::sync::RwLock;

use flambeau_core::{DeviceError, DeviceResult};

use crate::sys::{
    error_string, hipFuncGetAttribute, HipFunctionT, hipModuleGetFunction, hipModuleLaunchKernel,
    hipModuleLoadData, hipModuleUnload, HipModuleT, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
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
    raw: HipModuleT,
    device_id: i32,
    /// Resolved kernel-function handles, keyed by entry-point symbol name.
    /// The cache is insert-only in practice: every launch site in
    /// `crates/ops/src/hip/*` passes a string literal, and after warmup
    /// every entry point has been resolved once. The cache turns per-launch
    /// kernel resolution from `CString::new` + `hipModuleGetFunction` into
    /// an uncontended `RwLock::read` + `HashMap::get` — see [C1 in
    /// `RUST-PERF-CORRECTIONS.md`]. V2 continuous batching will contend the
    /// read lock at most N (num-ranks) ways; a `Mutex` would be fine today
    /// but `RwLock` is forward-compatible for cheap.
    kernel_cache: RwLock<HashMap<&'static str, HipFunctionT>>,
}

impl std::fmt::Debug for HipModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipModule")
            .field("raw", &(self.raw as usize))
            .field("device_id", &self.device_id)
            .field(
                "kernel_cache_size",
                &self
                    .kernel_cache
                    .read()
                    .map(|g| g.len())
                    .unwrap_or(usize::MAX),
            )
            .finish()
    }
}

// SAFETY: `HipModuleT` is an opaque driver handle with no Rust-side aliasing.
// HIP modules are immutable once loaded — kernel lookups on one module from
// multiple threads are safe per the HIP runtime contract. `HipModule` owns its
// handle and unloads on drop, so no cross-thread double-free risk.
unsafe impl Send for HipModule {}
// SAFETY: see `Send`. `&HipModule` only lets other threads resolve kernel
// symbols (`hipModuleGetFunction`) and read `device_id`; both are thread-safe.
unsafe impl Sync for HipModule {}

impl HipModule {
    /// Load `image` (a slice of ELF bytes as produced by our `kernels-hip`
    /// crate) onto `device_id`. Caller must have `HipDevice::bind()` in
    /// effect for the current thread.
    pub fn load(device_id: i32, image: &[u8]) -> DeviceResult<Self> {
        let mut m: HipModuleT = ptr::null_mut();
        // SAFETY: `hipModuleLoadData` reads the ELF image pointed to by
        // `image.as_ptr()` for its full length (driver-internal copy) and
        // writes a module handle through the out-pointer. `image` is a live
        // slice for the duration of this call; `&mut m` is valid for writes.
        let code = unsafe { hipModuleLoadData(&raw mut m, image.as_ptr().cast()) };
        check(code, "hipModuleLoadData")?;
        Ok(Self {
            raw: m,
            device_id,
            kernel_cache: RwLock::new(HashMap::new()),
        })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Resolve a kernel by its extern-C symbol name.
    /// Hot path: served from an uncontended read-lock on the per-module
    /// kernel cache. Cold path (first call for this name) goes through
    /// `hipModuleGetFunction` under the write lock and populates the cache.
    /// `name` must be a `'static` string — every call site in
    /// `crates/ops/src/hip/*` passes a literal, and the cache key borrows
    /// that literal without allocating. If `name` is genuinely dynamic,
    /// call [`Self::kernel_dynamic`] which pays the CString + String cost.
    /// # Errors
    /// Returns `DeviceError::Backend` if the name contains a NUL byte or
    /// the driver reports `hipModuleGetFunction` failure.
    pub fn kernel(&self, name: &'static str) -> DeviceResult<HipKernel<'_>> {
        if let Some(raw) = self
            .kernel_cache
            .read()
            .ok()
            .and_then(|g| g.get(name).copied())
        {
            return Ok(HipKernel {
                raw,
                name,
                _module: PhantomData,
            });
        }
        let raw = self.resolve(name)?;
        if let Ok(mut guard) = self.kernel_cache.write() {
            // `entry` not `insert` — tolerates a concurrent resolve during
            // the brief window between the read-miss and the write-acquire.
            guard.entry(name).or_insert(raw);
        }
        Ok(HipKernel {
            raw,
            name,
            _module: PhantomData,
        })
    }

    /// Resolve a kernel whose name is not known at compile time. Bypasses the
    /// kernel cache (the cache key is `&'static str`) — pays the `CString +
    /// String` cost per call. Prefer [`Self::kernel`] on the hot path.
    /// # Errors
    /// Same as [`Self::kernel`].
    pub fn kernel_dynamic(&self, name: &str) -> DeviceResult<HipKernel<'_>> {
        let raw = self.resolve(name)?;
        Ok(HipKernel {
            raw,
            // "<dyn>" is never read on the hot path; `HipKernel.name` is only
            // used by the `Debug` impl. Keeping it as `&'static str` avoids
            // a per-call allocation while the kernel handle itself carries
            // the identifying address.
            name: "<dyn>",
            _module: PhantomData,
        })
    }

    /// Raw driver resolution. Shared between `kernel` (cached) and
    /// `kernel_dynamic` (not cached).
    fn resolve(&self, name: &str) -> DeviceResult<HipFunctionT> {
        let cname = CString::new(name).map_err(|_nul| DeviceError::Backend {
            backend: BACKEND,
            code: -1,
            message: format!("kernel name contains NUL: {name:?}"),
        })?;
        let mut f: HipFunctionT = ptr::null_mut();
        // SAFETY: `self.raw` is a live module handle (owned, unloaded only on
        // drop). `cname` is a valid, NUL-terminated C string (constructed from
        // `CString::new` above). The driver writes the function handle through
        // `&mut f`, which is valid for writes.
        let code = unsafe { hipModuleGetFunction(&raw mut f, self.raw, cname.as_ptr()) };
        check(code, "hipModuleGetFunction")?;
        Ok(f)
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `self.raw` was returned by `hipModuleLoadData` in `load()`
            // and is not shared (no `Clone`). The null guard above is defensive
            // against a partially-constructed instance.
            let _ = unsafe { hipModuleUnload(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

/// A kernel handle derived from a `HipModule`. Borrows from the module so
/// the module stays alive as long as any kernel handle does.
/// `name` is `&'static str`: hot-path call sites pass string literals and
/// the cache keys borrow those literals. The `_dyn_name` field is kept
/// around as `None` in the hot path so `Debug`/error messages still work;
/// callers using [`HipModule::kernel_dynamic`] populate it.
pub struct HipKernel<'m> {
    raw: HipFunctionT,
    pub name: &'static str,
    _module: PhantomData<&'m HipModule>,
}

impl std::fmt::Debug for HipKernel<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipKernel")
            .field("raw", &(self.raw as usize))
            .field("name", &self.name)
            .finish()
    }
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
/// `push_slot` tags a pushed arg with a
/// [`ScalarSlot`](crate::graph_capture::ScalarSlot) so post-capture we
/// can bind the slot to its kernel-node arg index for later
/// `hipGraphExecKernelNodeSetParams` updates. `push` (untagged) is
/// unchanged.
#[derive(Default)]
pub struct KernelArgs<'a> {
    ptrs: Vec<*mut std::os::raw::c_void>,
    /// (slot, arg_index_within_this_launch) for every tagged push. Only
    /// drained into the thread-local capture state at launch time — so
    /// untagged launches pay zero cost.
    tagged_slots: Vec<(crate::graph_capture::ScalarSlot, usize)>,
    _marker: PhantomData<&'a ()>,
}

impl std::fmt::Debug for KernelArgs<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelArgs")
            .field("len", &self.ptrs.len())
            .field("tagged", &self.tagged_slots.len())
            .finish()
    }
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
        self.ptrs.push(std::ptr::from_ref::<T>(v) as *mut _);
    }

    /// Push an argument tagged as an updateable scalar. Same as `push`
    /// for the launch itself; additionally, during an active graph
    /// capture the tag is recorded so the exec can later update this
    /// arg in-place via `hipGraphExecKernelNodeSetParams`.
    /// Outside a capture scope this is functionally identical to
    /// `push` — the tag is stored in the `KernelArgs` but never
    /// consumed. Ops that want to remain captureable should use
    /// `push_slot` for every scalar that might vary per ubatch (pos,
    /// start_position, n_k_tokens, ...).
    pub fn push_slot<T>(&mut self, v: &'a T, slot: crate::graph_capture::ScalarSlot) {
        let arg_index = self.ptrs.len();
        self.push(v);
        self.tagged_slots.push((slot, arg_index));
    }

    fn as_raw(&mut self) -> *mut *mut std::os::raw::c_void {
        self.ptrs.as_mut_ptr()
    }

    /// Same as `as_raw` but exposed publicly so callers that reuse a
    /// `KernelArgs` across many launches can pass the pointer to
    /// [`HipKernel::launch_raw`] directly. See docs: this saves
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
        self.ptrs[idx] = std::ptr::from_ref::<T>(v) as *mut _;
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

impl HipKernel<'_> {
    /// Query the kernel's static attributes — VGPR count, shared-mem
    /// footprint, etc. Cheap in-process call; no kernel launch.
    pub fn attributes(&self) -> DeviceResult<FuncAttributes> {
        fn q(
            raw: HipFunctionT,
            attr: std::os::raw::c_int,
            ctx: &'static str,
        ) -> DeviceResult<i32> {
            let mut v: std::os::raw::c_int = 0;
            // SAFETY: `raw` is a live function handle (invariant of the enclosing
            // `HipKernel`, which borrows from its `HipModule`). `&mut v` is valid
            // for writes of `sizeof(int)`; the driver does not read from it.
            let code = unsafe { hipFuncGetAttribute(&raw mut v, attr, raw) };
            check(code, ctx)?;
            Ok(v)
        }
        let num_regs = q(
            self.raw,
            HIP_FUNC_ATTRIBUTE_NUM_REGS,
            "hipFuncGetAttribute NUM_REGS",
        )?
        .max(0) as u32;
        let shared = q(
            self.raw,
            HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
            "hipFuncGetAttribute SHARED",
        )?
        .max(0) as u32;
        let local = q(
            self.raw,
            HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            "hipFuncGetAttribute LOCAL",
        )?
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
        // If a capture is active on this thread, record the
        // launch's arity + tagged slots so post-capture the exec can build
        // a SlotMap. Outside capture this is a no-op.
        crate::graph_capture::record_launch(args.len(), &args.tagged_slots);

        // SAFETY: the outer fn is `unsafe`; the caller's contract (see doc
        // comment above) covers arg-storage lifetime, launch-cfg bounds, and
        // device-pointer validity. `self.raw` is live (borrowed from its
        // owning `HipModule` via `PhantomData`). `args.as_raw()` points into
        // `args.ptrs`, which lives until this function returns — `hipModule-
        // LaunchKernel` copies the pointer array synchronously before return.
        let code = unsafe {
            hipModuleLaunchKernel(
                self.raw,
                cfg.grid.0,
                cfg.grid.1,
                cfg.grid.2,
                cfg.block.0,
                cfg.block.1,
                cfg.block.2,
                cfg.shared_bytes,
                stream.raw_handle() as crate::sys::HipStreamT,
                args.as_raw(),
                ptr::null_mut(),
            )
        };
        if code == HIP_SUCCESS {
            Ok(())
        } else {
            Err(DeviceError::Backend {
                backend: BACKEND,
                code,
                message: format!(
                    "hipModuleLaunchKernel({}) grid={:?} block={:?}: {}",
                    self.name,
                    cfg.grid,
                    cfg.block,
                    error_string(code)
                ),
            })
        }
    }

    /// Lower-latency launch: caller pre-built the arg pointer array.
    /// Used by reuse-pool pattern — callers that launch the same
    /// kernel many times with a stable arg layout can construct a
    /// `KernelArgs` once, call `.raw_ptrs()` once, and invoke this in
    /// a tight loop without re-running the builder each iteration.
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
        // SAFETY: the outer fn is `unsafe`; the caller's contract (see doc
        // comment above) covers arg-array validity and per-slot storage
        // lifetime. `self.raw` is live (module borrow via `PhantomData`).
        // `hipModuleLaunchKernel` copies the arg pointer array synchronously
        // before return.
        let code = unsafe {
            hipModuleLaunchKernel(
                self.raw,
                cfg.grid.0,
                cfg.grid.1,
                cfg.grid.2,
                cfg.block.0,
                cfg.block.1,
                cfg.block.2,
                cfg.shared_bytes,
                stream.raw_handle() as crate::sys::HipStreamT,
                args_ptr,
                ptr::null_mut(),
            )
        };
        if code == HIP_SUCCESS {
            Ok(())
        } else {
            Err(DeviceError::Backend {
                backend: BACKEND,
                code,
                message: format!(
                    "hipModuleLaunchKernel({}) grid={:?} block={:?}: {}",
                    self.name,
                    cfg.grid,
                    cfg.block,
                    error_string(code)
                ),
            })
        }
    }
}
