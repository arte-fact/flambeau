//! `CudaModule` / `CudaKernel` — load a `.cubin` and launch kernels by name.

use std::collections::HashMap;
use std::ffi::CString;
use std::marker::PhantomData;
use std::sync::RwLock;

use flambeau_core::DeviceResult;

use crate::device::CudaStream;
use crate::sys::{
    cuFuncGetAttribute, cuLaunchKernel, cuModuleGetFunction, cuModuleLoadData,
    cuModuleUnload, error_string, CUfunction, CUmodule, CUDA_SUCCESS,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    CU_FUNC_ATTRIBUTE_NUM_REGS, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
};

const BACKEND: &str = "cuda";

fn check(code: std::os::raw::c_int, ctx: &'static str) -> DeviceResult<()> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(flambeau_core::DeviceError::Backend {
            backend: BACKEND,
            code,
            message: format!("{ctx}: {}", error_string(code)),
        })
    }
}

/// A loaded `.cubin` code object on one device. Kernel handles are cached by
/// `&'static str` name (every launch site is a string literal), so the cache
/// is uncontended once warm.
pub struct CudaModule {
    module: CUmodule,
    device_id: i32,
    cache: RwLock<HashMap<&'static str, CUfunction>>,
}

// SAFETY: `CUmodule`/`CUfunction` are opaque driver handles, valid across
// threads once loaded; the cache is `RwLock`-guarded. The module owns its
// handle and unloads it on drop.
unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl CudaModule {
    /// Load a cubin image into a module on the current context. Caller must
    /// have the owning device's context current.
    pub fn load(device_id: i32, image: &[u8]) -> DeviceResult<Self> {
        let mut module: CUmodule = std::ptr::null_mut();
        // SAFETY: `cuModuleLoadData` reads the image bytes and writes a module
        // handle through the out-pointer. `image` outlives the call.
        check(
            unsafe { cuModuleLoadData(&raw mut module, image.as_ptr() as *const _) },
            "cuModuleLoadData",
        )?;
        Ok(Self { module, device_id, cache: RwLock::new(HashMap::new()) })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Resolve a kernel by `&'static str` name, caching the handle.
    pub fn kernel(&self, name: &'static str) -> DeviceResult<CudaKernel<'_>> {
        if let Some(&raw) = self.cache.read().expect("module cache poisoned").get(name) {
            return Ok(CudaKernel { raw, name, _module: PhantomData });
        }
        let raw = self.resolve(name)?;
        self.cache.write().expect("module cache poisoned").insert(name, raw);
        Ok(CudaKernel { raw, name, _module: PhantomData })
    }

    fn resolve(&self, name: &str) -> DeviceResult<CUfunction> {
        let cname = CString::new(name).map_err(|_| flambeau_core::DeviceError::Backend {
            backend: BACKEND,
            code: -1,
            message: format!("kernel name {name:?} has an interior NUL"),
        })?;
        let mut func: CUfunction = std::ptr::null_mut();
        // SAFETY: writes a function handle through the out-pointer; `cname`
        // outlives the call.
        check(
            unsafe { cuModuleGetFunction(&raw mut func, self.module, cname.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(func)
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            // SAFETY: `self.module` was returned by `cuModuleLoadData`, owned,
            // not aliased.
            let _ = unsafe { cuModuleUnload(self.module) };
            self.module = std::ptr::null_mut();
        }
    }
}

/// A kernel handle borrowed from a `CudaModule`. Zero-copy; never Clone.
pub struct CudaKernel<'m> {
    raw: CUfunction,
    pub name: &'static str,
    _module: PhantomData<&'m CudaModule>,
}

impl std::fmt::Debug for CudaKernel<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaKernel")
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
        Self { grid: (grid_x, 1, 1), block: (block_x, 1, 1), shared_bytes: 0 }
    }
}

/// Kernel argument buffer. Stores a pointer to each arg's storage; the
/// underlying storage must outlive the launch (the kernel reads it
/// asynchronously on the stream).
#[derive(Default)]
pub struct KernelArgs<'a> {
    ptrs: Vec<*mut std::os::raw::c_void>,
    _marker: PhantomData<&'a ()>,
}

impl std::fmt::Debug for KernelArgs<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelArgs").field("len", &self.ptrs.len()).finish()
    }
}

impl<'a> KernelArgs<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push an argument by reference; it must live until the launch retires.
    pub fn push<T>(&mut self, v: &'a T) {
        self.ptrs.push(std::ptr::from_ref::<T>(v) as *mut _);
    }

    /// Overwrite arg `idx` (reuse-pool callers that update pointers per launch).
    pub fn set<T>(&mut self, idx: usize, v: &'a T) {
        debug_assert!(idx < self.ptrs.len(), "set({idx}) out of bounds");
        self.ptrs[idx] = std::ptr::from_ref::<T>(v) as *mut _;
    }

    pub fn raw_ptrs(&mut self) -> *mut *mut std::os::raw::c_void {
        self.ptrs.as_mut_ptr()
    }

    pub fn len(&self) -> usize {
        self.ptrs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ptrs.is_empty()
    }
}

/// Per-kernel static attributes from `cuFuncGetAttribute`.
#[derive(Debug, Clone, Copy)]
pub struct FuncAttributes {
    pub num_regs: i32,
    pub shared_size_bytes: i32,
    pub local_size_bytes: i32,
    pub max_threads_per_block: i32,
}

impl CudaKernel<'_> {
    /// Query static kernel attributes (register/shared/local usage).
    pub fn attributes(&self) -> DeviceResult<FuncAttributes> {
        let get = |attr: std::os::raw::c_int, ctx: &'static str| -> DeviceResult<i32> {
            let mut v: std::os::raw::c_int = 0;
            // SAFETY: writes an int through the out-pointer; `self.raw` is live.
            check(unsafe { cuFuncGetAttribute(&raw mut v, attr, self.raw) }, ctx)?;
            Ok(v)
        };
        Ok(FuncAttributes {
            num_regs: get(CU_FUNC_ATTRIBUTE_NUM_REGS, "cuFuncGetAttribute(NUM_REGS)")?,
            shared_size_bytes: get(
                CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                "cuFuncGetAttribute(SHARED_SIZE_BYTES)",
            )?,
            local_size_bytes: get(
                CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                "cuFuncGetAttribute(LOCAL_SIZE_BYTES)",
            )?,
            max_threads_per_block: get(
                CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                "cuFuncGetAttribute(MAX_THREADS_PER_BLOCK)",
            )?,
        })
    }

    /// Launch the kernel on `stream` with `cfg` + `args`.
    pub fn launch(&self, stream: &CudaStream, cfg: LaunchCfg, mut args: KernelArgs<'_>) -> DeviceResult<()> {
        let params = if args.is_empty() { std::ptr::null_mut() } else { args.raw_ptrs() };
        // SAFETY: `self.raw` is a live function handle; `params` points at
        // `args`'s pointer array, which outlives this synchronous enqueue. The
        // arg storage behind each pointer is the caller's responsibility to
        // keep alive until the stream retires (KernelArgs lifetime contract).
        check(
            unsafe {
                cuLaunchKernel(
                    self.raw,
                    cfg.grid.0,
                    cfg.grid.1,
                    cfg.grid.2,
                    cfg.block.0,
                    cfg.block.1,
                    cfg.block.2,
                    cfg.shared_bytes,
                    stream.raw(),
                    params,
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}
