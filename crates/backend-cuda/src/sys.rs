//! Hand-written FFI to the CUDA driver API (`libcuda`, `cu*`): the surface the
//! backend needs — device/context/stream/event/alloc/memcpy/module/launch +
//! error text.
//!
//! The 64-bit driver entry points are exported as `_v2` symbols (the bare
//! `cuMemAlloc` etc. are legacy 32-bit-pointer versions), bound here via
//! `#[link_name]`. `CUresult` is treated as i32 and formatted through
//! `cuGetErrorName` / `cuGetErrorString`.

use std::os::raw::{c_char, c_int, c_uint, c_void};

/// `CUDA_SUCCESS` from `cuda.h`.
pub const CUDA_SUCCESS: c_int = 0;

// Opaque driver handles. CUcontext/CUstream/CUmodule/CUfunction/CUevent are
// `typedef struct CU*_st* CU*` — pass them opaquely; a void* newtype is enough.
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
/// `CUdevice` is an `int` ordinal.
pub type CUdevice = c_int;
/// `CUdeviceptr` is `unsigned long long` on the 64-bit driver ABI (`_v2`).
pub type CUdeviceptr = u64;

// `CUstream_flags`: default serialises with the NULL stream; non-blocking does
// not (the concurrent-stream pool).
pub const CU_STREAM_DEFAULT: c_uint = 0;
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;

// `CUevent_flags`: disable-timing is the latency-optimised ordering event
// (we only `cuEventRecord` / `cuStreamWaitEvent` on the hot path).
pub const CU_EVENT_DEFAULT: c_uint = 0;
pub const CU_EVENT_DISABLE_TIMING: c_uint = 2;

// `CUfunction_attribute` enum values from `cuda.h`.
pub const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 0;
pub const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: c_int = 1;
pub const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: c_int = 3;
pub const CU_FUNC_ATTRIBUTE_NUM_REGS: c_int = 4;

extern "C" {
    // --- init / device / context ---
    pub fn cuInit(flags: c_uint) -> c_int;
    pub fn cuDeviceGetCount(count: *mut c_int) -> c_int;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> c_int;
    /// Retain the device's primary context; ref-counted, released by
    /// `cuDevicePrimaryCtxRelease`.
    pub fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, device: CUdevice) -> c_int;
    #[link_name = "cuDevicePrimaryCtxRelease_v2"]
    pub fn cuDevicePrimaryCtxRelease(device: CUdevice) -> c_int;
    /// Set `ctx` as the calling thread's current context.
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> c_int;
    pub fn cuCtxGetCurrent(ctx: *mut CUcontext) -> c_int;
    pub fn cuCtxSynchronize() -> c_int;
    pub fn cuGetErrorName(error: c_int, str_: *mut *const c_char) -> c_int;
    pub fn cuGetErrorString(error: c_int, str_: *mut *const c_char) -> c_int;

    // --- memory ---
    #[link_name = "cuMemAlloc_v2"]
    pub fn cuMemAlloc(dptr: *mut CUdeviceptr, bytesize: usize) -> c_int;
    #[link_name = "cuMemFree_v2"]
    pub fn cuMemFree(dptr: CUdeviceptr) -> c_int;
    #[link_name = "cuMemcpyHtoDAsync_v2"]
    pub fn cuMemcpyHtoDAsync(
        dst_device: CUdeviceptr,
        src_host: *const c_void,
        byte_count: usize,
        stream: CUstream,
    ) -> c_int;
    #[link_name = "cuMemcpyDtoHAsync_v2"]
    pub fn cuMemcpyDtoHAsync(
        dst_host: *mut c_void,
        src_device: CUdeviceptr,
        byte_count: usize,
        stream: CUstream,
    ) -> c_int;
    #[link_name = "cuMemcpyDtoDAsync_v2"]
    pub fn cuMemcpyDtoDAsync(
        dst_device: CUdeviceptr,
        src_device: CUdeviceptr,
        byte_count: usize,
        stream: CUstream,
    ) -> c_int;
    /// Unified-addressing async copy: under UVA, `dst`/`src` device addresses
    /// are globally unique, so the driver routes a cross-device copy by pointer
    /// — peer-direct when `cuCtxEnablePeerAccess` authorised the edge,
    /// host-staged otherwise. No peer `CUcontext` argument needed.
    pub fn cuMemcpyAsync(
        dst: CUdeviceptr,
        src: CUdeviceptr,
        byte_count: usize,
        stream: CUstream,
    ) -> c_int;

    // --- stream ---
    pub fn cuStreamCreate(stream: *mut CUstream, flags: c_uint) -> c_int;
    #[link_name = "cuStreamDestroy_v2"]
    pub fn cuStreamDestroy(stream: CUstream) -> c_int;
    pub fn cuStreamSynchronize(stream: CUstream) -> c_int;

    // --- event ---
    pub fn cuEventCreate(event: *mut CUevent, flags: c_uint) -> c_int;
    #[link_name = "cuEventDestroy_v2"]
    pub fn cuEventDestroy(event: CUevent) -> c_int;
    pub fn cuEventRecord(event: CUevent, stream: CUstream) -> c_int;
    pub fn cuStreamWaitEvent(stream: CUstream, event: CUevent, flags: c_uint) -> c_int;
    pub fn cuEventSynchronize(event: CUevent) -> c_int;
    pub fn cuEventElapsedTime(ms: *mut f32, start: CUevent, end: CUevent) -> c_int;

    // --- module / launch ---
    /// Load a `.cubin` / fatbin image from memory.
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> c_int;
    pub fn cuModuleUnload(module: CUmodule) -> c_int;
    pub fn cuModuleGetFunction(
        func: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> c_int;
    pub fn cuLaunchKernel(
        f: CUfunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> c_int;
    pub fn cuFuncGetAttribute(value: *mut c_int, attrib: c_int, func: CUfunction) -> c_int;
}

/// Format a `CUresult` for error messages. The driver returns the strings
/// through out-pointers and can itself fail for an unknown code — fall back to
/// the raw integer then.
pub fn error_string(code: c_int) -> String {
    if code == CUDA_SUCCESS {
        return "CUDA_SUCCESS".into();
    }
    // SAFETY: both calls write a pointer to static driver storage through the
    // out-pointer and read nothing from it; we null-check before deref.
    unsafe {
        let mut name: *const c_char = std::ptr::null();
        let mut desc: *const c_char = std::ptr::null();
        let name_ok = cuGetErrorName(code, &raw mut name) == CUDA_SUCCESS && !name.is_null();
        let desc_ok = cuGetErrorString(code, &raw mut desc) == CUDA_SUCCESS && !desc.is_null();
        let name = if name_ok {
            std::ffi::CStr::from_ptr(name).to_string_lossy().into_owned()
        } else {
            format!("CUresult {code}")
        };
        if desc_ok {
            let desc = std::ffi::CStr::from_ptr(desc).to_string_lossy();
            format!("{name}: {desc}")
        } else {
            name
        }
    }
}
