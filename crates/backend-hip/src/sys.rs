//! Minimal FFI to HIP runtime (`libamdhip64`).
//!
//! We hand-write just the surface V1.2 needs — device/stream/alloc/memcpy/sync
//! plus error text. No bindgen-shaped full header import: it pulls in
//! thousands of symbols we will never use, bloats build time, and makes the
//! ABI target version harder to reason about.
//!
//! The `hipError_t` enum has hundreds of values; we treat it as an i32 and
//! rely on `hipGetErrorString` to format anything non-zero.

#![allow(
    non_camel_case_types,
    non_snake_case,
    reason = "hand-written FFI bindings mirror C HIP symbol names (hipError_t, \
              hipDeviceSynchronize, …). Renaming breaks one-to-one correspondence \
              with the upstream header and complicates diffs against ROCm releases."
)]

use std::os::raw::{c_char, c_int, c_uint, c_void};

pub const HIP_SUCCESS: c_int = 0;

// Opaque stream handle. HIP defines `typedef struct ihipStream_t* hipStream_t`;
// from Rust we only ever pass it opaquely, so a void* newtype is enough.
pub type hipStream_t = *mut c_void;

// Opaque module + function handles from hipModule_t / hipFunction_t.
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum hipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

extern "C" {
    pub fn hipGetDeviceCount(count: *mut c_int) -> c_int;
    pub fn hipSetDevice(device_id: c_int) -> c_int;
    pub fn hipGetDevice(device_id: *mut c_int) -> c_int;
    pub fn hipDeviceSynchronize() -> c_int;
    pub fn hipGetErrorString(error: c_int) -> *const c_char;

    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    pub fn hipFree(ptr: *mut c_void) -> c_int;

    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> c_int;

    pub fn hipStreamCreate(stream: *mut hipStream_t) -> c_int;
    pub fn hipStreamCreateWithFlags(stream: *mut hipStream_t, flags: c_uint) -> c_int;
    pub fn hipStreamDestroy(stream: hipStream_t) -> c_int;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> c_int;

    // Module / function launch surface. hipModuleLoadData parses an in-memory
    // ELF produced by `hipcc --cuda-device-only --no-gpu-bundle-output`.
    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> c_int;
    pub fn hipModuleUnload(module: hipModule_t) -> c_int;
    pub fn hipModuleGetFunction(
        func: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> c_int;
    pub fn hipModuleLaunchKernel(
        func: hipFunction_t,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> c_int;

    pub fn hipFuncGetAttribute(
        value: *mut c_int,
        attrib: c_int,
        hfunc: hipFunction_t,
    ) -> c_int;

    // Pinned (page-locked) host memory — required by V1.7.5.B's PP peer
    // copy host-bounce to keep DtoH + HtoD at full PCIe bandwidth.
    // Pageable memory forces the driver to stage through an internal
    // pinned buffer, halving throughput.
    pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
    pub fn hipHostFree(ptr: *mut c_void) -> c_int;
}

/// `hipHostMalloc` flag bits from `hip_runtime_api.h`. Use `Portable` to
/// make the pinned buffer usable from any HIP device in the cluster.
pub const HIP_HOST_MALLOC_DEFAULT: c_uint = 0;
pub const HIP_HOST_MALLOC_PORTABLE: c_uint = 1;

// `hipFunction_attribute` enum values, from `hip/driver_types.h`. We keep them
// here as explicit constants so the FFI is self-documenting.
pub const HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 0;
pub const HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: c_int = 1;
pub const HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: c_int = 3;
pub const HIP_FUNC_ATTRIBUTE_NUM_REGS: c_int = 4;

/// Convert a `hipError_t` return into a string for error messages.
pub fn error_string(code: c_int) -> String {
    if code == HIP_SUCCESS {
        return "hipSuccess".into();
    }
    // SAFETY: hipGetErrorString returns a pointer to static storage for every
    // known error code. For unknown codes the behaviour is implementation-
    // defined; we defensively check for null.
    unsafe {
        let p = hipGetErrorString(code);
        if p.is_null() {
            return format!("hipError {code} (no string)");
        }
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}
