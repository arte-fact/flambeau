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

    // V2.25.b — event primitives for cross-stream / cross-device DAG
    // scheduling (async peer-copy pipeline-parallel ubatch path).
    pub fn hipEventCreate(event: *mut hipEvent_t) -> c_int;
    pub fn hipEventCreateWithFlags(event: *mut hipEvent_t, flags: c_uint) -> c_int;
    pub fn hipEventDestroy(event: hipEvent_t) -> c_int;
    pub fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> c_int;
    pub fn hipStreamWaitEvent(stream: hipStream_t, event: hipEvent_t, flags: c_uint) -> c_int;
    pub fn hipEventSynchronize(event: hipEvent_t) -> c_int;

    // V2.26.a — graph-capture primitives. Record a sequence of kernel
    // launches + memcpys on a stream once, instantiate into an executable
    // graph, and replay per ubatch. Collapses per-ubatch Rust FFI /
    // driver launch overhead to a single graph-replay call.
    pub fn hipStreamBeginCapture(stream: hipStream_t, mode: c_uint) -> c_int;
    pub fn hipStreamEndCapture(stream: hipStream_t, graph: *mut hipGraph_t) -> c_int;
    pub fn hipGraphInstantiate(
        exec: *mut hipGraphExec_t,
        graph: hipGraph_t,
        err_node: *mut c_void,
        log_buf: *mut c_char,
        buf_size: usize,
    ) -> c_int;
    pub fn hipGraphLaunch(exec: hipGraphExec_t, stream: hipStream_t) -> c_int;
    pub fn hipGraphExecDestroy(exec: hipGraphExec_t) -> c_int;
    pub fn hipGraphDestroy(graph: hipGraph_t) -> c_int;

    // V2.26.a-i2 — graph-node introspection + in-place param update on an
    // instantiated exec. Lets us capture a forward pass once and replay it
    // with updated scalar params (e.g. pos, start_position) per ubatch.
    pub fn hipGraphGetNodes(
        graph: hipGraph_t,
        nodes: *mut hipGraphNode_t,
        num_nodes: *mut usize,
    ) -> c_int;
    pub fn hipGraphNodeGetType(node: hipGraphNode_t, ntype: *mut c_int) -> c_int;
    pub fn hipGraphKernelNodeGetParams(
        node: hipGraphNode_t,
        params: *mut hipKernelNodeParams,
    ) -> c_int;
    pub fn hipGraphExecKernelNodeSetParams(
        exec: hipGraphExec_t,
        node: hipGraphNode_t,
        params: *const hipKernelNodeParams,
    ) -> c_int;

    // V2.26.a-i5b — 1D memcpy-node in-place update on an instantiated exec.
    // Used to retarget the KV-cache append memcpys per ubatch (dst is
    // pos-dependent; src and size stay fixed).
    pub fn hipGraphExecMemcpyNodeSetParams1D(
        exec: hipGraphExec_t,
        node: hipGraphNode_t,
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: hipMemcpyKind,
    ) -> c_int;
}

// Opaque graph handles from hip_runtime_api.h.
pub type hipGraph_t = *mut c_void;
pub type hipGraphExec_t = *mut c_void;
pub type hipGraphNode_t = *mut c_void;

/// `dim3` as laid out in `hip/amd_detail/amd_hip_runtime.h` — three u32s,
/// no padding. Same layout as CUDA's dim3. Align 4, size 12.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct hipDim3 {
    pub x: c_uint,
    pub y: c_uint,
    pub z: c_uint,
}

/// `hipKernelNodeParams` from `hip/hip_runtime_api.h`. Field order matches
/// the HIP header exactly — changing the order silently breaks the ABI.
/// The 4-byte pads after each `dim3` are inserted by repr(C) to align the
/// following pointer fields to 8.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct hipKernelNodeParams {
    pub block_dim: hipDim3,
    pub extra: *mut *mut c_void,
    pub func: *mut c_void,
    pub grid_dim: hipDim3,
    pub kernel_params: *mut *mut c_void,
    pub shared_mem_bytes: c_uint,
}

/// `hipGraphNodeType` enum values from hip_runtime_api.h. We only care
/// about Kernel for now; others listed for future use.
pub const HIP_GRAPH_NODE_TYPE_KERNEL: c_int = 0;
pub const HIP_GRAPH_NODE_TYPE_MEMCPY: c_int = 1;
pub const HIP_GRAPH_NODE_TYPE_MEMSET: c_int = 2;

// hipStreamCaptureMode — `hipStreamCaptureModeRelaxed` lets the capturing
// thread call host APIs that would otherwise trip "illegal during capture"
// guards. Required because our capture closure runs Rust code (pointer
// derivation, scratch slot selection) interleaved with kernel launches.
pub const HIP_STREAM_CAPTURE_MODE_GLOBAL: c_uint = 0;
pub const HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_uint = 1;
pub const HIP_STREAM_CAPTURE_MODE_RELAXED: c_uint = 2;

// Opaque event handle. hipEvent_t ≡ `struct ihipEvent_t *`.
pub type hipEvent_t = *mut c_void;

// Flag passed to hipEventCreateWithFlags for a latency-optimised event
// (no timing — we only use events for dependency tracking, not profiling).
pub const hipEventDisableTiming: c_uint = 0x2;

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
