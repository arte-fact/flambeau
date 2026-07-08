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
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;

// `CUstreamCaptureMode`. GLOBAL is the default whole-process capture mode.
// RELAXED lets the capturing thread call host APIs (pointer derivation,
// context bind, scratch-slot selection) interleaved with the enqueues — our
// slot-capture closure runs Rust between launches, so it needs RELAXED.
pub const CU_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;
pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
pub const CU_STREAM_CAPTURE_MODE_RELAXED: c_int = 2;
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

/// Opaque graph-node handle. `CUgraphNode` is `typedef struct CUgraphNode_st*`;
/// pass it opaquely like the other driver handles.
pub type CUgraphNode = *mut c_void;
/// `CUarray` opaque handle — only ever NULL for our device/host memcpys.
pub type CUarray = *mut c_void;

// `CUgraphNodeType` enum values from `cuda.h`. Only kernel + memcpy are
// emitted by our forward-path ops; the rest are enumerated for completeness.
pub const CU_GRAPH_NODE_TYPE_KERNEL: c_int = 0;
pub const CU_GRAPH_NODE_TYPE_MEMCPY: c_int = 1;

// `CUmemorytype` enum values from `cuda.h`.
pub const CU_MEMORYTYPE_HOST: c_uint = 0x01;
pub const CU_MEMORYTYPE_DEVICE: c_uint = 0x02;
pub const CU_MEMORYTYPE_ARRAY: c_uint = 0x03;
pub const CU_MEMORYTYPE_UNIFIED: c_uint = 0x04;

/// `CUDA_KERNEL_NODE_PARAMS` **v1** layout (`CUDA_KERNEL_NODE_PARAMS_st` /
/// `CUDA_KERNEL_NODE_PARAMS_v1` in `cuda.h`). Field order + types match the
/// header exactly — a mismatch silently writes garbage into the launch arg
/// frame. The default `CUDA_KERNEL_NODE_PARAMS` typedef in a modern `cuda.h`
/// aliases the *v2* struct (with trailing `kern` + `ctx` fields) and the
/// `cuGraphExecKernelNodeSetParams` macro redirects to `_v2`; we bind the
/// **base** symbol below (no `_v2`), whose driver entry point consumes this
/// v1 struct.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
pub struct CUDA_KERNEL_NODE_PARAMS {
    pub func: CUfunction,
    pub gridDimX: c_uint,
    pub gridDimY: c_uint,
    pub gridDimZ: c_uint,
    pub blockDimX: c_uint,
    pub blockDimY: c_uint,
    pub blockDimZ: c_uint,
    pub sharedMemBytes: c_uint,
    pub kernelParams: *mut *mut c_void,
    pub extra: *mut *mut c_void,
}

/// `CUDA_MEMCPY3D` (`CUDA_MEMCPY3D_st` / `CUDA_MEMCPY3D_v2` in `cuda.h`).
/// All offsets/pitches/extents are `size_t`. Field order matches the header
/// exactly. For a 1D copy we set `Height = Depth = 1`, `WidthInBytes = count`,
/// zero the offsets/pitches, and set only the src/dst memory-type + pointer
/// pair implied by the copy direction.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
pub struct CUDA_MEMCPY3D {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcZ: usize,
    pub srcLOD: usize,
    pub srcMemoryType: c_uint,
    pub srcHost: *const c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub reserved0: *mut c_void,
    pub srcPitch: usize,
    pub srcHeight: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstZ: usize,
    pub dstLOD: usize,
    pub dstMemoryType: c_uint,
    pub dstHost: *mut c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub reserved1: *mut c_void,
    pub dstPitch: usize,
    pub dstHeight: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
    pub Depth: usize,
}

extern "C" {
    // --- init / device / context ---
    pub fn cuInit(flags: c_uint) -> c_int;
    pub fn cuDeviceGetCount(count: *mut c_int) -> c_int;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> c_int;
    /// `*can = 1` iff `dev` can directly dereference allocations on `peer`.
    pub fn cuDeviceCanAccessPeer(can: *mut c_int, dev: CUdevice, peer: CUdevice) -> c_int;
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

    // --- graph capture / replay ---
    /// Begin recording work issued to `stream` into a graph.
    #[link_name = "cuStreamBeginCapture_v2"]
    pub fn cuStreamBeginCapture(stream: CUstream, mode: c_int) -> c_int;
    /// End capture; `*graph` receives the recorded graph.
    pub fn cuStreamEndCapture(stream: CUstream, graph: *mut CUgraph) -> c_int;
    pub fn cuGraphInstantiateWithFlags(exec: *mut CUgraphExec, graph: CUgraph, flags: u64) -> c_int;
    pub fn cuGraphLaunch(exec: CUgraphExec, stream: CUstream) -> c_int;
    #[link_name = "cuGraphDestroy"]
    pub fn cuGraphDestroy(graph: CUgraph) -> c_int;
    #[link_name = "cuGraphExecDestroy"]
    pub fn cuGraphExecDestroy(exec: CUgraphExec) -> c_int;

    // --- graph node introspection + per-node param rebind ---
    /// `nodes = null` writes the node count through `num_nodes`; otherwise
    /// fills `nodes` (in dispatch order) with up to `*num_nodes` handles.
    pub fn cuGraphGetNodes(graph: CUgraph, nodes: *mut CUgraphNode, num_nodes: *mut usize)
        -> c_int;
    pub fn cuGraphNodeGetType(node: CUgraphNode, ntype: *mut c_int) -> c_int;
    /// Read a kernel node's params (v1 struct). Used to seed the Rust-side
    /// shadow with the source graph's captured `kernelParams` pointer array —
    /// its entries are driver-owned stable storage, safe to reuse for the args
    /// a later `set_slot` leaves untouched. Bound to the **base** symbol so the
    /// v1 `CUDA_KERNEL_NODE_PARAMS` ABI matches (the `_v2` symbol expects the
    /// larger v2 struct).
    #[link_name = "cuGraphKernelNodeGetParams"]
    pub fn cuGraphKernelNodeGetParams(
        node: CUgraphNode,
        params: *mut CUDA_KERNEL_NODE_PARAMS,
    ) -> c_int;
    /// In-place kernel-node param update on an instantiated exec (v1 struct).
    /// Bound to the **base** symbol (NOT `_v2`) so the v1 ABI matches.
    #[link_name = "cuGraphExecKernelNodeSetParams"]
    pub fn cuGraphExecKernelNodeSetParams(
        exec: CUgraphExec,
        node: CUgraphNode,
        params: *const CUDA_KERNEL_NODE_PARAMS,
    ) -> c_int;
    /// In-place memcpy-node param update on an instantiated exec. `ctx` is the
    /// context the copy executes in (the capture-time current context).
    pub fn cuGraphExecMemcpyNodeSetParams(
        exec: CUgraphExec,
        node: CUgraphNode,
        copy_params: *const CUDA_MEMCPY3D,
        ctx: CUcontext,
    ) -> c_int;
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
