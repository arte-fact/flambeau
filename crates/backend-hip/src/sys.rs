//! Minimal FFI to HIP runtime (`libamdhip64`).
//! We hand-write just the surface needs — device/stream/alloc/memcpy/sync
//! plus error text. No bindgen-shaped full header import: it pulls in
//! thousands of symbols we will never use, bloats build time, and makes the
//! ABI target version harder to reason about.
//! The `hipError_t` enum has hundreds of values; we treat it as an i32 and
//! rely on `hipGetErrorString` to format anything non-zero.


use std::os::raw::{c_char, c_int, c_uint, c_void};

pub const HIP_SUCCESS: c_int = 0;

/// `hipErrorPeerAccessAlreadyEnabled` from `hip/hip_runtime_api.h`.
/// Returned by `hipDeviceEnablePeerAccess` when the (current device, peer)
/// edge is already authorised — benign on cluster re-bind paths and
/// treated as success by `HipCluster::new`.
///
/// Code 705 is `hipErrorPeerAccessNotEnabled` (the opposite); using 705
/// here silently disabled BAR1 P2P on any second `HipCluster::new` over
/// the same devices — the AR fell back to host-bounce for any topology
/// where serve.rs's state-side cluster construction preceded
/// `try_build_bar_ar`'s.
pub const HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED: c_int = 704;

// Opaque stream handle. HIP defines `typedef struct ihipStream_t* HipStreamT`;
// from Rust we only ever pass it opaquely, so a void* newtype is enough.
pub type HipStreamT = *mut c_void;

// Opaque module + function handles from HipModuleT / HipFunctionT.
pub type HipModuleT = *mut c_void;
pub type HipFunctionT = *mut c_void;

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
        stream: HipStreamT,
    ) -> c_int;

    /// Cross-device peer copy. `dst_device_id` / `src_device_id` are the
    /// HIP device IDs holding the respective allocations; peer access
    /// must be enabled between them (`hipDeviceEnablePeerAccess`). On
    /// gfx906 + ROCm with BAR1 mapping (Above-4G + Resizable-BAR in
    /// BIOS) this routes via PCIe BAR1 with no host bounce — the
    /// pattern llama.cpp uses for its `-sm layer` PP handoff. Earlier
    /// concerns about `hipMemcpyPeerAsync` leaving the source stream
    /// in an unsync'able state (cluster.rs docstring) didn't reproduce
    /// in 2026-05-24 testing against llama.cpp; the call works
    /// reliably when enqueued on the producer stream + ordered with
    /// `hipEventRecord` + `hipStreamWaitEvent` on the consumer side.
    pub fn hipMemcpyPeerAsync(
        dst: *mut c_void,
        dst_device_id: c_int,
        src: *const c_void,
        src_device_id: c_int,
        size_bytes: usize,
        stream: HipStreamT,
    ) -> c_int;

    pub fn hipStreamCreate(stream: *mut HipStreamT) -> c_int;
    pub fn hipStreamCreateWithFlags(stream: *mut HipStreamT, flags: c_uint) -> c_int;
    pub fn hipStreamDestroy(stream: HipStreamT) -> c_int;
    pub fn hipStreamSynchronize(stream: HipStreamT) -> c_int;

    // Module / function launch surface. hipModuleLoadData parses an in-memory
    // ELF produced by `hipcc --cuda-device-only --no-gpu-bundle-output`.
    pub fn hipModuleLoadData(module: *mut HipModuleT, image: *const c_void) -> c_int;
    pub fn hipModuleUnload(module: HipModuleT) -> c_int;
    pub fn hipModuleGetFunction(
        func: *mut HipFunctionT,
        module: HipModuleT,
        name: *const c_char,
    ) -> c_int;
    pub fn hipModuleLaunchKernel(
        func: HipFunctionT,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: HipStreamT,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> c_int;

    pub fn hipFuncGetAttribute(value: *mut c_int, attrib: c_int, hfunc: HipFunctionT) -> c_int;

    // Pinned (page-locked) host memory — required by PP peer
    // copy host-bounce to keep DtoH + HtoD at full PCIe bandwidth.
    // Pageable memory forces the driver to stage through an internal
    // pinned buffer, halving throughput.
    pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
    pub fn hipHostFree(ptr: *mut c_void) -> c_int;

    // peer access for BAR1-mapped P2P kernels.
    // `hipDeviceCanAccessPeer(can, dev, peer)` writes 1 to `*can` if `dev`
    // is allowed to read/write `peer`'s memory through PCIe BAR1 once peer
    // access is enabled. On gfx906 PCIe-only rigs the matrix is symmetric
    // and dense (every pair returns 1) provided the BIOS exposes
    // Above-4G-Decoding + Resizable-BAR. If a pair returns 0, the BIOS or
    // motherboard topology blocks BAR1 mapping and the BAR1 P2P AllReduce
    // path must be skipped on that pair (host-bounce stays as fallback).
    // `hipDeviceEnablePeerAccess(peer, flags)` is a per-thread, per-device
    // operation: it grants the *current* HIP device (last `hipSetDevice`)
    // permission to dereference pointers owned by `peer`. `flags` is
    // reserved (HIP requires it to be 0). The "already enabled" return
    // (`hipErrorPeerAccessAlreadyEnabled`, code 705) is benign and is
    // treated as success by the cluster bring-up path.
    // Unlike `hipMemcpyPeerAsync`, which on this rig has been observed to
    // submit successfully but leave the source stream in an unsync'able
    // state (see `cluster.rs` header), the BAR1 *direct dereference*
    // mechanism enabled by these calls works reliably on the same
    // gfx906 + ROCm 7.1.x topology — mi50grad ships it in production
    // and flambeau ports the same path here for TP decode AllReduce.
    pub fn hipDeviceCanAccessPeer(
        can_access_peer: *mut c_int,
        device_id: c_int,
        peer_device_id: c_int,
    ) -> c_int;
    pub fn hipDeviceEnablePeerAccess(peer_device_id: c_int, flags: c_uint) -> c_int;
    pub fn hipDeviceDisablePeerAccess(peer_device_id: c_int) -> c_int;

    // 5.b — event primitives for cross-stream / cross-device DAG
    // scheduling (async peer-copy pipeline-parallel ubatch path).
    pub fn hipEventCreate(event: *mut HipEventT) -> c_int;
    pub fn hipEventCreateWithFlags(event: *mut HipEventT, flags: c_uint) -> c_int;
    pub fn hipEventDestroy(event: HipEventT) -> c_int;
    pub fn hipEventRecord(event: HipEventT, stream: HipStreamT) -> c_int;
    pub fn hipStreamWaitEvent(stream: HipStreamT, event: HipEventT, flags: c_uint) -> c_int;
    pub fn hipEventSynchronize(event: HipEventT) -> c_int;
    /// measure ms between two recorded events. Both
    /// events must have been created **without** `HIP_EVENT_DISABLE_TIMING`
    /// for the timestamp to be valid.
    pub fn hipEventElapsedTime(ms: *mut f32, start: HipEventT, stop: HipEventT) -> c_int;

    // 6.a — graph-capture primitives. Record a sequence of kernel
    // launches + memcpys on a stream once, instantiate into an executable
    // graph, and replay per ubatch. Collapses per-ubatch Rust FFI /
    // driver launch overhead to a single graph-replay call.
    pub fn hipStreamBeginCapture(stream: HipStreamT, mode: c_uint) -> c_int;
    /// capture stream work INTO an existing graph. Multiple
    /// streams can capture into the SAME `HipGraphT` simultaneously,
    /// resolving cross-stream events as internal graph edges. Beta API in
    /// ROCm 7.x; `dependencyData` must be NULL.
    pub fn hipStreamBeginCaptureToGraph(
        stream: HipStreamT,
        graph: HipGraphT,
        dependencies: *const HipGraphNodeT,
        dependency_data: *const c_void,
        num_dependencies: usize,
        mode: c_uint,
    ) -> c_int;
    pub fn hipStreamEndCapture(stream: HipStreamT, graph: *mut HipGraphT) -> c_int;
    /// create an empty graph for capture-to-graph use.
    pub fn hipGraphCreate(graph: *mut HipGraphT, flags: c_uint) -> c_int;
    pub fn hipGraphInstantiate(
        exec: *mut HipGraphExecT,
        graph: HipGraphT,
        err_node: *mut c_void,
        log_buf: *mut c_char,
        buf_size: usize,
    ) -> c_int;
    pub fn hipGraphLaunch(exec: HipGraphExecT, stream: HipStreamT) -> c_int;
    pub fn hipGraphExecDestroy(exec: HipGraphExecT) -> c_int;
    pub fn hipGraphDestroy(graph: HipGraphT) -> c_int;

    // 6.a-i2 — graph-node introspection + in-place param update on an
    // instantiated exec. Lets us capture a forward pass once and replay it
    // with updated scalar params (e.g. pos, start_position) per ubatch.
    pub fn hipGraphGetNodes(
        graph: HipGraphT,
        nodes: *mut HipGraphNodeT,
        num_nodes: *mut usize,
    ) -> c_int;
    pub fn hipGraphNodeGetType(node: HipGraphNodeT, ntype: *mut c_int) -> c_int;
    pub fn hipGraphKernelNodeGetParams(
        node: HipGraphNodeT,
        params: *mut hipKernelNodeParams,
    ) -> c_int;
    pub fn hipGraphExecKernelNodeSetParams(
        exec: HipGraphExecT,
        node: HipGraphNodeT,
        params: *const hipKernelNodeParams,
    ) -> c_int;

    // 6.a-i5b — 1D memcpy-node in-place update on an instantiated exec.
    // Used to retarget the KV-cache append memcpys per ubatch (dst is
    // pos-dependent; src and size stay fixed).
    pub fn hipGraphExecMemcpyNodeSetParams1D(
        exec: HipGraphExecT,
        node: HipGraphNodeT,
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: hipMemcpyKind,
    ) -> c_int;
}

// Opaque graph handles from hip_runtime_api.h.
pub type HipGraphT = *mut c_void;
pub type HipGraphExecT = *mut c_void;
pub type HipGraphNodeT = *mut c_void;

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

// Opaque event handle. HipEventT ≡ `struct ihipEvent_t *`.
pub type HipEventT = *mut c_void;

// Flag passed to hipEventCreateWithFlags for a latency-optimised event
// (no timing — we only use events for dependency tracking, not profiling).
pub const HIP_EVENT_DISABLE_TIMING: c_uint = 0x2;

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
