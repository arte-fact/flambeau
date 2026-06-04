//! `HipDevice` and `HipStream` — safe wrappers around `libamdhip64`.
//! One `HipDevice` ≈ one GPU. Internally it owns:
//! - a default stream (every implicit launch goes here unless the caller
//!   creates another stream);
//! - a device id used by `hipSetDevice` before any operation.
//!   The device is pinned to its id — `HipDevice::new(0)` sets the context to
//!   device 0 and expects every subsequent operation on that instance to happen
//!   on that device. Callers holding devices for multiple GPUs must call
//!   `bind()` (or its equivalent `hipSetDevice`) before issuing work on a
//!   device that isn't the current HIP context.

use std::os::raw::c_int;
use std::ptr;

use flambeau_core::{CopyDirection, Device, DeviceError, DevicePtr, DeviceResult, Stream};

use crate::sys::{
    self, error_string, hipFree, hipGetDevice, hipGetDeviceCount, hipMalloc, hipMemcpyAsync,
    hipMemcpyKind, hipSetDevice, hipStreamCreate, hipStreamCreateWithFlags, hipStreamDestroy,
    hipStreamSynchronize, HipStreamT, HIP_SUCCESS,
};

const BACKEND: &str = "hip";

#[cfg(feature = "dev_trace")]
fn dev_flag(name: &str) -> bool {
    std::env::var(name).is_ok()
}
#[cfg(not(feature = "dev_trace"))]
#[inline(always)]
fn dev_flag(_name: &str) -> bool {
    false
}

fn check(code: c_int, ctx: &'static str) -> DeviceResult<()> {
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

/// Number of HIP devices visible to this process.
pub fn device_count() -> DeviceResult<i32> {
    let mut count: c_int = 0;
    // SAFETY: `hipGetDeviceCount` writes an `int` through the out-pointer and
    // reads nothing from it. `&mut count` is valid for writes of `sizeof(int)`.
    let code = unsafe { hipGetDeviceCount(&raw mut count) };
    check(code, "hipGetDeviceCount")?;
    Ok(count)
}

/// Bind the current thread's HIP context to `device_id`.
pub fn bind(device_id: i32) -> DeviceResult<()> {
    // SAFETY: `hipSetDevice` takes an `int` by value and touches no caller memory.
    // A negative or out-of-range id is returned as an error via the return code.
    check(unsafe { hipSetDevice(device_id) }, "hipSetDevice")
}

/// Return the HIP context's current device id.
pub fn current_device() -> DeviceResult<i32> {
    let mut id: c_int = -1;
    // SAFETY: `hipGetDevice` writes an `int` through the out-pointer and reads
    // nothing from it. `&mut id` is valid for writes of `sizeof(int)`.
    check(unsafe { hipGetDevice(&raw mut id) }, "hipGetDevice")?;
    Ok(id)
}

/// A HIP stream. Drop destroys the underlying `HipStreamT`.
pub struct HipStream {
    ptr: HipStreamT,
    device_id: i32,
}

impl std::fmt::Debug for HipStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipStream")
            .field("ptr", &(self.ptr as usize))
            .field("device_id", &self.device_id)
            .finish()
    }
}

// SAFETY: `HipStreamT` is an opaque driver handle with no Rust-side aliasing.
// HIP streams are documented as safe to pass between threads (operations
// serialise within a stream driver-side). `HipStream` owns its handle and
// destroys it on drop, so there is no cross-thread double-free risk.
unsafe impl Send for HipStream {}
// SAFETY: See `Send`. `&HipStream` only lets other threads read the handle
// pointer and call `hipStreamSynchronize` / submit work — both are thread-safe
// per the HIP runtime contract.
unsafe impl Sync for HipStream {}

impl HipStream {
    /// Create a new stream on `device_id`. Caller must have `bind(device_id)`
    /// in effect for the current thread.
    /// NB: `hipStreamCreate` creates a **blocking** stream (serialises with
    /// the null stream). For truly-concurrent streams on the same device
    /// use [`Self::new_non_blocking`].
    pub fn new(device_id: i32) -> DeviceResult<Self> {
        let mut s: HipStreamT = ptr::null_mut();
        // SAFETY: `hipStreamCreate` writes a stream handle through the
        // out-pointer. `&mut s` is valid for writes of a `HipStreamT`.
        check(unsafe { hipStreamCreate(&raw mut s) }, "hipStreamCreate")?;
        Ok(Self { ptr: s, device_id })
    }

    /// 5.g — create a non-blocking stream (`hipStreamNonBlocking` = 1).
    /// These streams do NOT serialise with the null stream and can run
    /// concurrently with each other on the same device, subject to
    /// occupancy. Used by `HipCluster::reserve_aux_streams` so the
    /// 5.d async ubatch pipeline truly overlaps lanes on the same
    /// device.
    pub fn new_non_blocking(device_id: i32) -> DeviceResult<Self> {
        let mut s: HipStreamT = ptr::null_mut();
        const HIP_STREAM_NON_BLOCKING: std::os::raw::c_uint = 1;
        // SAFETY: `hipStreamCreateWithFlags` writes an opaque handle.
        check(
            unsafe { hipStreamCreateWithFlags(&raw mut s, HIP_STREAM_NON_BLOCKING) },
            "hipStreamCreateWithFlags",
        )?;
        Ok(Self { ptr: s, device_id })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub(crate) fn raw(&self) -> HipStreamT {
        self.ptr
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` was returned by `hipStreamCreate` in `new()`
            // and is not shared with any other owning type (streams are Send
            // but never Clone). The null check above guards against a partially-
            // constructed instance. Destroy errors are non-actionable at drop.
            let _ = unsafe { hipStreamDestroy(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

impl Stream for HipStream {
    fn raw_handle(&self) -> usize {
        self.ptr as usize
    }

    fn synchronize(&self) -> DeviceResult<()> {
        // SAFETY: `self.ptr` is a live stream handle (owned, non-null once
        // constructed; drop sets it back to null but no method can observe that).
        check(
            unsafe { hipStreamSynchronize(self.ptr) },
            "hipStreamSynchronize",
        )
    }
}

/// 5.b — HIP event for cross-stream DAG scheduling. Used by the async
/// peer-copy pipeline in `HipCluster::peer_copy_via_host_async`.
/// Created with `HIP_EVENT_DISABLE_TIMING` — we never call `hipEventElapsedTime`,
/// just `hipEventRecord` / `hipStreamWaitEvent`. Drop destroys the handle.
pub struct HipEvent {
    ptr: crate::sys::HipEventT,
    device_id: i32,
}

impl std::fmt::Debug for HipEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipEvent")
            .field("ptr", &(self.ptr as usize))
            .field("device_id", &self.device_id)
            .finish()
    }
}

// SAFETY: HipEventT is an opaque driver handle. Events are thread-safe
// per the HIP runtime contract; record/wait ops serialise at the driver.
unsafe impl Send for HipEvent {}
unsafe impl Sync for HipEvent {}

impl HipEvent {
    /// Create a new timing-disabled event on `device_id`. Caller must have
    /// `bind(device_id)` in effect.
    pub fn new(device_id: i32) -> DeviceResult<Self> {
        let mut e: crate::sys::HipEventT = ptr::null_mut();
        // SAFETY: `hipEventCreateWithFlags` writes an opaque handle through
        // the out-pointer. `&mut e` is valid for a `HipEventT`.
        check(
            unsafe {
                crate::sys::hipEventCreateWithFlags(&raw mut e, crate::sys::HIP_EVENT_DISABLE_TIMING)
            },
            "hipEventCreateWithFlags",
        )?;
        Ok(Self { ptr: e, device_id })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Record this event on `stream`. After previous work on `stream`
    /// completes, the event transitions to the recorded state.
    pub fn record(&self, stream: &HipStream) -> DeviceResult<()> {
        // SAFETY: both handles are live (owned by Self / caller).
        check(
            unsafe { crate::sys::hipEventRecord(self.ptr, stream.raw()) },
            "hipEventRecord",
        )
    }

    /// Insert a wait on `stream`: subsequent work enqueued on `stream`
    /// won't start until the event transitions to recorded. Non-blocking
    /// on the host — the wait is driver-side.
    pub fn stream_wait(&self, stream: &HipStream) -> DeviceResult<()> {
        // SAFETY: both handles are live. flags=0 for standard semantics.
        check(
            unsafe { crate::sys::hipStreamWaitEvent(stream.raw(), self.ptr, 0) },
            "hipStreamWaitEvent",
        )
    }

    /// timing-enabled event constructor (omits
    /// `HIP_EVENT_DISABLE_TIMING` so `hipEventElapsedTime` returns valid
    /// data). Use only for profiling instrumentation; the timing-
    /// disabled `new()` is cheaper for ordering-only events on the
    /// production hot path.
    pub fn new_timing(device_id: i32) -> DeviceResult<Self> {
        let mut e: crate::sys::HipEventT = ptr::null_mut();
        // SAFETY: hipEventCreate writes through the out-pointer; flags
        // default = timing enabled.
        check(
            unsafe { crate::sys::hipEventCreate(&raw mut e) },
            "hipEventCreate (timing)",
        )?;
        Ok(Self { ptr: e, device_id })
    }

    /// synchronous wait until this event is reached.
    /// Used by the profiler harness so the host wall-clock observes
    /// device-side completion before reading `elapsed_ms_since`.
    pub fn synchronize(&self) -> DeviceResult<()> {
        // SAFETY: handle owned by Self.
        check(
            unsafe { crate::sys::hipEventSynchronize(self.ptr) },
            "hipEventSynchronize",
        )
    }

    /// return milliseconds between `start.record(stream)`
    /// and `self.record(stream)`. Both events must be timing-enabled
    /// (created via [`new_timing`]). Caller is responsible for syncing
    /// the events first (or calling [`synchronize`]).
    pub fn elapsed_ms_since(&self, start: &HipEvent) -> DeviceResult<f32> {
        let mut ms: f32 = 0.0;
        // SAFETY: ms is a stack scalar; both event handles live.
        check(
            unsafe { crate::sys::hipEventElapsedTime(&raw mut ms, start.ptr, self.ptr) },
            "hipEventElapsedTime",
        )?;
        Ok(ms)
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: self.ptr was returned by hipEventCreateWithFlags and
            // is not aliased (events are owned, not Clone).
            let _ = unsafe { crate::sys::hipEventDestroy(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

/// 6.a — executable HIP graph, instantiated from a stream-capture
/// recording. Replay issues the whole captured sequence to a stream with
/// a single driver call, collapsing per-kernel launch overhead.
/// Construction flow: `HipGraphExec::capture(stream, |s| { ...enqueue work on s... })`.
/// The closure issues whatever kernels / memcpys make up the subgraph; on
/// return we end capture, instantiate, and hold the executable. Drop
/// destroys both the recording graph and the instantiated exec.
pub struct HipGraphExec {
    exec: crate::sys::HipGraphExecT,
    /// Source graph kept alive for the lifetime of the exec. HIP's node
    /// introspection (`hipGraphKernelNodeGetParams`) requires the source
    /// graph be live; destroying it before reads fails with
    /// `hipErrorInvalidValue`. Destroyed in Drop after the exec.
    graph: crate::sys::HipGraphT,
    device_id: i32,
    /// Kernel-type nodes in dispatch order, enumerated at capture time
    /// via `hipGraphGetNodes` + `hipGraphNodeGetType` filtering. Stored
    /// in a Vec<usize> because `HipGraphNodeT` is a `*mut c_void` and
    /// doesn't implement Send/Sync out of the box; we cast back when
    /// calling the param-update FFI.
    kernel_nodes: Vec<usize>,
    /// 6.a-i5b — memcpy-type graph nodes in dispatch order. Same
    /// `Vec<usize>` trick as `kernel_nodes`.
    memcpy_nodes: Vec<usize>,
    /// 6.a-i3 — slot → (kernel_node_idx, arg_idx, arity) bindings
    /// accumulated from tagged pushes during capture.
    slot_map: crate::graph_capture::SlotMap,
    /// 6.a-i4 — shadow of each kernel node's current `kernelParams`
    /// pointer array, kept in sync with the exec. Without this, every
    /// `set_slot` would read from `hipGraphKernelNodeGetParams` (which
    /// returns the *source graph* params — unchanged across exec
    /// updates), so consecutive `set_slot` calls to the same node would
    /// clobber each other. Indexed by kernel-node ordinal; cached
    /// `hipKernelNodeParams` metadata (func, dims, sharedMemBytes) is
    /// stored alongside so we don't re-fetch on every update.
    node_shadows: std::cell::RefCell<Vec<NodeShadow>>,
    /// 6.a-i5b — shadow of each memcpy node's current params
    /// (dst, src, count, kind). Same motivation as `node_shadows`:
    /// `hipGraphMemcpyNodeGetParams` returns the source-graph params,
    /// not the exec's. Indexed by memcpy-node ordinal.
    memcpy_shadows: std::cell::RefCell<Vec<MemcpyShadow>>,
}

/// Per-memcpy-node shadow used by `set_memcpy_slot`.
#[derive(Clone, Copy, Debug)]
struct MemcpyShadow {
    dst: usize,
    src: usize,
    count: usize,
    kind: crate::sys::hipMemcpyKind,
}

// SAFETY: MemcpyShadow holds raw pointer values (usize) with no
// aliasing — updates happen only via &mut RefCell<Vec<_>> guarded by
// HipGraphExec's Send/Sync impl.
unsafe impl Send for MemcpyShadow {}
unsafe impl Sync for MemcpyShadow {}

/// Per-kernel-node shadow used by `set_slot`. `ptrs` is our Rust-side
/// copy of the exec's current kernelParams — mutated in place on
/// `set_slot`, then handed to `hipGraphExecKernelNodeSetParams` via its
/// as_mut_ptr(). `meta` preserves the capture-time launch geometry
/// because those fields never change across slot updates.
#[derive(Clone)]
struct NodeShadow {
    ptrs: Vec<*mut std::os::raw::c_void>,
    meta: crate::sys::hipKernelNodeParams,
}

// SAFETY: NodeShadow wraps raw pointers that point to driver-owned
// staging or caller-owned scalar storage. Moves/access happen only via
// `&mut RefCell<Vec<NodeShadow>>` from HipGraphExec, which is already
// guarded for Send/Sync above. The raw pointers are used only in FFI
// calls that don't alias our Rust-owned state.
unsafe impl Send for NodeShadow {}
unsafe impl Sync for NodeShadow {}

impl std::fmt::Debug for HipGraphExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipGraphExec")
            .field("exec", &(self.exec as usize))
            .field("device_id", &self.device_id)
            .field("kernel_nodes", &self.kernel_nodes.len())
            .finish()
    }
}

// SAFETY: HipGraphExecT is an opaque driver handle. Graph-exec launch is
// documented thread-safe on HIP — the same exec can be replayed from
// multiple threads as long as the stream argument is not shared.
unsafe impl Send for HipGraphExec {}
unsafe impl Sync for HipGraphExec {}

impl HipGraphExec {
    /// Capture the closure's stream work into an executable graph.
    /// `stream` must be a non-null-stream (capture is illegal on the null
    /// stream). The closure should enqueue all kernels / memcpys that
    /// make up the subgraph on `stream`. Capture mode is `Relaxed` so
    /// the closure can run Rust-side work (pointer math, scratch-slot
    /// selection) interleaved with the enqueues.
    pub fn capture<F>(stream: &HipStream, f: F) -> DeviceResult<Self>
    where
        F: FnOnce(&HipStream) -> DeviceResult<()>,
    {
        // 6.a-i3 — enable the thread-local capture recorder for the
        // duration of this capture. Every `HipKernel::launch` issued
        // inside `f` will append a LaunchRecord that we later zip with
        // the graph's kernel nodes to build a SlotMap.
        let capture_scope = crate::graph_capture::CaptureScope::begin();

        // SAFETY: stream is owned+live. Relaxed capture mode lets host APIs
        // run during capture; the closure just enqueues on `stream`.
        check(
            unsafe {
                crate::sys::hipStreamBeginCapture(
                    stream.raw(),
                    crate::sys::HIP_STREAM_CAPTURE_MODE_RELAXED,
                )
            },
            "hipStreamBeginCapture",
        )?;

        let closure_result = f(stream);

        let mut graph: crate::sys::HipGraphT = ptr::null_mut();
        // SAFETY: end capture writes the recorded graph through &graph.
        // Must be called whether the closure failed or not — otherwise the
        // stream stays in capture mode and every subsequent submit errors.
        let end_code = unsafe { crate::sys::hipStreamEndCapture(stream.raw(), &raw mut graph) };

        // End the recorder scope regardless of the closure's outcome so
        // the thread-local state is always reset. Intentionally drain
        // here so any error paths below don't leave the state set.
        let capture_state = capture_scope.end();
        let launches = capture_state.launches.clone();

        closure_result?;
        check(end_code, "hipStreamEndCapture")?;

        // Enumerate nodes before instantiate so callers can address them
        // by ordinal. Filter into kernel vs memcpy buckets in dispatch
        // order. The graph stays alive for the full exec lifetime
        // because `hipGraph*NodeGetParams` requires a live source graph.
        let (kernel_nodes, memcpy_nodes) = unsafe { collect_nodes_by_type(graph) }?;

        let mut exec: crate::sys::HipGraphExecT = ptr::null_mut();
        // SAFETY: graph is the handle just returned by end-capture. The
        // err_node + log_buf out-params are optional; pass null / zero.
        let inst_code = unsafe {
            crate::sys::hipGraphInstantiate(
                &raw mut exec,
                graph,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            )
        };
        if inst_code != HIP_SUCCESS {
            // Instantiate failed — clean up the source graph before
            // returning the error; otherwise we leak it.
            let _ = unsafe { crate::sys::hipGraphDestroy(graph) };
            check(inst_code, "hipGraphInstantiate")?;
        }

        // Zip recorded kernel launches + memcpys with the graph's
        // corresponding node buckets. If no slot pushes happened on
        // either side, this produces empty maps (still valid).
        let memcpys = capture_state.memcpys;
        let slot_map = crate::graph_capture::SlotMap::from_recorder(
            &launches,
            kernel_nodes.len(),
            &memcpys,
            memcpy_nodes.len(),
        )
        .map_err(|msg| DeviceError::Backend {
            backend: BACKEND,
            code: -1,
            message: format!("HipGraphExec::capture slot-map: {msg}"),
        })?;

        // 6.a-i4 — initialise the per-node shadow so `set_slot`
        // reads state that survives across consecutive updates. We only
        // populate shadows for nodes that have at least one slot bound
        // (lazy init for others happens on first set_slot — see `set_slot`).
        let mut node_shadows: Vec<NodeShadow> = Vec::with_capacity(kernel_nodes.len());
        for (node_idx, &raw_node) in kernel_nodes.iter().enumerate() {
            let node = raw_node as crate::sys::HipGraphNodeT;
            let mut params = crate::sys::hipKernelNodeParams {
                block_dim: crate::sys::hipDim3::default(),
                extra: ptr::null_mut(),
                func: ptr::null_mut(),
                grid_dim: crate::sys::hipDim3::default(),
                kernel_params: ptr::null_mut(),
                shared_mem_bytes: 0,
            };
            // SAFETY: node is live (from hipGraphGetNodes on the still-
            // alive source graph); params is local stack storage.
            check(
                unsafe { crate::sys::hipGraphKernelNodeGetParams(node, &raw mut params) },
                "hipGraphKernelNodeGetParams (shadow init)",
            )?;
            // Determine arity. Without a dedicated query, pull it from
            // the slot_map if a slot is bound on this node; else leave 0
            // and the shadow's ptrs Vec stays empty — set_slot errors
            // with a clear message if the node has no bound slots.
            let arity = slot_map
                .bindings_for_node(node_idx)
                .next()
                .map(|b| b.arity)
                .unwrap_or(0);
            let ptrs = if arity > 0 && !params.kernel_params.is_null() {
                // SAFETY: params.kernel_params points to a driver-owned
                // array of `arity` void* entries.
                let slice: &[*mut std::os::raw::c_void] =
                    unsafe { std::slice::from_raw_parts(params.kernel_params, arity) };
                slice.to_vec()
            } else {
                Vec::new()
            };
            node_shadows.push(NodeShadow { ptrs, meta: params });
        }

        // 6.a-i5b — seed the memcpy shadows from the recorded
        // memcpy params. Index into shadows == memcpy-node ordinal.
        let mut memcpy_shadows: Vec<MemcpyShadow> = Vec::with_capacity(memcpy_nodes.len());
        for rec in memcpys.iter() {
            memcpy_shadows.push(MemcpyShadow {
                dst: rec.dst,
                src: rec.src,
                count: rec.count,
                kind: rec.kind,
            });
        }

        Ok(Self {
            exec,
            graph,
            device_id: stream.device_id(),
            kernel_nodes,
            memcpy_nodes,
            slot_map,
            node_shadows: std::cell::RefCell::new(node_shadows),
            memcpy_shadows: std::cell::RefCell::new(memcpy_shadows),
        })
    }

    /// multi-stream capture into ONE shared graph, no slot
    /// machinery. Creates an empty graph, begins capture on every
    /// stream pointing to that graph, runs `f` (which may issue
    /// kernels + cross-stream events on any subset of those streams),
    /// ends capture on each stream, and instantiates a single
    /// `HipGraphExec`. Cross-stream `hipEventRecord` /
    /// `hipStreamWaitEvent` are resolved as internal graph edges.
    /// Replay: `launch(stream)` issues the whole shared graph from one
    /// stream — the captured fan-out across the original capturing
    /// streams is preserved as graph topology and replayed on the HIP
    /// runtime's internal worker streams.
    /// Use case: per-stage capture in hybrid pp+tp where the layer
    /// loop issues work on every TP-rank's stream and synchronises
    /// cross-rank via events (e.g. AR all-reduce). The
    /// per-stream-separate-graph form (`hipStreamBeginCapture` × N)
    /// can NOT capture cross-stream events — at replay each graph
    /// launches in isolation and the wait fails to resolve.
    /// Limitation: returns a HipGraphExec with EMPTY slot map.
    /// `set_slot` / `set_memcpy_slot` will error. K/V append
    /// destinations + `n_tokens_kv` are frozen at capture time.
    pub fn capture_into_shared_graph<F>(streams: &[&HipStream], f: F) -> DeviceResult<Self>
    where
        F: FnOnce() -> DeviceResult<()>,
    {
        if streams.is_empty() {
            return Err(DeviceError::Backend {
                backend: BACKEND,
                code: -1,
                message: "capture_into_shared_graph: no streams".to_string(),
            });
        }

        // 1. Allocate one empty graph that all streams will capture
        // INTO. With `hipStreamBeginCaptureToGraph`, the runtime
        // appends to this graph as the closure runs; with plain
        // `hipStreamBeginCapture`, kernel launches in multi-stream
        // mode error out with `hipModuleLaunchKernel: invalid
        // argument` on the first launch (ROCm 7.1.1 issue —
        // multi-stream concurrent capture without a shared graph
        // target rejects launches).
        let mut shared_graph: crate::sys::HipGraphT = ptr::null_mut();
        check(
            unsafe { crate::sys::hipGraphCreate(&raw mut shared_graph, 0) },
            "hipGraphCreate",
        )?;

        let _capture_scope = crate::graph_capture::CaptureScope::begin();

        // 2. Begin capture-to-graph on every stream against the shared
        // graph. The runtime tracks per-stream capture sessions but
        // appends every captured node to `shared_graph`. Cross-stream
        // events recorded inside the closure resolve as graph edges
        // between nodes from different originating streams.
        for (i, stream) in streams.iter().enumerate() {
            let code = unsafe {
                crate::sys::hipStreamBeginCaptureToGraph(
                    stream.raw(),
                    shared_graph,
                    ptr::null(),
                    ptr::null(),
                    0,
                    crate::sys::HIP_STREAM_CAPTURE_MODE_RELAXED,
                )
            };
            if code != HIP_SUCCESS {
                for prior in &streams[..i] {
                    let mut g: crate::sys::HipGraphT = ptr::null_mut();
                    let _ = unsafe { crate::sys::hipStreamEndCapture(prior.raw(), &raw mut g) };
                }
                let _ = unsafe { crate::sys::hipGraphDestroy(shared_graph) };
                check(code, "hipStreamBeginCaptureToGraph")?;
            }
        }

        let trace = dev_flag("FLAMBEAU_GRAPH_TRACE");
        if trace {
            eprintln!("[capture_into_shared_graph] closure starting");
        }
        // 3. Run the closure — issues kernels + cross-stream events.
        let closure_result = f();
        if trace {
            eprintln!(
                "[capture_into_shared_graph] closure returned: {}",
                if closure_result.is_ok() { "ok" } else { "err" }
            );
        }

        // 4. End capture on every stream so they leave capture mode and
        // subsequent kernel launches succeed (a stuck-in-capture
        // stream rejects launches with "invalid argument" forever).
        // finding on ROCm 7.1.1 with
        // `hipStreamBeginCaptureToGraph` + cross-stream events
        // (Coder-Next pp2tp2):
        // - When the closure RAN TO COMPLETION (no errors), every
        // end-capture call returns 904
        // (`hipErrorStreamCaptureUnmatched`) — there is no
        // "primary" returning success. Calling end-capture on a
        // second stream after the first returned 904 SIGSEGVs.
        // - When the closure ABORTED early (kernel error in
        // capture), end-capture on every stream returns 1
        // (Invalid value) but does NOT SIGSEGV; this DOES
        // return the streams to non-capture state.
        // Strategy: best-effort end-capture all streams. To avoid
        // the post-success SIGSEGV pattern, we stop iterating as
        // soon as we observe a "session closed" condition (success
        // OR Unmatched). On the failure path (closure aborted),
        // every end-capture returns Invalid — keep iterating to
        // recover all streams.
        const HIP_ERROR_STREAM_CAPTURE_UNMATCHED: i32 = 904;
        let mut last_end_code: i32 = HIP_SUCCESS;
        let mut had_clean_close = false;
        for (i, stream) in streams.iter().enumerate() {
            if had_clean_close {
                if trace {
                    eprintln!("[capture_into_shared_graph] skip end stream {i} (session already closed cleanly)");
                }
                continue;
            }
            if trace {
                eprintln!("[capture_into_shared_graph] end capture stream {i}");
            }
            let mut out_graph: crate::sys::HipGraphT = ptr::null_mut();
            let code = unsafe { crate::sys::hipStreamEndCapture(stream.raw(), &raw mut out_graph) };
            if trace {
                eprintln!(
                    "[capture_into_shared_graph] end stream {i} -> code {code} out_graph 0x{:x}",
                    out_graph as usize
                );
            }
            // Drop any returned graph immediately — none of them are
            // usable on ROCm 7.1.1 multi-stream + cross-stream events.
            if !out_graph.is_null() {
                let _ = unsafe { crate::sys::hipGraphDestroy(out_graph) };
            }
            last_end_code = code;
            // Only treat the SUCCESS / Unmatched cases as "session
            // closed cleanly"; in those cases stop, because further
            // end-capture has been observed to SIGSEGV. Other errors
            // (Invalid value etc.) mean the session was already
            // invalidated by a prior closure error, and end-capture
            // on each stream is necessary to clear state.
            if code == HIP_SUCCESS || code == HIP_ERROR_STREAM_CAPTURE_UNMATCHED {
                had_clean_close = true;
            }
        }
        let _ = _capture_scope.end();
        let _ = unsafe { crate::sys::hipGraphDestroy(shared_graph) };

        closure_result?;
        Err(DeviceError::Backend {
            backend: BACKEND,
            code: last_end_code,
            message: format!(
                "capture_into_shared_graph: ROCm 7.1.1 multi-stream \
                 capture-to-graph with cross-stream events returns no \
                 usable graph (last end-capture code {last_end_code}). \
                 hipStreamBeginCaptureToGraph is documented beta with \
                 'outstanding issues' — confirmed broken on multi-device \
                 captures here. Disable FLAMBEAU_DECODE_GRAPH on \
                 TP/hybrid topologies; revisit on newer ROCm."
            ),
        })
    }

    /// Update a captured memcpy node's dst pointer. Used at replay time
    /// to retarget (typically) KV-cache append memcpys — src and count
    /// stay fixed between replays (the scratch layout is stable), only
    /// dst advances with the KV cache's tail position.
    /// # Safety
    /// `new_dst` must be a live device pointer valid for `binding.count`
    /// bytes of writes on the exec's device, for the full duration of
    /// the next replay.
    pub unsafe fn set_memcpy_slot(
        &self,
        slot: crate::graph_capture::MemcpySlot,
        new_dst: flambeau_core::DevicePtr,
    ) -> DeviceResult<()> {
        let binding =
            self.slot_map
                .get_memcpy(slot)
                .copied()
                .ok_or_else(|| DeviceError::Backend {
                    backend: BACKEND,
                    code: -1,
                    message: format!(
                        "HipGraphExec::set_memcpy_slot: slot id={} not bound",
                        slot.id()
                    ),
                })?;
        let mut shadows = self.memcpy_shadows.borrow_mut();
        let shadow = &mut shadows[binding.memcpy_node_idx];
        shadow.dst = new_dst.0;

        let node = self.memcpy_node_handle(binding.memcpy_node_idx)?;
        // SAFETY: exec + node are live. dst is a caller-provided live
        // device pointer per the outer unsafe contract. src / count /
        // kind come from the shadow (capture-time values, valid to
        // reuse).
        check(
            unsafe {
                crate::sys::hipGraphExecMemcpyNodeSetParams1D(
                    self.exec,
                    node,
                    shadow.dst as *mut std::os::raw::c_void,
                    shadow.src as *const std::os::raw::c_void,
                    shadow.count,
                    shadow.kind,
                )
            },
            "hipGraphExecMemcpyNodeSetParams1D",
        )
    }

    fn memcpy_node_handle(&self, idx: usize) -> DeviceResult<crate::sys::HipGraphNodeT> {
        self.memcpy_nodes
            .get(idx)
            .copied()
            .map(|u| u as crate::sys::HipGraphNodeT)
            .ok_or_else(|| DeviceError::Backend {
                backend: BACKEND,
                code: -1,
                message: format!(
                    "Memcpy node({idx}) out of range (have {} nodes)",
                    self.memcpy_nodes.len()
                ),
            })
    }

    /// Return the slot→(node, arg) map accumulated during capture. Empty
    /// if the capture closure didn't push any slots via `KernelArgs::push_slot`.
    pub fn slot_map(&self) -> &crate::graph_capture::SlotMap {
        &self.slot_map
    }

    /// Update the updateable scalar bound to `slot` with `new_value`, by
    /// building a fresh `kernelParams` pointer array (cloning the current
    /// array from `hipGraphKernelNodeGetParams`) and calling
    /// `hipGraphExecKernelNodeSetParams`.
    /// Every untouched arg keeps its current driver-side pointer — the
    /// driver's previous per-node staging buffer stays readable at least
    /// until the next SetParams on the same node, which is when the
    /// values for those pointers are re-snapshotted.
    /// # Safety
    /// - `new_value` must have the exact size + type the captured
    ///   kernel expects at this arg slot. Wrong size writes garbage.
    /// - `new_value` must remain live until this call returns (the HIP
    ///   runtime copies the value-by-pointer during SetParams).
    pub unsafe fn set_slot<T>(
        &self,
        slot: crate::graph_capture::ScalarSlot,
        new_value: &T,
    ) -> DeviceResult<()> {
        let binding = self
            .slot_map
            .get(slot)
            .copied()
            .ok_or_else(|| DeviceError::Backend {
                backend: BACKEND,
                code: -1,
                message: format!(
                    "HipGraphExec::set_slot: slot id={} not bound in this exec's slot_map",
                    slot.id()
                ),
            })?;
        let mut shadows = self.node_shadows.borrow_mut();
        let shadow = &mut shadows[binding.kernel_node_idx];
        // Mutate the Rust-side shadow and hand the full array to the
        // driver. Reading via hipGraphKernelNodeGetParams would fetch
        // the SOURCE graph's params, which never change across
        // hipGraphExecKernelNodeSetParams calls — consecutive set_slot
        // updates would clobber each other. The shadow preserves the
        // latest state across updates.
        shadow.ptrs[binding.arg_index] =
            std::ptr::from_ref::<T>(new_value) as *mut std::os::raw::c_void;

        let new_params = crate::sys::hipKernelNodeParams {
            block_dim: shadow.meta.block_dim,
            extra: std::ptr::null_mut(),
            func: shadow.meta.func,
            grid_dim: shadow.meta.grid_dim,
            kernel_params: shadow.ptrs.as_mut_ptr(),
            shared_mem_bytes: shadow.meta.shared_mem_bytes,
        };
        // SAFETY: shadow.ptrs has `binding.arity` entries (initialised
        // from the capture-time kernel_params); kernel_params[arg_index]
        // was just replaced with `new_value`'s stable reference (caller
        // contract). Other entries preserve the exec's current pointers.
        unsafe { self.set_kernel_node_params(binding.kernel_node_idx, &new_params) }
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Number of kernel-launch nodes captured (excludes memcpys, memsets,
    /// and driver-inserted sync nodes).
    pub fn num_kernel_nodes(&self) -> usize {
        self.kernel_nodes.len()
    }

    /// Read the current kernel-node params of kernel node `idx`. Used to
    /// seed an update: callers typically read, substitute one field
    /// (e.g. a new `kernelParams` pointer array), then pass the result
    /// back via [`Self::set_kernel_node_params`].
    pub fn get_kernel_node_params(
        &self,
        idx: usize,
    ) -> DeviceResult<crate::sys::hipKernelNodeParams> {
        let node = self.kernel_node_handle(idx)?;
        let mut params = crate::sys::hipKernelNodeParams {
            block_dim: crate::sys::hipDim3::default(),
            extra: ptr::null_mut(),
            func: ptr::null_mut(),
            grid_dim: crate::sys::hipDim3::default(),
            kernel_params: ptr::null_mut(),
            shared_mem_bytes: 0,
        };
        // SAFETY: node is a live kernel-node handle (captured from our
        // graph before destroy); &mut params is valid for writes of
        // hipKernelNodeParams.
        check(
            unsafe { crate::sys::hipGraphKernelNodeGetParams(node, &raw mut params) },
            "hipGraphKernelNodeGetParams",
        )?;
        Ok(params)
    }

    /// Update kernel-node `idx`'s launch parameters on the instantiated
    /// exec. The driver copies the param *values* at call-time (into its
    /// internal per-node staging buffer), so callers may drop the
    /// pointer-array backing storage after the call returns.
    /// # Safety
    /// `params.kernel_params` must point to an array of at least
    /// `kernel_arity` `*mut c_void` entries, each pointing to storage of
    /// the exact size and type the kernel's signature expects at that
    /// argument index. Wrong size / type silently writes garbage into
    /// the launch's arg frame.
    pub unsafe fn set_kernel_node_params(
        &self,
        idx: usize,
        params: &crate::sys::hipKernelNodeParams,
    ) -> DeviceResult<()> {
        let node = self.kernel_node_handle(idx)?;
        // SAFETY: exec + node are live; params is caller's responsibility
        // per the outer unsafe contract above.
        check(
            unsafe { crate::sys::hipGraphExecKernelNodeSetParams(self.exec, node, params) },
            "hipGraphExecKernelNodeSetParams",
        )
    }

    fn kernel_node_handle(&self, idx: usize) -> DeviceResult<crate::sys::HipGraphNodeT> {
        self.kernel_nodes
            .get(idx)
            .copied()
            .map(|u| u as crate::sys::HipGraphNodeT)
            .ok_or_else(|| DeviceError::Backend {
                backend: BACKEND,
                code: -1,
                message: format!(
                    "kernel_node({idx}) out of range (have {} kernel nodes)",
                    self.kernel_nodes.len()
                ),
            })
    }

    /// Replay the captured subgraph on `stream`. The replay is
    /// asynchronous — call `stream.synchronize()` to wait.
    pub fn launch(&self, stream: &HipStream) -> DeviceResult<()> {
        // SAFETY: self.exec is a live instantiated exec; stream is a
        // non-null live handle. Graph launches are independent of the
        // stream's recording state.
        check(
            unsafe { crate::sys::hipGraphLaunch(self.exec, stream.raw()) },
            "hipGraphLaunch",
        )
    }
}

/// Walk every node of `graph`, bucketed by type, preserving dispatch
/// order within each bucket. All node handles from `hipGraphGetNodes`
/// remain valid against the later-instantiated exec even after the
/// source graph is destroyed — that's the whole point of exposing them
/// to `hipGraphExec*NodeSetParams`.
/// Returns `(kernel_nodes, memcpy_nodes)`. Other node types (memsets,
/// host nodes, empty nodes, graph nodes) are currently discarded —
/// none of them are emitted by our current forward-path ops.
/// # Safety
/// `graph` must be a live, end-captured `HipGraphT`.
unsafe fn collect_nodes_by_type(
    graph: crate::sys::HipGraphT,
) -> DeviceResult<(Vec<usize>, Vec<usize>)> {
    let mut count: usize = 0;
    // SAFETY: hipGraphGetNodes with nodes=null writes count through
    // the num_nodes pointer and touches nothing else.
    check(
        unsafe { crate::sys::hipGraphGetNodes(graph, ptr::null_mut(), &raw mut count) },
        "hipGraphGetNodes(count)",
    )?;
    if count == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut nodes: Vec<crate::sys::HipGraphNodeT> = vec![ptr::null_mut(); count];
    let mut out_count = count;
    // SAFETY: nodes buffer has `count` slots; out_count starts at count.
    check(
        unsafe { crate::sys::hipGraphGetNodes(graph, nodes.as_mut_ptr(), &raw mut out_count) },
        "hipGraphGetNodes",
    )?;
    nodes.truncate(out_count);

    let mut kernel_nodes = Vec::new();
    let mut memcpy_nodes = Vec::new();
    for node in nodes {
        let mut ntype: i32 = -1;
        // SAFETY: node is live (just returned by hipGraphGetNodes).
        check(
            unsafe { crate::sys::hipGraphNodeGetType(node, &raw mut ntype) },
            "hipGraphNodeGetType",
        )?;
        match ntype {
            t if t == crate::sys::HIP_GRAPH_NODE_TYPE_KERNEL => {
                kernel_nodes.push(node as usize);
            }
            t if t == crate::sys::HIP_GRAPH_NODE_TYPE_MEMCPY => {
                memcpy_nodes.push(node as usize);
            }
            _ => {}
        }
    }
    Ok((kernel_nodes, memcpy_nodes))
}

impl Drop for HipGraphExec {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            // SAFETY: self.exec was returned by hipGraphInstantiate and is
            // not aliased (HipGraphExec is owned, not Clone).
            let _ = unsafe { crate::sys::hipGraphExecDestroy(self.exec) };
            self.exec = ptr::null_mut();
        }
        if !self.graph.is_null() {
            // SAFETY: self.graph was returned by hipStreamEndCapture and is
            // not aliased. Destroy after the exec so the exec's dependency
            // on node handles is released first.
            let _ = unsafe { crate::sys::hipGraphDestroy(self.graph) };
            self.graph = ptr::null_mut();
        }
    }
}

/// A HIP device. Holds a device id and a default stream.
/// Construction calls `hipSetDevice` once, but there is no guarantee that the
/// process's HIP context stays on this device across calls — callers driving
/// multiple GPUs from one thread must `HipDevice::bind()` before operations.
#[derive(Debug)]
pub struct HipDevice {
    id: i32,
    default_stream: HipStream,
}

impl HipDevice {
    /// Create a new device wrapper for HIP device `id`. Sets the current
    /// HIP context to `id` and creates a default stream on it.
    pub fn new(id: i32) -> DeviceResult<Self> {
        let count = device_count()?;
        if count <= 0 {
            return Err(DeviceError::NoDevices { backend: BACKEND });
        }
        if id < 0 || id >= count {
            return Err(DeviceError::InvalidDeviceId {
                requested: id,
                count,
            });
        }
        bind(id)?;
        let default_stream = HipStream::new(id)?;
        Ok(Self { id, default_stream })
    }

    /// Bind the current thread's HIP context to this device. Required
    /// before alloc/free/stream ops when multiple devices are in use.
    pub fn bind(&self) -> DeviceResult<()> {
        bind(self.id)
    }

    /// 6.a-i5b — graph-captureable variant of `memcpy_async` that
    /// tags the memcpy with a [`MemcpySlot`]
    /// (from `crate::graph_capture`). Under a capture scope, the memcpy
    /// is recorded with `slot`; post-capture the exec's `SlotMap` binds
    /// the slot to the resulting memcpy graph node, enabling
    /// `HipGraphExec::set_memcpy_slot` to retarget dst / src per
    /// replay.
    /// # Safety
    /// Same as [`flambeau_core::Device::memcpy_async`] — `dst` and `src`
    /// must be valid for `bytes` in their respective address spaces
    /// per `dir`, and neither aliased by another pending op on the
    /// same stream.
    pub unsafe fn memcpy_async_slot(
        &self,
        stream: &HipStream,
        dir: CopyDirection,
        dst: DevicePtr,
        src: DevicePtr,
        bytes: usize,
        slot: crate::graph_capture::MemcpySlot,
    ) -> DeviceResult<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.bind()?;
        let kind = match dir {
            CopyDirection::HostToDevice => sys::hipMemcpyKind::HostToDevice,
            CopyDirection::DeviceToHost => sys::hipMemcpyKind::DeviceToHost,
            CopyDirection::DeviceToDevice => sys::hipMemcpyKind::DeviceToDevice,
        };
        crate::graph_capture::record_memcpy(Some(slot), dst.0, src.0, bytes, kind);
        // SAFETY: same invariants as `Device::memcpy_async`; see the
        // trait impl below for detail.
        let code = unsafe {
            sys::hipMemcpyAsync(
                dst.0 as *mut _,
                src.0 as *const _,
                bytes,
                kind,
                stream.raw(),
            )
        };
        check(code, "hipMemcpyAsync")
    }
}

impl Device for HipDevice {
    type Stream = HipStream;

    fn backend(&self) -> &'static str {
        BACKEND
    }

    fn id(&self) -> i32 {
        self.id
    }

    fn default_stream(&self) -> &Self::Stream {
        &self.default_stream
    }

    fn new_stream(&self) -> DeviceResult<Self::Stream> {
        self.bind()?;
        HipStream::new(self.id)
    }

    fn alloc(&self, bytes: usize) -> DeviceResult<DevicePtr> {
        if bytes == 0 {
            return Ok(DevicePtr::NULL);
        }
        self.bind()?;
        let mut p: *mut std::os::raw::c_void = ptr::null_mut();
        // SAFETY: `hipMalloc` writes a pointer through the out-pointer and
        // reads nothing from it. `&mut p` is valid for writes of `sizeof(void*)`.
        // The returned device pointer is owned by `DevicePtr`.
        let code = unsafe { hipMalloc(&raw mut p, bytes) };
        if code != HIP_SUCCESS {
            return Err(DeviceError::Alloc {
                backend: BACKEND,
                device: self.id,
                bytes,
                reason: error_string(code),
            });
        }
        Ok(DevicePtr(p as usize))
    }

    unsafe fn dealloc(&self, ptr: DevicePtr, _bytes: usize) -> DeviceResult<()> {
        if ptr.is_null() {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: caller's contract on `Device::dealloc` is that `ptr` was
        // returned by a prior `alloc` on a compatible device and is not
        // aliased. The null guard above handles `DevicePtr::NULL`.
        check(unsafe { hipFree(ptr.0 as *mut _) }, "hipFree")
    }

    unsafe fn memcpy_async(
        &self,
        stream: &Self::Stream,
        dir: CopyDirection,
        dst: DevicePtr,
        src: DevicePtr,
        bytes: usize,
    ) -> DeviceResult<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.bind()?;
        let kind = match dir {
            CopyDirection::HostToDevice => hipMemcpyKind::HostToDevice,
            CopyDirection::DeviceToHost => hipMemcpyKind::DeviceToHost,
            CopyDirection::DeviceToDevice => hipMemcpyKind::DeviceToDevice,
        };
        // 6.a-i5b — record the memcpy for graph-slot binding when
        // inside a capture scope. `slot: None` here — the Device trait
        // surface doesn't carry per-call slot info; callers that want
        // a tagged memcpy go through `HipDevice::memcpy_async_slot`.
        crate::graph_capture::record_memcpy(None, dst.0, src.0, bytes, kind);
        // SAFETY: caller's contract on `Device::memcpy_async` is that `src`
        // and `dst` are each valid for reads/writes of `bytes` in their
        // respective address spaces per `dir`, and that neither is aliased
        // by another pending op on the same stream. `stream.raw()` is a
        // live handle owned by the caller's `HipStream`.
        let code = unsafe {
            hipMemcpyAsync(
                dst.0 as *mut _,
                src.0 as *const _,
                bytes,
                kind,
                stream.raw(),
            )
        };
        check(code, "hipMemcpyAsync")
    }

    fn synchronize(&self) -> DeviceResult<()> {
        self.bind()?;
        check(
            // SAFETY: `hipDeviceSynchronize` takes no arguments and operates
            // on the thread-current HIP context, which `self.bind()` just set.
            unsafe { sys::hipDeviceSynchronize() },
            "hipDeviceSynchronize",
        )
    }
}

impl HipDevice {
    /// Cross-device peer copy, enqueued on `stream` (which must belong
    /// to the source device). Peer access between `self.id()` (source)
    /// and `dst_device_id` must already be enabled by the cluster
    /// bring-up (`probe_and_enable_peer_access`); otherwise this fails
    /// with `hipErrorInvalidValue` on first invocation.
    ///
    /// Modeled after llama.cpp's `ggml_backend_cuda_cpy_tensor_async`
    /// — the producer enqueues the cross-device copy on its own
    /// stream, records an event, and the consumer's stream waits on
    /// that event before reading `dst`. Eliminates the host-RAM
    /// bounce that the legacy `peer_copy_via_host` path uses.
    ///
    /// # Safety
    /// `dst` must be a live device allocation on `dst_device_id`,
    /// valid for `bytes` writes; `src` must be a live allocation on
    /// `self.id()`, valid for `bytes` reads. No other in-flight op on
    /// `stream` may alias either pointer.
    pub unsafe fn memcpy_peer_async(
        &self,
        stream: &HipStream,
        dst: DevicePtr,
        dst_device_id: i32,
        src: DevicePtr,
        bytes: usize,
    ) -> DeviceResult<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: caller's contract above; `stream.raw()` is owned by
        // the source-device HipStream so its driver context matches
        // the just-bound source device.
        let code = unsafe {
            sys::hipMemcpyPeerAsync(
                dst.0 as *mut _,
                dst_device_id,
                src.0 as *const _,
                self.id,
                bytes,
                stream.raw(),
            )
        };
        check(code, "hipMemcpyPeerAsync")
    }

    /// Cross-device peer PULL: copy `src` (on `src_device_id`) into `dst`
    /// (on `self`) enqueued on `stream` (which must belong to `self`, the
    /// DESTINATION device). The DMA copy engine sources the peer bytes
    /// coherently — unlike an in-kernel BAR1 shader load, which is stale
    /// on gfx906 PCIe P2P (see doc/DETERMINISM_INVESTIGATION.md). Used by
    /// the deterministic DtoD AllReduce: each rank pulls peer partials
    /// into local scratch on its own stream, then sums locally.
    ///
    /// # Safety
    /// `dst` must be a live allocation on `self` valid for `bytes` writes;
    /// `src` a live allocation on `src_device_id` valid for `bytes` reads;
    /// peer access authorised at cluster bring-up; no in-flight op on
    /// `stream` may alias either pointer.
    pub unsafe fn memcpy_peer_in_async(
        &self,
        stream: &HipStream,
        dst: DevicePtr,
        src: DevicePtr,
        src_device_id: i32,
        bytes: usize,
    ) -> DeviceResult<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: caller's contract above; `stream.raw()` belongs to
        // `self` (the destination device), matching the just-bound device.
        let code = unsafe {
            sys::hipMemcpyPeerAsync(
                dst.0 as *mut _,
                self.id,
                src.0 as *const _,
                src_device_id,
                bytes,
                stream.raw(),
            )
        };
        check(code, "hipMemcpyPeerAsync(pull)")
    }
}

impl Drop for HipDevice {
    fn drop(&mut self) {
        // The thread's currently-bound HIP device might be a different
        // rank by the time this fires. `hipStreamDestroy` on the
        // default_stream needs THIS device's context to be active,
        // otherwise the runtime defers teardown and crashes at process
        // exit when its atexit handler iterates orphaned streams. Bind
        // first; the inner `HipStream::Drop` then sees a matching context.
        let _ = bind(self.id);
    }
}
