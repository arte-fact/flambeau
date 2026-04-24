//! `HipDevice` and `HipStream` — safe wrappers around `libamdhip64`.
//!
//! One `HipDevice` ≈ one GPU. Internally it owns:
//! - a default stream (every implicit launch goes here unless the caller
//!   creates another stream);
//! - a device id used by `hipSetDevice` before any operation.
//!
//! The device is pinned to its id — `HipDevice::new(0)` sets the context to
//! device 0 and expects every subsequent operation on that instance to happen
//! on that device. Callers holding devices for multiple GPUs must call
//! `bind()` (or its equivalent `hipSetDevice`) before issuing work on a
//! device that isn't the current HIP context.

use std::os::raw::c_int;
use std::ptr;

use flambeau_core::{CopyDirection, Device, DeviceError, DevicePtr, DeviceResult, Stream};

use crate::sys::{
    self, error_string, hipFree, hipGetDevice, hipGetDeviceCount, hipMalloc, hipMemcpyAsync,
    hipMemcpyKind, hipSetDevice, hipStreamCreate, hipStreamCreateWithFlags, hipStreamDestroy,
    hipStreamSynchronize,
    hipStream_t, HIP_SUCCESS,
};

const BACKEND: &str = "hip";

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

/// A HIP stream. Drop destroys the underlying `hipStream_t`.
pub struct HipStream {
    ptr: hipStream_t,
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

// SAFETY: `hipStream_t` is an opaque driver handle with no Rust-side aliasing.
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
    ///
    /// NB: `hipStreamCreate` creates a **blocking** stream (serialises with
    /// the null stream). For truly-concurrent streams on the same device
    /// use [`Self::new_non_blocking`].
    pub fn new(device_id: i32) -> DeviceResult<Self> {
        let mut s: hipStream_t = ptr::null_mut();
        // SAFETY: `hipStreamCreate` writes a stream handle through the
        // out-pointer. `&mut s` is valid for writes of a `hipStream_t`.
        check(unsafe { hipStreamCreate(&raw mut s) }, "hipStreamCreate")?;
        Ok(Self {
            ptr: s,
            device_id,
        })
    }

    /// V2.25.g — create a non-blocking stream (`hipStreamNonBlocking` = 1).
    /// These streams do NOT serialise with the null stream and can run
    /// concurrently with each other on the same device, subject to
    /// occupancy. Used by `HipCluster::reserve_aux_streams` so the
    /// V2.25.d async ubatch pipeline truly overlaps lanes on the same
    /// device.
    pub fn new_non_blocking(device_id: i32) -> DeviceResult<Self> {
        let mut s: hipStream_t = ptr::null_mut();
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

    pub(crate) fn raw(&self) -> hipStream_t {
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

/// V2.25.b — HIP event for cross-stream DAG scheduling. Used by the async
/// peer-copy pipeline in `HipCluster::peer_copy_via_host_async`.
///
/// Created with `hipEventDisableTiming` — we never call `hipEventElapsedTime`,
/// just `hipEventRecord` / `hipStreamWaitEvent`. Drop destroys the handle.
pub struct HipEvent {
    ptr: crate::sys::hipEvent_t,
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

// SAFETY: hipEvent_t is an opaque driver handle. Events are thread-safe
// per the HIP runtime contract; record/wait ops serialise at the driver.
unsafe impl Send for HipEvent {}
unsafe impl Sync for HipEvent {}

impl HipEvent {
    /// Create a new timing-disabled event on `device_id`. Caller must have
    /// `bind(device_id)` in effect.
    pub fn new(device_id: i32) -> DeviceResult<Self> {
        let mut e: crate::sys::hipEvent_t = ptr::null_mut();
        // SAFETY: `hipEventCreateWithFlags` writes an opaque handle through
        // the out-pointer. `&mut e` is valid for a `hipEvent_t`.
        check(
            unsafe {
                crate::sys::hipEventCreateWithFlags(
                    &raw mut e,
                    crate::sys::hipEventDisableTiming,
                )
            },
            "hipEventCreateWithFlags",
        )?;
        Ok(Self { ptr: e, device_id })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub(crate) fn raw(&self) -> crate::sys::hipEvent_t {
        self.ptr
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

/// V2.26.a — executable HIP graph, instantiated from a stream-capture
/// recording. Replay issues the whole captured sequence to a stream with
/// a single driver call, collapsing per-kernel launch overhead.
///
/// Construction flow: `HipGraphExec::capture(stream, |s| { ...enqueue work on s... })`.
/// The closure issues whatever kernels / memcpys make up the subgraph; on
/// return we end capture, instantiate, and hold the executable. Drop
/// destroys both the recording graph and the instantiated exec.
pub struct HipGraphExec {
    exec: crate::sys::hipGraphExec_t,
    /// Source graph kept alive for the lifetime of the exec. HIP's node
    /// introspection (`hipGraphKernelNodeGetParams`) requires the source
    /// graph be live; destroying it before reads fails with
    /// `hipErrorInvalidValue`. Destroyed in Drop after the exec.
    graph: crate::sys::hipGraph_t,
    device_id: i32,
    /// Kernel-type nodes in dispatch order, enumerated at capture time
    /// via `hipGraphGetNodes` + `hipGraphNodeGetType` filtering. Stored
    /// in a Vec<usize> because `hipGraphNode_t` is a `*mut c_void` and
    /// doesn't implement Send/Sync out of the box; we cast back when
    /// calling the param-update FFI.
    kernel_nodes: Vec<usize>,
}

impl std::fmt::Debug for HipGraphExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipGraphExec")
            .field("exec", &(self.exec as usize))
            .field("device_id", &self.device_id)
            .field("kernel_nodes", &self.kernel_nodes.len())
            .finish()
    }
}

// SAFETY: hipGraphExec_t is an opaque driver handle. Graph-exec launch is
// documented thread-safe on HIP — the same exec can be replayed from
// multiple threads as long as the stream argument is not shared.
unsafe impl Send for HipGraphExec {}
unsafe impl Sync for HipGraphExec {}

impl HipGraphExec {
    /// Capture the closure's stream work into an executable graph.
    ///
    /// `stream` must be a non-null-stream (capture is illegal on the null
    /// stream). The closure should enqueue all kernels / memcpys that
    /// make up the subgraph on `stream`. Capture mode is `Relaxed` so
    /// the closure can run Rust-side work (pointer math, scratch-slot
    /// selection) interleaved with the enqueues.
    pub fn capture<F>(stream: &HipStream, f: F) -> DeviceResult<Self>
    where
        F: FnOnce(&HipStream) -> DeviceResult<()>,
    {
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

        let mut graph: crate::sys::hipGraph_t = ptr::null_mut();
        // SAFETY: end capture writes the recorded graph through &graph.
        // Must be called whether the closure failed or not — otherwise the
        // stream stays in capture mode and every subsequent submit errors.
        let end_code = unsafe { crate::sys::hipStreamEndCapture(stream.raw(), &raw mut graph) };

        closure_result?;
        check(end_code, "hipStreamEndCapture")?;

        // Enumerate nodes before instantiate so callers can address them
        // by ordinal. Filter to kernel nodes in dispatch order. The graph
        // stays alive for the full exec lifetime because
        // `hipGraphKernelNodeGetParams` requires a live source graph.
        let kernel_nodes = unsafe { collect_kernel_nodes(graph) }?;

        let mut exec: crate::sys::hipGraphExec_t = ptr::null_mut();
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

        Ok(Self {
            exec,
            graph,
            device_id: stream.device_id(),
            kernel_nodes,
        })
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
    ///
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

    fn kernel_node_handle(&self, idx: usize) -> DeviceResult<crate::sys::hipGraphNode_t> {
        self.kernel_nodes
            .get(idx)
            .copied()
            .map(|u| u as crate::sys::hipGraphNode_t)
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

/// Walk every node of `graph`, filter to kernel nodes, return their
/// handles in dispatch order. All node handles from `hipGraphGetNodes`
/// remain valid against the later-instantiated exec even after the
/// source graph is destroyed — that's the whole point of exposing them
/// to `hipGraphExecKernelNodeSetParams`.
///
/// # Safety
/// `graph` must be a live, end-captured `hipGraph_t`.
unsafe fn collect_kernel_nodes(graph: crate::sys::hipGraph_t) -> DeviceResult<Vec<usize>> {
    // Query the node count first (nodes=null + num_nodes=&count).
    let mut count: usize = 0;
    // SAFETY: hipGraphGetNodes with nodes=null writes count through
    // the num_nodes pointer and touches nothing else.
    check(
        unsafe { crate::sys::hipGraphGetNodes(graph, ptr::null_mut(), &raw mut count) },
        "hipGraphGetNodes(count)",
    )?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut nodes: Vec<crate::sys::hipGraphNode_t> = vec![ptr::null_mut(); count];
    let mut out_count = count;
    // SAFETY: nodes buffer has `count` slots; out_count starts at count.
    check(
        unsafe {
            crate::sys::hipGraphGetNodes(graph, nodes.as_mut_ptr(), &raw mut out_count)
        },
        "hipGraphGetNodes",
    )?;
    nodes.truncate(out_count);

    let mut kernel_nodes = Vec::with_capacity(out_count);
    for node in nodes {
        let mut ntype: i32 = -1;
        // SAFETY: node is live (just returned by hipGraphGetNodes), ntype
        // is local valid storage.
        check(
            unsafe { crate::sys::hipGraphNodeGetType(node, &raw mut ntype) },
            "hipGraphNodeGetType",
        )?;
        if ntype == crate::sys::HIP_GRAPH_NODE_TYPE_KERNEL {
            kernel_nodes.push(node as usize);
        }
    }
    Ok(kernel_nodes)
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
///
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
