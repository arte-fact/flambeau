//! `CudaGraphExec` — an executable CUDA graph captured from a stream recording.
//! Replay (`launch`) issues the whole captured sequence with one driver call,
//! collapsing per-op launch overhead.
//!
//! `capture` is the plain no-slot path (capture → instantiate → replay).
//! `capture_with_slots` additionally records, per launch, the arity + any
//! `KernelArgs::push_slot` tags and, per `memcpy_async_slot`, the tagged
//! memcpy — then zips those records with the graph's kernel / memcpy nodes
//! (enumerated via `cuGraphGetNodes`) into a `SlotMap`. `set_slot` /
//! `set_memcpy_slot` then rebind a scalar kernel arg / a memcpy source pointer
//! on the instantiated exec between replays (progressive dispatch).

use std::cell::RefCell;
use std::ptr;

use flambeau_core::{CopyDirection, DeviceError, DevicePtr, DeviceResult};

use crate::device::{check, CudaStream};
use crate::graph_capture::{MemcpySlot, ScalarSlot, SlotMap};
use crate::sys::{
    cuCtxGetCurrent, cuGraphDestroy, cuGraphExecDestroy, cuGraphExecKernelNodeSetParams,
    cuGraphExecMemcpyNodeSetParams, cuGraphGetNodes, cuGraphInstantiateWithFlags,
    cuGraphKernelNodeGetParams, cuGraphLaunch, cuGraphNodeGetType, cuStreamBeginCapture,
    cuStreamEndCapture, CUcontext, CUdeviceptr, CUgraph, CUgraphExec, CUgraphNode,
    CUDA_KERNEL_NODE_PARAMS, CUDA_MEMCPY3D, CU_GRAPH_NODE_TYPE_KERNEL, CU_GRAPH_NODE_TYPE_MEMCPY,
    CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST, CU_STREAM_CAPTURE_MODE_RELAXED,
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
};

const BACKEND: &str = "cuda";

/// Per-kernel-node shadow used by `set_slot`. `ptrs` is our Rust-side copy of
/// the exec's current `kernelParams` array — seeded at capture time from
/// `cuGraphKernelNodeGetParams` (driver-owned, stable pointers), mutated in
/// place on `set_slot`, then handed to `cuGraphExecKernelNodeSetParams`. `meta`
/// preserves the capture-time launch geometry (func, dims, sharedMemBytes),
/// which never changes across slot updates.
#[derive(Clone)]
struct NodeShadow {
    ptrs: Vec<*mut std::os::raw::c_void>,
    meta: CUDA_KERNEL_NODE_PARAMS,
}

// SAFETY: NodeShadow wraps raw pointers that point to driver-owned staging or
// caller-owned scalar storage. Access happens only via `&mut RefCell<Vec<_>>`
// from CudaGraphExec, already guarded for Send/Sync. The pointers are used
// only in FFI calls that don't alias our Rust-owned state.
unsafe impl Send for NodeShadow {}
unsafe impl Sync for NodeShadow {}

/// Per-memcpy-node shadow used by `set_memcpy_slot`. Same motivation as
/// `NodeShadow`: we keep the last-known (dst, src, count, dir) so a src-only
/// update preserves the other fields.
#[derive(Clone, Copy, Debug)]
struct MemcpyShadow {
    dst: usize,
    src: usize,
    count: usize,
    dir: CopyDirection,
}

// SAFETY: MemcpyShadow holds raw pointer values (usize) with no aliasing —
// updates happen only via `&mut RefCell<Vec<_>>` guarded by CudaGraphExec's
// Send/Sync impl.
unsafe impl Send for MemcpyShadow {}
unsafe impl Sync for MemcpyShadow {}

/// Executable CUDA graph + its source graph, instantiated from a stream
/// capture. Drop destroys the exec then the graph.
pub struct CudaGraphExec {
    exec: CUgraphExec,
    /// Source graph kept alive for the exec's lifetime. `cuGraphKernelNodeGet-
    /// Params` reads from the source graph's node storage; destroying it before
    /// those reads would fault. Destroyed in Drop after the exec.
    graph: CUgraph,
    device_id: i32,
    /// The context the graph was captured in — required by
    /// `cuGraphExecMemcpyNodeSetParams`.
    ctx: CUcontext,
    /// Kernel-type nodes in dispatch order, enumerated at capture time via
    /// `cuGraphGetNodes` + `cuGraphNodeGetType` filtering. Stored as
    /// `Vec<usize>` because `CUgraphNode` is a `*mut c_void` and isn't
    /// Send/Sync out of the box; cast back when calling the update FFI.
    kernel_nodes: Vec<usize>,
    /// Memcpy-type graph nodes in dispatch order. Same `Vec<usize>` trick.
    memcpy_nodes: Vec<usize>,
    /// Slot → (kernel_node_idx, arg_idx, arity) / memcpy bindings accumulated
    /// from tagged pushes during capture. Empty for the no-slot `capture` path.
    slot_map: SlotMap,
    /// Shadow of each kernel node's current `kernelParams` array, kept in sync
    /// with the exec. Without this, every `set_slot` would re-read from
    /// `cuGraphKernelNodeGetParams` (which returns the *source graph* params —
    /// unchanged across exec updates), so consecutive `set_slot` calls to the
    /// same node would clobber each other.
    node_shadows: RefCell<Vec<NodeShadow>>,
    /// Shadow of each memcpy node's current params (dst, src, count, dir).
    /// Same motivation as `node_shadows`.
    memcpy_shadows: RefCell<Vec<MemcpyShadow>>,
}

// SAFETY: CUgraphExec/CUgraph/CUcontext are opaque driver handles; replay is
// documented thread-safe and the shadows are RefCell-guarded (single-threaded
// mutation at replay). The struct owns its handles and destroys them once.
unsafe impl Send for CudaGraphExec {}
unsafe impl Sync for CudaGraphExec {}

impl std::fmt::Debug for CudaGraphExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaGraphExec")
            .field("exec", &(self.exec as usize))
            .field("device_id", &self.device_id)
            .field("kernel_nodes", &self.kernel_nodes.len())
            .field("memcpy_nodes", &self.memcpy_nodes.len())
            .finish()
    }
}

impl CudaGraphExec {
    /// Capture the work that `f` enqueues on `stream` into an executable graph
    /// (no slot map). `stream` must be a created (non-NULL) stream; thread-local
    /// capture mode keeps the recording isolated from other threads' streams.
    /// `set_slot` / `set_memcpy_slot` on the result error (empty slot map) —
    /// use [`Self::capture_with_slots`] to make args rebindable per replay.
    pub fn capture<F>(stream: &CudaStream, f: F) -> DeviceResult<Self>
    where
        F: FnOnce(&CudaStream) -> DeviceResult<()>,
    {
        let ctx = current_ctx()?;
        // SAFETY: `stream.raw()` is a live non-NULL stream handle.
        check(
            unsafe { cuStreamBeginCapture(stream.raw(), CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) },
            "cuStreamBeginCapture",
        )?;
        let body = f(stream);
        // End capture regardless of `body` so the stream leaves capture state,
        // then surface the closure error before instantiating.
        let mut graph: CUgraph = ptr::null_mut();
        let end = check(
            unsafe { cuStreamEndCapture(stream.raw(), &raw mut graph) },
            "cuStreamEndCapture",
        );
        body?;
        end?;

        let mut exec: CUgraphExec = ptr::null_mut();
        // SAFETY: `graph` is the just-captured graph; flags 0 = default.
        let inst = check(
            unsafe { cuGraphInstantiateWithFlags(&raw mut exec, graph, 0) },
            "cuGraphInstantiateWithFlags",
        );
        if let Err(e) = inst {
            // SAFETY: graph was captured and is destroyed exactly once here.
            unsafe { cuGraphDestroy(graph) };
            return Err(e);
        }
        Ok(CudaGraphExec {
            exec,
            graph,
            device_id: stream.device_id(),
            ctx,
            kernel_nodes: Vec::new(),
            memcpy_nodes: Vec::new(),
            slot_map: SlotMap::default(),
            node_shadows: RefCell::new(Vec::new()),
            memcpy_shadows: RefCell::new(Vec::new()),
        })
    }

    /// Capture `f`'s stream work into an executable graph WITH a slot map, so
    /// scalar kernel args tagged via `KernelArgs::push_slot` and memcpy sources
    /// issued via `CudaDevice::memcpy_async_slot` can be rebound per replay.
    /// Capture mode is RELAXED so the closure may run host-side Rust (pointer
    /// math, scratch-slot selection) interleaved with the enqueues.
    pub fn capture_with_slots<F>(stream: &CudaStream, f: F) -> DeviceResult<Self>
    where
        F: FnOnce(&CudaStream) -> DeviceResult<()>,
    {
        // Enable the thread-local capture recorder for the duration of this
        // capture. Every `CudaKernel::launch` / `memcpy_async*` issued inside
        // `f` appends a record we later zip with the graph's nodes.
        let capture_scope = crate::graph_capture::CaptureScope::begin();
        let ctx = current_ctx()?;

        // SAFETY: stream is owned+live. RELAXED capture lets host APIs run
        // during capture; the closure just enqueues on `stream`.
        check(
            unsafe { cuStreamBeginCapture(stream.raw(), CU_STREAM_CAPTURE_MODE_RELAXED) },
            "cuStreamBeginCapture",
        )?;

        let closure_result = f(stream);

        let mut graph: CUgraph = ptr::null_mut();
        // SAFETY: end capture writes the recorded graph through &graph. Must be
        // called whether the closure failed or not, else the stream stays in
        // capture mode and every subsequent submit errors.
        let end_code = unsafe { cuStreamEndCapture(stream.raw(), &raw mut graph) };

        // Drain the recorder regardless of outcome so the thread-local is reset.
        let capture_state = capture_scope.end();
        let launches = capture_state.launches.clone();

        closure_result?;
        check(end_code, "cuStreamEndCapture")?;

        // Enumerate nodes before instantiate so callers address them by
        // ordinal. The graph stays alive for the exec lifetime because
        // `cuGraphKernelNodeGetParams` reads the source graph's node storage.
        // SAFETY: `graph` is the live, end-captured graph.
        let (kernel_nodes, memcpy_nodes) = unsafe { collect_nodes_by_type(graph) }?;

        let mut exec: CUgraphExec = ptr::null_mut();
        // SAFETY: `graph` is the just-captured graph; flags 0 = default.
        let inst_code = unsafe { cuGraphInstantiateWithFlags(&raw mut exec, graph, 0) };
        if inst_code != crate::sys::CUDA_SUCCESS {
            // SAFETY: graph destroyed exactly once on the failure path.
            unsafe { cuGraphDestroy(graph) };
            check(inst_code, "cuGraphInstantiateWithFlags")?;
        }

        let memcpys = capture_state.memcpys;
        let slot_map =
            SlotMap::from_recorder(&launches, kernel_nodes.len(), &memcpys, memcpy_nodes.len())
                .map_err(|msg| DeviceError::Backend {
                    backend: BACKEND,
                    code: -1,
                    message: format!("CudaGraphExec::capture_with_slots slot-map: {msg}"),
                })?;

        // Seed the per-kernel-node shadow so `set_slot` reads state that
        // survives across consecutive updates. Only nodes with a bound slot get
        // a populated `ptrs`; others keep an empty Vec (set_slot on them errors
        // via the slot_map lookup first, so this is never indexed).
        let mut node_shadows: Vec<NodeShadow> = Vec::with_capacity(kernel_nodes.len());
        for (node_idx, &raw_node) in kernel_nodes.iter().enumerate() {
            let node = raw_node as CUgraphNode;
            let mut params = empty_kernel_node_params();
            // SAFETY: node is live (from cuGraphGetNodes on the still-alive
            // source graph); &mut params is valid for writes of the v1 struct.
            check(
                unsafe { cuGraphKernelNodeGetParams(node, &raw mut params) },
                "cuGraphKernelNodeGetParams (shadow init)",
            )?;
            let arity = slot_map
                .bindings_for_node(node_idx)
                .next()
                .map(|b| b.arity)
                .unwrap_or(0);
            let ptrs = if arity > 0 && !params.kernelParams.is_null() {
                // SAFETY: params.kernelParams points to a driver-owned array of
                // `arity` void* entries.
                let slice: &[*mut std::os::raw::c_void] =
                    unsafe { std::slice::from_raw_parts(params.kernelParams, arity) };
                slice.to_vec()
            } else {
                Vec::new()
            };
            node_shadows.push(NodeShadow { ptrs, meta: params });
        }

        // Seed memcpy shadows from the recorded params. Index == memcpy-node
        // ordinal (from_recorder proved the counts line up).
        let mut memcpy_shadows: Vec<MemcpyShadow> = Vec::with_capacity(memcpy_nodes.len());
        for rec in memcpys.iter() {
            memcpy_shadows.push(MemcpyShadow {
                dst: rec.dst,
                src: rec.src,
                count: rec.count,
                dir: rec.dir,
            });
        }

        Ok(CudaGraphExec {
            exec,
            graph,
            device_id: stream.device_id(),
            ctx,
            kernel_nodes,
            memcpy_nodes,
            slot_map,
            node_shadows: RefCell::new(node_shadows),
            memcpy_shadows: RefCell::new(memcpy_shadows),
        })
    }

    /// Replay the captured graph on `stream`. Asynchronous — call
    /// `stream.synchronize()` to wait.
    pub fn launch(&self, stream: &CudaStream) -> DeviceResult<()> {
        // SAFETY: `self.exec` is a live instantiated graph exec.
        check(unsafe { cuGraphLaunch(self.exec, stream.raw()) }, "cuGraphLaunch")
    }

    /// The device this graph was captured on.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// The slot→(node, arg) map accumulated during capture. Empty for the
    /// no-slot `capture` path.
    pub fn slot_map(&self) -> &SlotMap {
        &self.slot_map
    }

    /// Number of kernel-launch nodes captured (excludes memcpys, memsets, and
    /// driver-inserted sync nodes).
    pub fn num_kernel_nodes(&self) -> usize {
        self.kernel_nodes.len()
    }

    /// Update the scalar bound to `slot` with `new_value`, by mutating the
    /// node's Rust-side `kernelParams` shadow (one arg pointer swapped, the
    /// rest preserved) and calling `cuGraphExecKernelNodeSetParams`.
    /// # Safety
    /// - `new_value` must have the exact size + type the captured kernel
    ///   expects at this arg slot. Wrong size writes garbage.
    /// - `new_value` must remain live until this call returns (the driver
    ///   copies the value-by-pointer during SetParams).
    pub unsafe fn set_slot<T>(&self, slot: ScalarSlot, new_value: &T) -> DeviceResult<()> {
        let binding = self.slot_map.get(slot).copied().ok_or_else(|| DeviceError::Backend {
            backend: BACKEND,
            code: -1,
            message: format!(
                "CudaGraphExec::set_slot: slot id={} not bound in this exec's slot_map",
                slot.id()
            ),
        })?;
        let node = self.kernel_node_handle(binding.kernel_node_idx)?;
        let mut shadows = self.node_shadows.borrow_mut();
        let shadow = &mut shadows[binding.kernel_node_idx];
        // Mutate the Rust-side shadow and hand the full array to the driver.
        // Re-reading via cuGraphKernelNodeGetParams would fetch the SOURCE
        // graph's params (never updated by SetParams), so consecutive set_slot
        // calls would clobber each other. The shadow preserves the latest
        // state across updates.
        shadow.ptrs[binding.arg_index] =
            std::ptr::from_ref::<T>(new_value) as *mut std::os::raw::c_void;

        let new_params = CUDA_KERNEL_NODE_PARAMS {
            func: shadow.meta.func,
            gridDimX: shadow.meta.gridDimX,
            gridDimY: shadow.meta.gridDimY,
            gridDimZ: shadow.meta.gridDimZ,
            blockDimX: shadow.meta.blockDimX,
            blockDimY: shadow.meta.blockDimY,
            blockDimZ: shadow.meta.blockDimZ,
            sharedMemBytes: shadow.meta.sharedMemBytes,
            kernelParams: shadow.ptrs.as_mut_ptr(),
            extra: ptr::null_mut(),
        };
        // SAFETY: exec + node are live; shadow.ptrs has `binding.arity` entries
        // (seeded from the capture-time kernelParams); the swapped entry points
        // to `new_value` (caller contract), the rest to driver-owned storage.
        check(
            unsafe { cuGraphExecKernelNodeSetParams(self.exec, node, &new_params) },
            "cuGraphExecKernelNodeSetParams",
        )
    }

    /// Retarget the SOURCE pointer of the memcpy node bound to `slot`. Used at
    /// replay time to re-source (typically) a per-token DtoD copy — dst, count
    /// and direction stay fixed between replays, only src advances.
    /// # Safety
    /// `new_src` must be a live pointer valid for `count` bytes of reads on the
    /// exec's device, in the copy's source address space, for the full duration
    /// of the next replay.
    pub unsafe fn set_memcpy_slot(&self, slot: MemcpySlot, new_src: DevicePtr) -> DeviceResult<()> {
        let binding =
            self.slot_map.get_memcpy(slot).copied().ok_or_else(|| DeviceError::Backend {
                backend: BACKEND,
                code: -1,
                message: format!("CudaGraphExec::set_memcpy_slot: slot id={} not bound", slot.id()),
            })?;
        let node = self.memcpy_node_handle(binding.memcpy_node_idx)?;
        let mut shadows = self.memcpy_shadows.borrow_mut();
        let shadow = &mut shadows[binding.memcpy_node_idx];
        shadow.src = new_src.0;
        let copy = memcpy3d_1d(shadow.dst, shadow.src, shadow.count, shadow.dir);
        // SAFETY: exec + node live; src is a caller-provided live pointer per
        // the outer unsafe contract; dst / count / dir come from the shadow
        // (capture-time values, valid to reuse). `self.ctx` is the capture-time
        // context.
        check(
            unsafe { cuGraphExecMemcpyNodeSetParams(self.exec, node, &copy, self.ctx) },
            "cuGraphExecMemcpyNodeSetParams",
        )
    }

    fn kernel_node_handle(&self, idx: usize) -> DeviceResult<CUgraphNode> {
        self.kernel_nodes
            .get(idx)
            .copied()
            .map(|u| u as CUgraphNode)
            .ok_or_else(|| DeviceError::Backend {
                backend: BACKEND,
                code: -1,
                message: format!(
                    "kernel_node({idx}) out of range (have {} kernel nodes)",
                    self.kernel_nodes.len()
                ),
            })
    }

    fn memcpy_node_handle(&self, idx: usize) -> DeviceResult<CUgraphNode> {
        self.memcpy_nodes
            .get(idx)
            .copied()
            .map(|u| u as CUgraphNode)
            .ok_or_else(|| DeviceError::Backend {
                backend: BACKEND,
                code: -1,
                message: format!(
                    "memcpy_node({idx}) out of range (have {} memcpy nodes)",
                    self.memcpy_nodes.len()
                ),
            })
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        // SAFETY: exec then graph, each destroyed exactly once.
        unsafe {
            if !self.exec.is_null() {
                cuGraphExecDestroy(self.exec);
            }
            if !self.graph.is_null() {
                cuGraphDestroy(self.graph);
            }
        }
    }
}

/// Read the calling thread's current context (for memcpy-node updates).
fn current_ctx() -> DeviceResult<CUcontext> {
    let mut ctx: CUcontext = ptr::null_mut();
    // SAFETY: writes a context handle through the out-pointer, reads nothing.
    check(unsafe { cuCtxGetCurrent(&raw mut ctx) }, "cuCtxGetCurrent")?;
    Ok(ctx)
}

fn empty_kernel_node_params() -> CUDA_KERNEL_NODE_PARAMS {
    CUDA_KERNEL_NODE_PARAMS {
        func: ptr::null_mut(),
        gridDimX: 0,
        gridDimY: 0,
        gridDimZ: 0,
        blockDimX: 0,
        blockDimY: 0,
        blockDimZ: 0,
        sharedMemBytes: 0,
        kernelParams: ptr::null_mut(),
        extra: ptr::null_mut(),
    }
}

/// Build a 1D `CUDA_MEMCPY3D` (Height = Depth = 1) for a `count`-byte copy from
/// `src` to `dst`, selecting the src/dst memory-type + pointer pair implied by
/// `dir`. All offsets/pitches zeroed.
fn memcpy3d_1d(dst: usize, src: usize, count: usize, dir: CopyDirection) -> CUDA_MEMCPY3D {
    let (src_kind, src_host, src_dev) = match dir {
        CopyDirection::HostToDevice => {
            (CU_MEMORYTYPE_HOST, src as *const std::os::raw::c_void, 0)
        }
        CopyDirection::DeviceToHost | CopyDirection::DeviceToDevice => {
            (CU_MEMORYTYPE_DEVICE, ptr::null(), src as CUdeviceptr)
        }
    };
    let (dst_kind, dst_host, dst_dev) = match dir {
        CopyDirection::DeviceToHost => (CU_MEMORYTYPE_HOST, dst as *mut std::os::raw::c_void, 0),
        CopyDirection::HostToDevice | CopyDirection::DeviceToDevice => {
            (CU_MEMORYTYPE_DEVICE, ptr::null_mut(), dst as CUdeviceptr)
        }
    };
    CUDA_MEMCPY3D {
        srcXInBytes: 0,
        srcY: 0,
        srcZ: 0,
        srcLOD: 0,
        srcMemoryType: src_kind,
        srcHost: src_host,
        srcDevice: src_dev,
        srcArray: ptr::null_mut(),
        reserved0: ptr::null_mut(),
        srcPitch: 0,
        srcHeight: 0,
        dstXInBytes: 0,
        dstY: 0,
        dstZ: 0,
        dstLOD: 0,
        dstMemoryType: dst_kind,
        dstHost: dst_host,
        dstDevice: dst_dev,
        dstArray: ptr::null_mut(),
        reserved1: ptr::null_mut(),
        dstPitch: 0,
        dstHeight: 0,
        WidthInBytes: count,
        Height: 1,
        Depth: 1,
    }
}

/// Walk every node of `graph`, bucketed by type, preserving dispatch order
/// within each bucket. Node handles from `cuGraphGetNodes` remain valid against
/// the later-instantiated exec — that's the point of exposing them to
/// `cuGraphExec*NodeSetParams`. Returns `(kernel_nodes, memcpy_nodes)`; other
/// node types are discarded (our forward-path ops emit only these two).
/// # Safety
/// `graph` must be a live, end-captured `CUgraph`.
unsafe fn collect_nodes_by_type(graph: CUgraph) -> DeviceResult<(Vec<usize>, Vec<usize>)> {
    let mut count: usize = 0;
    // SAFETY: with nodes=null, cuGraphGetNodes writes the count through
    // num_nodes and touches nothing else.
    check(
        unsafe { cuGraphGetNodes(graph, ptr::null_mut(), &raw mut count) },
        "cuGraphGetNodes(count)",
    )?;
    if count == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut nodes: Vec<CUgraphNode> = vec![ptr::null_mut(); count];
    let mut out_count = count;
    // SAFETY: nodes buffer has `count` slots; out_count starts at count.
    check(
        unsafe { cuGraphGetNodes(graph, nodes.as_mut_ptr(), &raw mut out_count) },
        "cuGraphGetNodes",
    )?;
    nodes.truncate(out_count);

    let mut kernel_nodes = Vec::new();
    let mut memcpy_nodes = Vec::new();
    for node in nodes {
        let mut ntype: std::os::raw::c_int = -1;
        // SAFETY: node is live (just returned by cuGraphGetNodes).
        check(
            unsafe { cuGraphNodeGetType(node, &raw mut ntype) },
            "cuGraphNodeGetType",
        )?;
        match ntype {
            t if t == CU_GRAPH_NODE_TYPE_KERNEL => kernel_nodes.push(node as usize),
            t if t == CU_GRAPH_NODE_TYPE_MEMCPY => memcpy_nodes.push(node as usize),
            _ => {}
        }
    }
    Ok((kernel_nodes, memcpy_nodes))
}
