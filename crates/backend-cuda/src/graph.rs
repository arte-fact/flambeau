//! `CudaGraphExec` — an executable CUDA graph captured from a stream recording.
//! Replay (`launch`) issues the whole captured sequence with one driver call,
//! collapsing per-op launch overhead. This is the core capture / instantiate /
//! replay surface; the shadow-param slot map (per-replay kernel / memcpy
//! parameter rebind for progressive dispatch) is not implemented here.

use flambeau_core::DeviceResult;

use crate::device::{check, CudaStream};
use crate::sys::{
    cuGraphDestroy, cuGraphExecDestroy, cuGraphInstantiateWithFlags, cuGraphLaunch, cuStreamBeginCapture,
    cuStreamEndCapture, CUgraph, CUgraphExec, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
};

/// Executable CUDA graph + its source graph, instantiated from a stream
/// capture. Drop destroys the exec then the graph.
pub struct CudaGraphExec {
    exec: CUgraphExec,
    graph: CUgraph,
    device_id: i32,
}

// SAFETY: CUgraphExec/CUgraph are opaque driver handles; replay is documented
// thread-safe and the struct is otherwise immutable after construction.
unsafe impl Send for CudaGraphExec {}
unsafe impl Sync for CudaGraphExec {}

impl std::fmt::Debug for CudaGraphExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaGraphExec")
            .field("exec", &(self.exec as usize))
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl CudaGraphExec {
    /// Capture the work that `f` enqueues on `stream` into an executable graph.
    /// `stream` must be a created (non-NULL) stream; thread-local capture mode
    /// keeps the recording isolated from other threads' streams.
    pub fn capture<F>(stream: &CudaStream, f: F) -> DeviceResult<Self>
    where
        F: FnOnce(&CudaStream) -> DeviceResult<()>,
    {
        // SAFETY: `stream.raw()` is a live non-NULL stream handle.
        check(
            unsafe { cuStreamBeginCapture(stream.raw(), CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) },
            "cuStreamBeginCapture",
        )?;
        let body = f(stream);
        // End capture regardless of `body` so the stream leaves capture state,
        // then surface the closure error before instantiating.
        let mut graph: CUgraph = std::ptr::null_mut();
        let end = check(unsafe { cuStreamEndCapture(stream.raw(), &raw mut graph) }, "cuStreamEndCapture");
        body?;
        end?;

        let mut exec: CUgraphExec = std::ptr::null_mut();
        // SAFETY: `graph` is the just-captured graph; flags 0 = default instantiate.
        let inst = check(unsafe { cuGraphInstantiateWithFlags(&raw mut exec, graph, 0) }, "cuGraphInstantiateWithFlags");
        if let Err(e) = inst {
            // SAFETY: graph was captured and is destroyed exactly once here.
            unsafe { cuGraphDestroy(graph) };
            return Err(e);
        }
        Ok(CudaGraphExec { exec, graph, device_id: stream.device_id() })
    }

    /// Replay the captured graph on `stream`.
    pub fn launch(&self, stream: &CudaStream) -> DeviceResult<()> {
        // SAFETY: `self.exec` is a live instantiated graph exec.
        check(unsafe { cuGraphLaunch(self.exec, stream.raw()) }, "cuGraphLaunch")
    }

    /// The device this graph was captured on.
    pub fn device_id(&self) -> i32 {
        self.device_id
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
