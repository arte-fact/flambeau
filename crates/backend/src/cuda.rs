//! CUDA binding for [`Backend`](crate::Backend).

use flambeau_backend_cuda::{CudaGraphExec, CudaStream};
use flambeau_core::{DevicePtr, MemcpySlot, ScalarSlot};

use crate::{Backend, DecodeGraphExec};

/// CUDA backend tag. Never constructed; only names the concrete CUDA type set
/// for a generic executor's `B: Backend` parameter.
pub enum CudaBackend {}

impl Backend for CudaBackend {
    type Device = flambeau_backend_cuda::CudaDevice;
    type Stream = flambeau_backend_cuda::CudaStream;
    type Event = flambeau_backend_cuda::CudaEvent;
    type Cluster = flambeau_backend_cuda::CudaCluster;
    type Registry = flambeau_ops::CudaOpsRegistry;
    type Ops<'a> = flambeau_ops::CudaOps<'a>;
    type DecodeGraph = CudaGraphExec;

    fn supports_decode_graph() -> bool {
        true
    }

    fn ops<'a>(reg: &'a Self::Registry, stream: &'a Self::Stream) -> Self::Ops<'a> {
        flambeau_ops::CudaOps::new(reg, stream)
    }

    fn new_device(id: i32) -> anyhow::Result<Self::Device> {
        Ok(flambeau_backend_cuda::CudaDevice::new(id)?)
    }

    fn new_registry(device: &Self::Device) -> anyhow::Result<Self::Registry> {
        Ok(flambeau_ops::CudaOpsRegistry::new(device)?)
    }
}

impl DecodeGraphExec<CudaBackend> for CudaGraphExec {
    fn capture(
        stream: &CudaStream,
        body: impl FnOnce(&CudaStream) -> anyhow::Result<()>,
    ) -> anyhow::Result<Self> {
        // `capture_with_slots` wants a `DeviceResult`-returning closure; stash
        // any `anyhow` error from `body` out-of-band so we can surface it after
        // capture ends (the stream must always leave capture state).
        let mut body_err: Option<anyhow::Error> = None;
        let res = CudaGraphExec::capture_with_slots(stream, |s| match body(s) {
            Ok(()) => Ok(()),
            Err(e) => {
                body_err = Some(e);
                Err(flambeau_core::DeviceError::Backend {
                    backend: "cuda",
                    code: -1,
                    message: "decode-graph capture body failed".to_string(),
                })
            }
        });
        if let Some(e) = body_err {
            return Err(e);
        }
        res.map_err(|e| anyhow::anyhow!("CudaGraphExec::capture_with_slots: {e:?}"))
    }

    fn set_scalar_i32(&self, slot: ScalarSlot, value: &i32) -> anyhow::Result<()> {
        // SAFETY: `value` is an i32 matching the tagged kernel arg; it outlives
        // the next launch per the trait contract (the driver keeps the backing
        // store alive through `launch`).
        unsafe { self.set_slot(slot, value) }
            .map_err(|e| anyhow::anyhow!("decode-graph set_scalar_i32: {e:?}"))
    }

    fn set_pointer_u64(&self, slot: ScalarSlot, value: &u64) -> anyhow::Result<()> {
        // SAFETY: `value` is a u64 pointer arg matching the tagged slot and
        // outlives the next launch per the trait contract.
        unsafe { self.set_slot(slot, value) }
            .map_err(|e| anyhow::anyhow!("decode-graph set_pointer_u64: {e:?}"))
    }

    fn set_memcpy_src(&self, slot: MemcpySlot, src: DevicePtr) -> anyhow::Result<()> {
        // SAFETY: `src` is a live device pointer valid for the copy's byte count
        // for the next replay.
        unsafe { self.set_memcpy_slot(slot, src) }
            .map_err(|e| anyhow::anyhow!("decode-graph set_memcpy_src: {e:?}"))
    }

    fn launch(&self, stream: &CudaStream) -> anyhow::Result<()> {
        CudaGraphExec::launch(self, stream)
            .map_err(|e| anyhow::anyhow!("decode-graph launch: {e:?}"))
    }
}
