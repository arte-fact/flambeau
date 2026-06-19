//! CUDA binding for [`Backend`](crate::Backend).

use crate::Backend;

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
