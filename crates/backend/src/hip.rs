//! HIP / ROCm binding for [`Backend`](crate::Backend).

use crate::Backend;

/// HIP backend tag. Never constructed; only names the concrete HIP type set
/// for a generic executor's `B: Backend` parameter.
pub enum HipBackend {}

impl Backend for HipBackend {
    type Device = flambeau_backend_hip::HipDevice;
    type Stream = flambeau_backend_hip::HipStream;
    type Event = flambeau_backend_hip::HipEvent;
    type Cluster = flambeau_backend_hip::HipCluster;
    type Registry = flambeau_ops::OpsRegistry;
    type Ops<'a> = flambeau_ops::HipOps<'a>;

    fn ops<'a>(reg: &'a Self::Registry, stream: &'a Self::Stream) -> Self::Ops<'a> {
        flambeau_ops::HipOps::new(reg, stream)
    }
}
