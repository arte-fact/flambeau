//! HIP / ROCm binding for [`Backend`](crate::Backend).

use flambeau_backend_hip::HipStream;
use flambeau_core::{DevicePtr, MemcpySlot, ScalarSlot};

use crate::{Backend, DecodeGraphExec};

/// HIP backend tag. Never constructed; only names the concrete HIP type set
/// for a generic executor's `B: Backend` parameter.
pub enum HipBackend {}

/// Null decode-graph executor for HIP. Uninhabited — the driver never captures
/// on HIP ([`Backend::supports_decode_graph`] returns `false` for
/// `HipBackend`), so this only exists to satisfy the associated type. `capture`
/// bails; the other methods are unreachable (no value can be constructed).
pub enum NullDecodeGraph {}

impl Backend for HipBackend {
    type Device = flambeau_backend_hip::HipDevice;
    type Stream = flambeau_backend_hip::HipStream;
    type Event = flambeau_backend_hip::HipEvent;
    type Cluster = flambeau_backend_hip::HipCluster;
    type Registry = flambeau_ops::OpsRegistry;
    type Ops<'a> = flambeau_ops::HipOps<'a>;
    type DecodeGraph = NullDecodeGraph;

    fn ops<'a>(reg: &'a Self::Registry, stream: &'a Self::Stream) -> Self::Ops<'a> {
        flambeau_ops::HipOps::new(reg, stream)
    }

    fn new_device(id: i32) -> anyhow::Result<Self::Device> {
        Ok(flambeau_backend_hip::HipDevice::new(id)?)
    }

    fn new_registry(device: &Self::Device) -> anyhow::Result<Self::Registry> {
        Ok(flambeau_ops::OpsRegistry::new(device)?)
    }
}

impl DecodeGraphExec<HipBackend> for NullDecodeGraph {
    fn capture(
        _stream: &HipStream,
        _body: impl FnOnce(&HipStream) -> anyhow::Result<()>,
    ) -> anyhow::Result<Self> {
        anyhow::bail!(
            "HipBackend has no decode-graph support (supports_decode_graph() == false); \
             the driver must never call capture on HIP"
        )
    }

    fn set_scalar_i32(&self, _slot: ScalarSlot, _value: &i32) -> anyhow::Result<()> {
        match *self {}
    }

    fn set_pointer_u64(&self, _slot: ScalarSlot, _value: &u64) -> anyhow::Result<()> {
        match *self {}
    }

    fn set_memcpy_src(&self, _slot: MemcpySlot, _src: DevicePtr) -> anyhow::Result<()> {
        match *self {}
    }

    fn launch(&self, _stream: &HipStream) -> anyhow::Result<()> {
        match *self {}
    }
}
