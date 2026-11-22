//! Backend-binding layer.
//!
//! [`Backend`] is the type-level bundle the topology executor (`flambeau-forward`)
//! and the model crates are generic over: every device / stream / event /
//! cluster / op-registry they touch is projected off one `B: Backend`
//! parameter, so they name no GPU vendor (forward CLAUDE.md rule 9).
//!
//! The seam traits themselves live at their natural layer — `Device` / `Event`
//! in `core`, `Cluster` in `runtime`, `Ops` in `ops`. This crate only ties
//! them together and hosts the per-vendor impls, because a `Backend` impl
//! must name concrete `Hip*` / `Cuda*` types and must sit above `ops` (it
//! references `Ops`), where neither `forward` (no arch types) nor `model-ops`
//! (no backend trait) will host it.

use flambeau_core::device::{Device, Event, Stream};
use flambeau_core::{DevicePtr, MemcpySlot, ScalarSlot};
use flambeau_ops::Ops;
use flambeau_runtime::Cluster;

/// One GPU backend's concrete type set. Implementors are uninhabited tags —
/// `Backend` is never constructed, only used at the type level to project the
/// associated types into generic executor / model signatures.
pub trait Backend: Send + Sync + 'static {
    /// Per-rank device (`HipDevice`, `CudaDevice`).
    type Device: Device<Stream = Self::Stream, Event = Self::Event>;
    /// Compute stream (`HipStream`, CUDA stream wrapper).
    type Stream: Stream;
    /// Cross-stream ordering event (`HipEvent`, `CudaEvent`).
    type Event: Event<Self::Stream>;
    /// Multi-GPU rank set (`HipCluster`, `CudaCluster`).
    type Cluster: Cluster<Device = Self::Device>;
    /// Persistent per-device kernel-module registry (`OpsRegistry`). Held by
    /// the executor for the session; loaded modules outlive every launch.
    type Registry: Send + Sync + 'static;
    /// Per-launch op object binding a `(registry, stream)` pair (`HipOps<'a>`).
    /// Cheap to build per stream; implements the portable [`Ops`] surface.
    type Ops<'a>: Ops
    where
        Self: 'a;

    /// Single-token decode-graph executor for this backend (real on CUDA, a
    /// null unit type on HIP). The session driver captures the n=1 decode
    /// forward once, then rebinds slots + replays through this seam — so no
    /// `Cuda*`/`Hip*` graph type leaks into `flambeau-forward` (rule 9).
    type DecodeGraph: DecodeGraphExec<Self>;

    /// Whether [`Self::DecodeGraph`] can actually capture + replay the n=1
    /// decode graph. A genuine capability query — CUDA answers `true`, HIP
    /// `false` (gfx906 graph capture is null-to-negative). Both answer
    /// concretely, so this is not an `is_<arch>` predicate: the driver gates on
    /// the capability, never on the backend's identity.
    fn supports_decode_graph() -> bool {
        false
    }

    /// Build the per-launch op object for a `(registry, stream)` pair —
    /// the backend-neutral form of `HipOps::new(reg, stream)`.
    fn ops<'a>(reg: &'a Self::Registry, stream: &'a Self::Stream) -> Self::Ops<'a>;

    /// Open device `id` on this backend — the backend-neutral form of
    /// `HipDevice::new(id)` / `CudaDevice::new(id)`. `Device::new` is inherent
    /// (not on the `Device` trait), so the bundle projects it here for generic
    /// single-device construction.
    fn new_device(id: i32) -> anyhow::Result<Self::Device>;

    /// Load the per-device kernel-module registry — the backend-neutral form of
    /// `OpsRegistry::new(device)` / `CudaOpsRegistry::new(device)`.
    fn new_registry(device: &Self::Device) -> anyhow::Result<Self::Registry>;
}

/// Capture + replay a single-token decode graph, rebinding per-token slots
/// between replays. Generic over the backend bundle `B` so the session driver
/// names no vendor type. Implemented by `CudaGraphExec` (real) and the HIP null
/// unit type (never constructed — `capture` bails and the driver gates on
/// [`Backend::supports_decode_graph`]).
pub trait DecodeGraphExec<B: Backend + ?Sized>: Sized + Send + Sync {
    /// Capture the stream work enqueued by `body` (run ONCE) into a replayable
    /// graph with a slot map. Slot-tagged kernel args + memcpy sources become
    /// rebindable via the setters below.
    fn capture(
        stream: &B::Stream,
        body: impl FnOnce(&B::Stream) -> anyhow::Result<()>,
    ) -> anyhow::Result<Self>;

    /// Rebind a scalar `i32` kernel arg. `value` must stay live until the next
    /// [`Self::launch`] (the graph executor holds the pointer across sibling
    /// slot updates on the same node).
    fn set_scalar_i32(&self, slot: ScalarSlot, value: &i32) -> anyhow::Result<()>;

    /// Rebind a `u64` pointer kernel arg. Same liveness contract as
    /// [`Self::set_scalar_i32`].
    fn set_pointer_u64(&self, slot: ScalarSlot, value: &u64) -> anyhow::Result<()>;

    /// Retarget a memcpy node's source pointer (the embedding-row lookup).
    fn set_memcpy_src(&self, slot: MemcpySlot, src: DevicePtr) -> anyhow::Result<()>;

    /// Replay the captured graph on `stream` (asynchronous).
    fn launch(&self, stream: &B::Stream) -> anyhow::Result<()>;
}

#[cfg(feature = "hip")]
mod hip;
#[cfg(feature = "hip")]
pub use hip::HipBackend;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;
