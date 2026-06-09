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

    /// Build the per-launch op object for a `(registry, stream)` pair —
    /// the backend-neutral form of `HipOps::new(reg, stream)`.
    fn ops<'a>(reg: &'a Self::Registry, stream: &'a Self::Stream) -> Self::Ops<'a>;
}

#[cfg(feature = "hip")]
mod hip;
#[cfg(feature = "hip")]
pub use hip::HipBackend;
