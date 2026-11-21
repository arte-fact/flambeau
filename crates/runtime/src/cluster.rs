//! `Cluster` — backend-neutral handle to a set of peer GPUs (one rank each).
//! The multi-device analog of `core::Device`: where `Device` is one GPU,
//! `Cluster` is the rank set a PP / TP / Hybrid topology is laid out over.
//! Backends implement it on their concrete cluster type (`HipCluster`,
//! `CudaCluster`); generic forward / server code threads `C: Cluster` rather
//! than naming a backend.
//!
//! Deliberately lean: only the operations consumer crates invoke through the
//! abstraction live here. Per-backend multi-GPU plumbing that no generic
//! caller touches (pinned host-bounce slabs, aux-stream pools, BAR1 peer
//! copy) stays on the concrete type — a trait method nothing calls
//! polymorphically would be the dead-default surface CLAUDE.md rule 3 bans.

use flambeau_core::device::Device;

/// A set of peer GPUs addressed by rank `0..ranks()`. Held behind an `Arc`
/// by the server and the topology drivers.
pub trait Cluster: Send + Sync + 'static {
    /// The per-rank device type (`HipDevice`, `CudaDevice`).
    type Device: Device;

    /// Number of ranks (GPUs) in the cluster.
    fn ranks(&self) -> usize;

    /// The device driving `rank`. Panics if `rank >= ranks()`.
    fn device(&self, rank: usize) -> &Self::Device;

    /// Are all off-diagonal peer edges authorised? The kernel-launched
    /// (BAR1 / NVLink P2P) AllReduce path requires a fully-reachable
    /// matrix; a single unreachable edge forces the host-bounce fallback.
    fn peer_access_full(&self) -> bool;
}
