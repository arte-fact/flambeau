//! Per-rank TP synchronization core.
//!
//! Both `gemma4::Gemma4TpStage` and `qwen3-moe::RankForwardScratchTp`
//! carry the same universal piece of state: a per-rank `producer_done_event`
//! recorded on each rank's compute stream after a partial-write
//! kernel, used by peer ranks' AR launches to `stream_wait` before
//! BAR1 reads. [`TpRankCore`] is the shared bundle so both model
//! crates embed one field and the cross-rank barrier helpers in
//! [`crate::tp_sync`] take a slice of cores rather than rebuilding
//! the pattern per model.
//!
//! Partial buffer storage stays on the model struct — the per-rank
//! `Buffer<F16, RowParallel<0>>` fields differ in naming and lifecycle
//! between the two models (qwen3-moe ping-pongs hidden_a/hidden_b;
//! gemma4 uses a single hidden + RawAllocTracker-managed partials).
//! `TpRankCore` deliberately stays narrow to keep both models'
//! existing scratch shape intact.

#![cfg(feature = "hip")]

use flambeau_backend_hip::{HipEvent, HipStream};
use flambeau_core::DeviceResult;

/// Per-rank universal TP synchronization state.
///
/// Embedded as `core: TpRankCore` by each model's per-rank stage
/// struct. The `producer_done_event` is recorded on the rank's
/// compute stream after a row-parallel partial-write kernel (Phase 1
/// / Phase 4 of the canonical TP layer composer); peer ranks then
/// `stream_wait` on this event before launching the AR kernel that
/// reads the partial via BAR1.
pub struct TpRankCore {
    pub rank: usize,
    pub device_id: i32,
    pub producer_done_event: HipEvent,
}

impl TpRankCore {
    /// Construct on a bound device. Caller must have `device.bind()`
    /// in effect (matching the `HipEvent::new` contract).
    pub fn new(rank: usize, device_id: i32) -> DeviceResult<Self> {
        Ok(Self {
            rank,
            device_id,
            producer_done_event: HipEvent::new(device_id)?,
        })
    }

    /// Record `producer_done_event` on `stream`. Host-non-blocking.
    pub fn record_producer_done(&self, stream: &HipStream) -> DeviceResult<()> {
        self.producer_done_event.record(stream)
    }
}
