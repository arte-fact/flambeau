//! Cross-rank stream synchronization helpers for the TP forward path.
//!
//! All TP collectives in flambeau use BAR1 P2P AllReduce kernels that
//! read peer partial buffers over the BAR aperture. The AR kernel
//! must not launch before the peer's Phase-1 partial-write kernel
//! completes — otherwise the BAR1 reads land on pre-Phase-1 bytes.
//!
//! [`cross_rank_event_barrier`] expresses this ordering as a driver-
//! side DAG edge using `HipEvent::record` + `HipEvent::stream_wait`,
//! avoiding the host-blocking `Stream::synchronize` alternative.
//!
//! Both gemma4 (`Gemma4TpStage::core.producer_done_event`) and
//! qwen3-moe (`RankForwardScratchTp::core.producer_done_event`) call
//! into this helper instead of inlining the record + cross-peer
//! `stream_wait` dance.

#![cfg(feature = "hip")]

use flambeau_backend_hip::HipCluster;
use flambeau_core::{Device, DeviceResult};

use crate::tp_rank_core::TpRankCore;

/// Cross-rank stream barrier via `producer_done_event`.
///
/// Step 1: every rank records its `producer_done_event` on its
/// compute stream. Step 2: every rank's stream waits on every peer
/// rank's event before subsequent work.
///
/// After this returns, any kernel launched on rank `r`'s default
/// stream is guaranteed (driver-side, no host block) not to execute
/// before every other rank's pre-call work has finished.
///
/// Replaces the inline pattern at gemma4 `tp.rs` Phases 2/5 and
/// qwen3-moe `forward/tp.rs::ar_residual`.
pub fn cross_rank_event_barrier(cluster: &HipCluster, cores: &[&TpRankCore]) -> DeviceResult<()> {
    let n = cluster.ranks();
    assert_eq!(
        cores.len(),
        n,
        "cross_rank_event_barrier: expected {n} cores, got {}",
        cores.len()
    );

    for r in 0..n {
        let device = cluster.device(r);
        device.bind()?;
        cores[r].record_producer_done(device.default_stream())?;
    }

    for r in 0..n {
        let device = cluster.device(r);
        device.bind()?;
        let stream = device.default_stream();
        for peer in 0..n {
            if peer != r {
                cores[peer].producer_done_event.stream_wait(stream)?;
            }
        }
    }

    Ok(())
}
