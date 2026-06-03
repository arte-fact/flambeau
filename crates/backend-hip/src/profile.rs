//! section-level profiling via HipEvent timestamps.
//! rocprofv3 7.1.1 SIGABRTs on multi-rank (≥ 4 ranks) on this rig (0
//! memory). For models that only run at pp4 (Coder-Next-80B, 35B-A3B
//! at all topologies), per-kernel attribution is unavailable through
//! the standard tool. This module provides a coarse alternative:
//! * Caller enables a thread-local section timer via [`enable`].
//! * Forward code along the hot path inserts [`mark`] calls at
//!   well-known section boundaries, each of which lazily allocates
//!   a timing-enabled `HipEvent` and records it on the supplied
//!   stream. When the timer is disabled, [`mark`] is a single
//!   thread-local check (~ns) and a fast-return.
//! * After the workload completes, caller calls [`flush`] to drain
//!   the recorded events and compute per-section ms via
//!   `hipEventElapsedTime`. Events are pairwise consecutive — section
//!   `name[i]` ms = `event[i+1].elapsed_since(event[i])`.
//!   The instrumentation never changes the production hot path's behaviour;
//!   all it adds when enabled is one event-record per mark (low-µs cost on
//!   gfx906) and `O(N_marks)` heap allocations.

use std::cell::RefCell;

use flambeau_core::Device;

use crate::{HipDevice, HipEvent, HipStream};

/// One recorded section boundary — name + the event that signals when
/// that boundary's preceding work is done on its stream.
pub struct SectionEvent {
    pub name: &'static str,
    pub event: HipEvent,
}

/// Per-section aggregated stats — populated by [`flush`].
#[derive(Debug, Clone)]
pub struct SectionStat {
    pub name: &'static str,
    pub total_ms: f32,
    pub count: usize,
    pub mean_ms: f32,
}

thread_local! {
    static TIMER: RefCell<Option<Vec<SectionEvent>>> = const { RefCell::new(None) };
}

/// Enable section recording on this thread. Re-arms the buffer, dropping
/// any previously-recorded events.
pub fn enable() {
    TIMER.with(|t| *t.borrow_mut() = Some(Vec::with_capacity(1024)));
}

/// Whether section recording is currently enabled on this thread. Used
/// internally by [`mark`] to fast-return when disabled.
pub fn is_enabled() -> bool {
    TIMER.with(|t| t.borrow().is_some())
}

/// Record a section boundary on `stream` (which must belong to `device`).
/// No-op when the thread-local timer is disabled.
/// # Errors
/// Propagates HipEvent allocation / record failures.
pub fn mark(name: &'static str, device: &HipDevice, stream: &HipStream) -> anyhow::Result<()> {
    TIMER.with(|t| -> anyhow::Result<()> {
        let mut guard = t.borrow_mut();
        let Some(events) = guard.as_mut() else {
            return Ok(());
        };
        let event = HipEvent::new_timing(device.id())
            .map_err(|e| anyhow::anyhow!("HipEvent::new_timing for `{name}`: {e}"))?;
        event
            .record(stream)
            .map_err(|e| anyhow::anyhow!("HipEvent::record for `{name}`: {e}"))?;
        events.push(SectionEvent { name, event });
        Ok(())
    })
}

/// Synchronise every recorded event, compute pairwise ms deltas, and
/// aggregate by section name. Disables the timer.
/// The delta for `events[i]` is the elapsed time between `events[i-1].event`
/// and `events[i].event`, attributed to `events[i].name`. The very first
/// event has no predecessor and contributes nothing.
pub fn flush() -> anyhow::Result<Vec<SectionStat>> {
    let events: Vec<SectionEvent> = TIMER.with(|t| t.borrow_mut().take().unwrap_or_default());
    if events.len() < 2 {
        return Ok(Vec::new());
    }
    // Sync once at the end — events were recorded async on possibly
    // different streams, so just sync each before reading. HipEvent::synchronize
    // is per-event; cheaper to sync the whole batch by syncing the last
    // event then trusting that prior events on the same stream are already
    // resolved. For cross-stream events we sync each.
    for ev in &events {
        ev.event
            .synchronize()
            .map_err(|e| anyhow::anyhow!("HipEvent::synchronize `{}`: {e}", ev.name))?;
    }

    use std::collections::HashMap;
    let mut by_name: HashMap<&'static str, (f32, usize)> = HashMap::new();
    for window in events.windows(2) {
        let prev = &window[0];
        let cur = &window[1];
        // hipEventElapsedTime requires both events to be on the same
        // device. Cross-device deltas are undefined; skip them.
        if prev.event.device_id() != cur.event.device_id() {
            continue;
        }
        let ms = cur
            .event
            .elapsed_ms_since(&prev.event)
            .map_err(|e| anyhow::anyhow!("elapsed_ms `{}` ← `{}`: {e}", cur.name, prev.name))?;
        let entry = by_name.entry(cur.name).or_insert((0.0, 0));
        entry.0 += ms;
        entry.1 += 1;
    }

    let mut stats: Vec<SectionStat> = by_name
        .into_iter()
        .map(|(name, (total, count))| SectionStat {
            name,
            total_ms: total,
            count,
            mean_ms: if count > 0 { total / count as f32 } else { 0.0 },
        })
        .collect();
    // Sort by total_ms descending so callers see the hot-spot first.
    stats.sort_unstable_by(|a, b| {
        b.total_ms
            .partial_cmp(&a.total_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(stats)
}
