//! `StageCommon` — shared bookkeeping for per-rank stage structs.
//!
//! Composed-into (not inherited-from) by every `*Stage` struct across
//! both arches × all three topologies (qwen3-moe PP/TP/Hybrid +
//! gemma4 PP/TP/Hybrid = 6 sites today, mistral / qwen3-coder-next on
//! the roadmap). Replaces the ~30-LOC `dispose` + Drop-warn pattern
//! that was being duplicated at each site.
//!
//! The pattern each Stage adopts:
//!
//! ```ignore
//! pub struct MyStage {
//!     pub common: StageCommon,         // <-- bookkeeping + tracker
//!     // arch / topology specific fields here:
//!     pub layer_weights: Vec<LayerWeights>,
//!     pub kv_caches: Vec<Option<KvCache<...>>>,
//!     pub scratch: MyScratchPtrs,
//!     // ...
//! }
//!
//! impl MyStage {
//!     pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
//!         // Per-stage cleanup that isn't in `raw_alloc` (e.g.
//!         // typed `KvCache` allocations, which own their own bytes).
//!         for kv in self.kv_caches.drain(..).flatten() {
//!             kv.dispose(device)?;
//!         }
//!         // Then drop everything tracked by the shared tracker.
//!         self.common.dispose(device)
//!     }
//! }
//!
//! impl Drop for MyStage {
//!     fn drop(&mut self) {
//!         self.common.warn_on_leak("flambeau_<arch>::<topology>::Stage");
//!     }
//! }
//! ```
//!
//! Stages allocate scratch + weight bytes through `common.raw_alloc`
//! so the single `dispose(device)` call frees the lot. The discipline
//! is "tracked-by-default, never `device.alloc` directly inside the
//! Stage".
//!
//! CLAUDE.md rule 14 motivates this: 6 existing duplicate sites with
//! a 7th + 8th on the roadmap is past the rule-of-three threshold.
//! Composition (each Stage *contains* StageCommon) keeps topology-
//! specific knobs (`stage_idx`, `partial_attn_f32`, `tp_moe_scratch`,
//! `core: TpRankCore`, …) concrete on the outer struct.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::HipDevice;

use crate::driver_utils::RawAllocTracker;

/// Per-rank stage bookkeeping. Holds the shared `RawAllocTracker` +
/// a rank tag for diagnostics + an idempotency flag for `dispose`.
pub struct StageCommon {
    /// All raw `(DevicePtr, bytes)` allocations the stage owns are
    /// tracked here. `dispose(device)` walks them and `device.dealloc`s.
    /// Stages should use `common.raw_alloc.alloc_*` (or `.track(...)`
    /// for externally-allocated pointers) so every byte is freed in
    /// one place.
    pub raw_alloc: RawAllocTracker,
    /// Rank or stage-index (model-side meaning). Used for Drop warning.
    pub rank: u32,
    /// HIP device id (= `cluster.device(rank).id()`). Used for Drop
    /// warning + diagnostics. Stages can also read this back if they
    /// need to bind without round-tripping through the cluster handle.
    pub device_id: i32,
    /// Idempotency latch. Flipped by `dispose`; checked by `Drop` so
    /// the Drop-warn fires only when the stage was actually leaked.
    disposed: bool,
}

impl StageCommon {
    pub fn new(rank: u32, device_id: i32) -> Self {
        Self {
            raw_alloc: RawAllocTracker::new(),
            rank,
            device_id,
            disposed: false,
        }
    }

    /// `true` after `dispose` has run (so the Drop-warn doesn't fire).
    pub fn is_disposed(&self) -> bool {
        self.disposed
    }

    /// Free every allocation tracked by `raw_alloc`. Idempotent — a
    /// second call is a no-op.
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        self.raw_alloc.dispose(device)
    }

    /// Drop helper. `tag` is included as a structured field so logs
    /// can identify the arch / topology that leaked. Emits a one-line
    /// warn whenever the stage was dropped without `dispose`. We
    /// deliberately do NOT also gate on `raw_alloc.is_empty()` — not
    /// every stage tracks its resources through the tracker (qwen3-moe
    /// shards own `DeviceTensor` fields directly), and gating on the
    /// tracker would silently suppress real leak warnings on those
    /// stages. Test fixtures that legitimately build a Stage without
    /// resources should call `dispose(device)` once at teardown
    /// (it's a no-op on an empty tracker).
    pub fn warn_on_leak(&self, tag: &'static str) {
        if !self.disposed {
            tracing::warn!(
                tag,
                rank = self.rank,
                device_id = self.device_id,
                allocs = self.raw_alloc.allocs.len(),
                "Stage dropped without dispose(device); device buffers leaked"
            );
        }
    }
}
