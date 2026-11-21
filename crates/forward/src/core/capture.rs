//! Backend-neutral decode-graph capture accumulator + rebind plan.
//!
//! During a *capturing* single-token decode forward, each composite that
//! issues a per-token-varying kernel arg or memcpy source allocates a slot
//! ([`ScalarSlot`] / [`MemcpySlot`], both `flambeau_core`) and records a
//! [`RebindEntry`] describing how to recompute that value from the token's
//! `(position, token_id)`. The forward runs ONCE; the session driver then
//! evaluates the accumulated [`DecodeRebindPlan`] every subsequent token —
//! `set_scalar_i32` / `set_pointer_u64` / `set_memcpy_src` per entry — and
//! replays the captured graph. The forward function does not re-run.
//!
//! Nothing here names a `Cuda*`/`Hip*` type (forward CLAUDE.md rule 9): the
//! plan is plain data, evaluated against the backend graph executor through
//! the `flambeau_backend::DecodeGraphExec` seam.

use flambeau_core::{MemcpySlot, ScalarSlot};

/// How a per-token `i32` scalar arg is recomputed from `position` each replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarRule {
    /// KV tail length passed to the attention kernels: `position + 1`.
    NTokensKv,
    /// Split-K window start for ring SWA layers: `(position + 1) - window`
    /// (saturating). `window == 0` collapses to `position + 1`, so this is
    /// only recorded for `window > 0` layers.
    ChunkBase { window: usize },
    /// KV write row for the append kernels: `position`.
    WritePos,
    /// A value that does not vary with `position` (e.g. `chunk_base == 0` on a
    /// full-context global layer). Rebound to the same constant every replay so
    /// a single `_slots` signature covers both the varying and fixed cases.
    Const { value: i32 },
}

impl ScalarRule {
    /// Recompute the scalar for `position`.
    pub fn eval(self, position: usize) -> i32 {
        match self {
            ScalarRule::NTokensKv => (position + 1) as i32,
            ScalarRule::ChunkBase { window } => (position + 1).saturating_sub(window) as i32,
            ScalarRule::WritePos => position as i32,
            ScalarRule::Const { value } => value,
        }
    }
}

/// One rebindable value recorded during a capture pass.
#[derive(Clone, Copy, Debug)]
pub enum RebindEntry {
    /// A scalar kernel arg tagged via a `_slots` op variant.
    ScalarI32 { slot: ScalarSlot, rule: ScalarRule },
    /// A `u64` pointer kernel arg = `base + position * stride_bytes`
    /// (contiguous, non-ring append dst). Rebound via `set_pointer_u64`.
    PointerU64 {
        slot: ScalarSlot,
        base: usize,
        stride_bytes: usize,
    },
    /// A `u64` pointer kernel arg = `base + (position - sub_tokens) *
    /// stride_bytes` — the SWA SlideBand K/V cache pointer, slid to the
    /// window start (`sub_tokens = window - 1`, so the offset is
    /// `position + 1 - window` rows). Only valid in-band (`position >=
    /// window > sub_tokens`); rebound via `set_pointer_u64`.
    PointerSlide {
        slot: ScalarSlot,
        base: usize,
        stride_bytes: usize,
        sub_tokens: usize,
    },
    /// A memcpy-node source = `base + token_id * stride_bytes` (the embedding
    /// row lookup). Rebound via `set_memcpy_src`.
    MemcpySrc {
        slot: MemcpySlot,
        base: usize,
        stride_bytes: usize,
    },
}

/// A witness of one captured split-K attention launch's grid shape, recorded so
/// the session driver can decide — arch-generically — whether a cached graph is
/// still valid at a later position. The captured graph bakes `n_chunks` (the
/// grid) and the regime-dependent kernel args (`ring_depth`, `kernel_window`,
/// `chunk_base`-rule) as constants; a token that would change any of them needs
/// a recapture, not a slot rebind. The driver recomputes `(ring_active,
/// n_chunks)` from the new position using the SAME arithmetic the composite ran
/// (`splitk_chunk_size`) and requires an exact match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnGridWitness {
    /// This layer ring-buffers its KV slab (SWA). `false` = full-context.
    pub is_ring_layer: bool,
    /// The layer's KV slab depth in rows (ring modulus). `max_seq_len` for
    /// full-context layers.
    pub slab_depth: usize,
    /// SWA window radius; `0` for a global/full-context layer.
    pub window_size: usize,
    /// `ring_active` at capture (ring layer whose slab has wrapped).
    pub ring_active: bool,
    /// The split-K grid.y baked at capture. The `_dpos` kernels run any
    /// chunk past the KV tail as an empty partial the combine weights 0, so
    /// the graph replays for every position needing at most this many chunks.
    pub n_chunks: usize,
    /// `chunk_size` (a baked kernel arg, not a slot) at capture.
    pub chunk_size: usize,
    /// Captured in the SWA SlideBand with the K/V pointers tagged as
    /// `PointerSlide` slots: the launch shape is constant in-band
    /// (`eff_n_tokens_kv = window`, `chunk_base = 0`), so the graph replays
    /// for any in-band position and needs recapture only at ring entry.
    pub slid: bool,
    /// Smallest KV tail the captured split-K replays bit-identically to the
    /// non-graph path (which takes the single-pass kernel below
    /// `SPLITK_MIN_TOKENS_KV + 1`).
    pub min_tokens_kv: usize,
}

/// The split-K chunk size a captured decode bakes for a `chunk_len`-token KV
/// tail: the non-graph heuristic, floored at `SPLITK_MIN_TOKENS_KV` so a tail
/// the non-graph path runs single-pass is one chunk of a reusable size.
pub fn decode_graph_chunk_size(chunk_len: usize) -> usize {
    flambeau_model_ops::splitk_chunk_size(chunk_len, crate::core::scratch::MAX_SPLITK_CHUNKS)
        .max(flambeau_ops::attention::SPLITK_MIN_TOKENS_KV)
}

/// When a capture marked uncapturable becomes worth re-attempting. Without
/// this the driver would pay a full capture+instantiate+discard EVERY token
/// while the uncapturable condition persists (measured ~5-7 ms/token).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetryAt {
    /// Position-dependent condition (SWA SlideBand): capturable again once
    /// `position >= this` (the layer's ring regime activates).
    Position(usize),
    /// Config-level condition (Q8 KV path, missing `_slots` wiring): fixed
    /// for the session, never re-attempt.
    Never,
}

/// Accumulator handed to the composites (via `CoreState::capture`) during the
/// single capture forward. Each composite, when capturing, calls one of the
/// `record_*` helpers to allocate a slot + record its rebind rule, tags the
/// kernel arg / memcpy with that slot, and proceeds exactly as the non-capture
/// path otherwise.
#[derive(Debug, Default)]
pub struct DecodeCapture {
    entries: Vec<RebindEntry>,
    witnesses: Vec<AttnGridWitness>,
    /// Set when a composite hits a branch that cannot be represented as a
    /// per-token rebind (single-pass attention with a baked KV length, an
    /// SWA pointer-slide, or a ring Q8 dst). The driver checks this after the
    /// capture forward and discards the graph (correctness falls back to a
    /// normal forward) rather than caching a graph that replays wrong. The
    /// first reason string is kept; the retry horizon escalates to the
    /// latest across all marks (`Never` dominates).
    uncapturable: Option<(&'static str, RetryAt)>,
}

impl DecodeCapture {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a scalar slot bound to `rule`, record it, and return it so the
    /// caller can tag the matching kernel arg via the `_slots` op variant.
    pub fn record_scalar(&mut self, rule: ScalarRule) -> ScalarSlot {
        let slot = ScalarSlot::new();
        self.entries.push(RebindEntry::ScalarI32 { slot, rule });
        slot
    }

    /// Allocate a scalar slot carrying a `u64` pointer = `base + pos * stride`.
    pub fn record_pointer(&mut self, base: usize, stride_bytes: usize) -> ScalarSlot {
        let slot = ScalarSlot::new();
        self.entries
            .push(RebindEntry::PointerU64 { slot, base, stride_bytes });
        slot
    }

    /// Allocate a scalar slot carrying a SlideBand `u64` pointer =
    /// `base + (pos - sub_tokens) * stride` (see [`RebindEntry::PointerSlide`]).
    pub fn record_pointer_slide(
        &mut self,
        base: usize,
        stride_bytes: usize,
        sub_tokens: usize,
    ) -> ScalarSlot {
        let slot = ScalarSlot::new();
        self.entries.push(RebindEntry::PointerSlide {
            slot,
            base,
            stride_bytes,
            sub_tokens,
        });
        slot
    }

    /// Allocate a memcpy slot carrying a source = `base + token_id * stride`.
    pub fn record_memcpy_src(&mut self, base: usize, stride_bytes: usize) -> MemcpySlot {
        let slot = MemcpySlot::new();
        self.entries
            .push(RebindEntry::MemcpySrc { slot, base, stride_bytes });
        slot
    }

    /// Record a split-K attention grid witness (see [`AttnGridWitness`]).
    pub fn record_attn_grid(&mut self, witness: AttnGridWitness) {
        self.witnesses.push(witness);
    }

    /// Mark this capture unusable for replay for a session-permanent reason
    /// (see [`DecodeCapture::uncapturable`]).
    pub fn mark_uncapturable(&mut self, why: &'static str) {
        self.mark_uncapturable_until(why, RetryAt::Never);
    }

    /// Mark unusable with an explicit retry horizon. The first reason string
    /// is kept; the retry horizon escalates to the latest across marks.
    pub fn mark_uncapturable_until(&mut self, why: &'static str, retry: RetryAt) {
        self.uncapturable = Some(match self.uncapturable {
            None => (why, retry),
            Some((first, prev)) => {
                let merged = match (prev, retry) {
                    (RetryAt::Never, _) | (_, RetryAt::Never) => RetryAt::Never,
                    (RetryAt::Position(a), RetryAt::Position(b)) => RetryAt::Position(a.max(b)),
                };
                (first, merged)
            }
        });
    }

    pub fn is_uncapturable(&self) -> Option<&'static str> {
        self.uncapturable.map(|(why, _)| why)
    }

    pub fn uncapturable_retry(&self) -> Option<RetryAt> {
        self.uncapturable.map(|(_, retry)| retry)
    }

    pub fn entries(&self) -> &[RebindEntry] {
        &self.entries
    }

    /// Freeze the accumulated entries + grid witnesses into an evaluable plan.
    pub fn into_plan(self) -> DecodeRebindPlan {
        DecodeRebindPlan {
            entries: self.entries,
            witnesses: self.witnesses,
        }
    }
}

/// A resolved value the driver must write onto the captured graph before a
/// replay. Produced by [`DecodeRebindPlan::evaluate`]; consumed by the
/// backend graph executor (`set_scalar_i32` / `set_pointer_u64` /
/// `set_memcpy_src`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RebindValue {
    ScalarI32 { slot: ScalarSlot, value: i32 },
    PointerU64 { slot: ScalarSlot, value: u64 },
    MemcpySrc { slot: MemcpySlot, value: usize },
}

/// The frozen per-token rebind program for one captured decode graph. Evaluated
/// against `(position, token_id)` each replay; the resulting [`RebindValue`]s
/// are applied to the graph executor before `launch`.
#[derive(Debug, Default)]
pub struct DecodeRebindPlan {
    pub entries: Vec<RebindEntry>,
    pub witnesses: Vec<AttnGridWitness>,
}

impl DecodeRebindPlan {
    /// Whether the captured graph is still valid to replay at `position` — i.e.
    /// every recorded split-K attention grid recomputes, with the composite's
    /// `chunk_size` arithmetic (`chunk_len_chunk_size`), to the SAME
    /// `(ring_active, chunk_size)` it had at capture and at most its baked
    /// `n_chunks`, at a KV tail of at least `min_tokens_kv`. Any mismatch, or a
    /// token that lands in the uncapturable SWA SlideBand, means the driver must
    /// recapture (grid dims / regime constants are baked, not slots).
    pub fn reusable_at(
        &self,
        position: usize,
        chunk_len_chunk_size: impl Fn(usize) -> usize,
    ) -> bool {
        let n_tokens_kv = position + 1;
        self.witnesses.iter().all(|w| {
            let ring_active = w.is_ring_layer && n_tokens_kv > w.slab_depth;
            if w.slid {
                // Slid capture replays only while still in the SlideBand
                // (window < KV tail <= slab_depth); ring entry or a slot
                // reset below the window forces recapture. In-band the
                // chunking is position-independent (chunk_len = window) —
                // the equality check below is a safety net.
                if ring_active || w.window_size == 0 || n_tokens_kv <= w.window_size
                    || n_tokens_kv > w.slab_depth
                {
                    return false;
                }
                let chunk_size = chunk_len_chunk_size(w.window_size).max(1);
                let n_chunks = w.window_size.div_ceil(chunk_size);
                return n_chunks <= w.n_chunks && chunk_size == w.chunk_size;
            }
            // SWA SlideBand at the new position — the composite would take the
            // pointer-slide branch this (non-slid) capture did not tag.
            if !ring_active && w.window_size > 0 && w.window_size < n_tokens_kv {
                return false;
            }
            let chunk_len = if ring_active && w.window_size > 0 {
                w.window_size
            } else {
                n_tokens_kv
            };
            let chunk_size = chunk_len_chunk_size(chunk_len).max(1);
            let n_chunks = chunk_len.div_ceil(chunk_size);
            ring_active == w.ring_active
                && n_tokens_kv >= w.min_tokens_kv
                && n_chunks <= w.n_chunks
                && chunk_size == w.chunk_size
        })
    }
    /// Resolve every entry for a token at `position` with id `token_id`.
    pub fn evaluate(&self, position: usize, token_id: u32) -> Vec<RebindValue> {
        self.entries
            .iter()
            .map(|e| match *e {
                RebindEntry::ScalarI32 { slot, rule } => RebindValue::ScalarI32 {
                    slot,
                    value: rule.eval(position),
                },
                RebindEntry::PointerU64 {
                    slot,
                    base,
                    stride_bytes,
                } => RebindValue::PointerU64 {
                    slot,
                    value: (base + position * stride_bytes) as u64,
                },
                RebindEntry::PointerSlide {
                    slot,
                    base,
                    stride_bytes,
                    sub_tokens,
                } => {
                    debug_assert!(
                        position > sub_tokens,
                        "PointerSlide evaluated out of band: position {position} <= sub_tokens {sub_tokens}"
                    );
                    RebindValue::PointerU64 {
                        slot,
                        value: (base + (position - sub_tokens) * stride_bytes) as u64,
                    }
                }
                RebindEntry::MemcpySrc {
                    slot,
                    base,
                    stride_bytes,
                } => RebindValue::MemcpySrc {
                    slot,
                    value: base + (token_id as usize) * stride_bytes,
                },
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_rule_eval_matches_composite_math() {
        assert_eq!(ScalarRule::NTokensKv.eval(300), 301);
        assert_eq!(ScalarRule::WritePos.eval(300), 300);
        assert_eq!(ScalarRule::ChunkBase { window: 1024 }.eval(1500), 477);
        // Below the window the split-K base saturates to 0 (pre-window regime).
        assert_eq!(ScalarRule::ChunkBase { window: 1024 }.eval(500), 0);
        assert_eq!(ScalarRule::Const { value: 0 }.eval(999), 0);
    }

    #[test]
    fn capture_accumulates_entries_in_order() {
        let mut cap = DecodeCapture::new();
        let embed = cap.record_memcpy_src(0x1000, 24);
        let n_kv = cap.record_scalar(ScalarRule::NTokensKv);
        let base = cap.record_scalar(ScalarRule::ChunkBase { window: 1024 });
        let wpos = cap.record_scalar(ScalarRule::WritePos);
        let dst = cap.record_pointer(0x9000, 34);
        assert!(cap.is_uncapturable().is_none());
        assert_eq!(cap.entries().len(), 5);
        // Slots are process-unique and distinct.
        assert_ne!(n_kv.id(), base.id());
        assert_ne!(base.id(), wpos.id());
        let _ = (embed, dst);
    }

    #[test]
    fn plan_evaluates_every_entry_from_position_and_token() {
        let mut cap = DecodeCapture::new();
        let embed = cap.record_memcpy_src(0x1000, 24);
        let n_kv = cap.record_scalar(ScalarRule::NTokensKv);
        let base = cap.record_scalar(ScalarRule::ChunkBase { window: 1024 });
        let wpos = cap.record_scalar(ScalarRule::WritePos);
        let dst = cap.record_pointer(0x9000, 34);
        let plan = cap.into_plan();
        let vals = plan.evaluate(1500, 42);
        assert_eq!(
            vals,
            vec![
                RebindValue::MemcpySrc { slot: embed, value: 0x1000 + 42 * 24 },
                RebindValue::ScalarI32 { slot: n_kv, value: 1501 },
                RebindValue::ScalarI32 { slot: base, value: 477 },
                RebindValue::ScalarI32 { slot: wpos, value: 1500 },
                RebindValue::PointerU64 { slot: dst, value: (0x9000 + 1500 * 34) as u64 },
            ]
        );
    }

    #[test]
    fn reusable_at_tracks_global_split_k_bucket() {
        // Global full-context layer captured at position 300 (KV tail 301),
        // chunk_size 128 → n_chunks = ceil(301/128) = 3.
        let cs = |cl: usize| if cl <= 256 { cl } else { 128 };
        let mut cap = DecodeCapture::new();
        cap.record_scalar(ScalarRule::NTokensKv);
        cap.record_attn_grid(AttnGridWitness {
            is_ring_layer: false,
            slab_depth: 4096,
            window_size: 0,
            ring_active: false,
            n_chunks: 3,
            chunk_size: 128,
            slid: false,
            min_tokens_kv: 0,
        });
        let plan = cap.into_plan();
        // Same bucket [257, 384]: n_chunks stays 3 → replayable.
        assert!(plan.reusable_at(300, cs));
        assert!(plan.reusable_at(383, cs)); // 384/128 = 3
        // Crosses into n_chunks = 4 at KV tail 385 → recapture.
        assert!(!plan.reusable_at(384, cs)); // 385/128 -> 4
        assert!(!plan.reusable_at(512, cs)); // 513/128 -> 5
    }

    #[test]
    fn baked_grid_replays_every_tail_with_its_chunk_size() {
        let mut cap = DecodeCapture::new();
        cap.record_attn_grid(AttnGridWitness {
            is_ring_layer: false,
            slab_depth: 8192,
            window_size: 0,
            ring_active: false,
            n_chunks: 32,
            chunk_size: decode_graph_chunk_size(1),
            slid: false,
            min_tokens_kv: 1,
        });
        let plan = cap.into_plan();
        // chunk_size 32 from the first decode token up to a 1024-token tail.
        for pos in [0_usize, 5, 31, 32, 33, 500, 1023] {
            assert!(plan.reusable_at(pos, decode_graph_chunk_size), "pos {pos}");
        }
        // Past 1024 the heuristic's chunk_size grows: recapture.
        assert!(!plan.reusable_at(1024, decode_graph_chunk_size));
    }

    #[test]
    fn min_tokens_kv_keeps_short_tails_off_the_graph() {
        let mut cap = DecodeCapture::new();
        cap.record_attn_grid(AttnGridWitness {
            is_ring_layer: false,
            slab_depth: 8192,
            window_size: 0,
            ring_active: false,
            n_chunks: 32,
            chunk_size: 32,
            slid: false,
            min_tokens_kv: 33,
        });
        let plan = cap.into_plan();
        assert!(!plan.reusable_at(31, decode_graph_chunk_size));
        assert!(plan.reusable_at(32, decode_graph_chunk_size));
    }

    #[test]
    fn reusable_at_rejects_slideband() {
        let cs = |cl: usize| if cl <= 256 { cl } else { 128 };
        let mut cap = DecodeCapture::new();
        cap.record_attn_grid(AttnGridWitness {
            is_ring_layer: true,
            slab_depth: 1280,
            window_size: 1024,
            ring_active: false,
            n_chunks: 6, // pre-window capture (KV tail ~700)
            chunk_size: 128,
            slid: false,
            min_tokens_kv: 0,
        });
        let plan = cap.into_plan();
        // Pre-window replay in the same n_chunks bucket is fine.
        assert!(plan.reusable_at(699, cs)); // 700/128 = 6
        // KV tail 1100: window(1024) < 1100 and not yet ring-wrapped (<=1280)
        // → SlideBand → not reusable.
        assert!(!plan.reusable_at(1099, cs));
    }

    #[test]
    fn pointer_slide_evaluates_window_start() {
        // window = 1024 → sub_tokens = 1023; stride = one KV cache row.
        // ptr(pos) = base + (pos + 1 - window) * stride.
        let base = 0x4000_0000_usize;
        let stride = 2048_usize;
        let mut cap = DecodeCapture::new();
        let k = cap.record_pointer_slide(base, stride, 1023);
        let plan = cap.into_plan();
        for (pos, offset_rows) in [(1024_usize, 1_usize), (1100, 77), (1279, 256)] {
            assert_eq!(
                plan.evaluate(pos, 7),
                vec![RebindValue::PointerU64 {
                    slot: k,
                    value: (base + offset_rows * stride) as u64,
                }]
            );
        }
    }

    #[test]
    fn slid_witness_reusable_across_band_only() {
        let cs = |cl: usize| if cl <= 256 { cl } else { 128 };
        let mut cap = DecodeCapture::new();
        cap.record_attn_grid(AttnGridWitness {
            is_ring_layer: true,
            slab_depth: 1280,
            window_size: 1024,
            ring_active: false,
            n_chunks: 8, // 1024/128, constant in-band
            chunk_size: 128,
            slid: true,
            min_tokens_kv: 0,
        });
        let plan = cap.into_plan();
        // In-band: window < KV tail <= slab_depth → positions [1024, 1279].
        assert!(plan.reusable_at(1024, cs));
        assert!(plan.reusable_at(1150, cs));
        assert!(plan.reusable_at(1279, cs));
        // Pre-window (KV tail <= window): the composite would not slide.
        assert!(!plan.reusable_at(1023, cs));
        assert!(!plan.reusable_at(500, cs));
        // Ring entry (KV tail > slab_depth): regime change → recapture.
        assert!(!plan.reusable_at(1280, cs));
        assert!(!plan.reusable_at(2000, cs));
    }

    #[test]
    fn uncapturable_keeps_first_reason() {
        let mut cap = DecodeCapture::new();
        cap.mark_uncapturable("single-pass baked n_tokens_kv");
        cap.mark_uncapturable("second reason ignored");
        assert_eq!(cap.is_uncapturable(), Some("single-pass baked n_tokens_kv"));
        assert_eq!(cap.uncapturable_retry(), Some(RetryAt::Never));
    }

    #[test]
    fn uncapturable_retry_escalates_to_latest() {
        let mut cap = DecodeCapture::new();
        cap.mark_uncapturable_until("slideband layer a", RetryAt::Position(1536));
        cap.mark_uncapturable_until("slideband layer b", RetryAt::Position(1280));
        assert_eq!(cap.is_uncapturable(), Some("slideband layer a"));
        assert_eq!(cap.uncapturable_retry(), Some(RetryAt::Position(1536)));
        cap.mark_uncapturable("q8 path");
        assert_eq!(cap.uncapturable_retry(), Some(RetryAt::Never));
    }
}
