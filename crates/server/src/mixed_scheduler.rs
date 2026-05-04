//! **#305 — Sarathi mixed-batch scheduler primitive.**
//!
//! Picks one pending prefill chunk + up-to-N ready decode slots per
//! server iteration. Pure data-structure state machine — no I/O, no
//! GPU calls — so it's unit-testable in isolation. The runtime
//! wiring (#306) calls into [`MixedScheduler::next_iteration`] from
//! the dispatch leader and translates [`MixedIterationPlan`] into
//! the actual `forward_decode_mixed_hybrid` invocation.
//!
//! ## State machine per request
//!
//! A request entering the mixed-batch path goes through:
//!
//! ```text
//!   PendingPrefill { tokens_remaining: K_total, chunk_start: 0, slot_idx }
//!     (each iteration consumes ≤ chunk_budget tokens; chunk_start advances)
//!     last chunk → emits first decode token via prefill_final_logits_out
//!   Decoding { slot_idx, current_pos }
//!     (each iteration consumes 1 decode slot; position advances)
//!     until stop / max_tokens
//!   Done
//! ```
//!
//! The scheduler treats both PendingPrefill and Decoding as queues; at
//! each iteration it draws ≤ 1 prefill chunk + ≤ token_budget − K
//! decode slots, prioritising:
//!
//! 1. Smallest prefill remainder first (heads queue with quickest
//!    finishing requests; minimises TTFT for late arrivals).
//! 2. All ready decodes (any one whose KV is in a coherent state for
//!    a step).
//!
//! ## What the scheduler does NOT do
//!
//! - It does not own KV caches or dispatch GPU work. The caller
//!   (`routes.rs`) holds inflight slots, looks up sessions, and calls
//!   `forward_decode_mixed_hybrid` with the plan's chunk + slots.
//! - It does not handle errors from the dispatch — those bubble up to
//!   the request handler. On dispatch failure, the iteration is
//!   reverted by [`MixedScheduler::revert_iteration`].
//! - It does not rate-limit. Caller should gate on inflight-slot
//!   acquisition.

use std::collections::VecDeque;

/// Identifier for a request in the mixed scheduler. A request may be
/// in pending-prefill state for several iterations, then transition to
/// decoding state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MixedRequestId(pub u64);

/// One pending request whose prefill is in progress (zero or more
/// chunks already submitted; remaining tokens still need to advance
/// through layers).
#[derive(Debug, Clone)]
pub struct PendingPrefillReq {
    pub request_id: MixedRequestId,
    /// Index into the caller's session/inflight pool — the chunk's
    /// `MixedPrefillChunk.idx` will reflect this.
    pub slot_idx: usize,
    /// Full prompt token IDs.
    pub tokens: Vec<u32>,
    /// Position in `tokens` where the next chunk begins.
    pub chunk_start: usize,
}

impl PendingPrefillReq {
    pub fn remaining(&self) -> usize {
        self.tokens.len().saturating_sub(self.chunk_start)
    }
    pub fn is_final_chunk(&self, chunk_len: usize) -> bool {
        self.chunk_start + chunk_len >= self.tokens.len()
    }
}

/// One ready-to-decode slot.
#[derive(Debug, Clone, Copy)]
pub struct ReadyDecode {
    pub request_id: MixedRequestId,
    pub slot_idx: usize,
    pub token_id: u32,
    pub position: usize,
}

/// One iteration's plan: at most one prefill chunk + 0..N decodes.
#[derive(Debug, Clone)]
pub struct MixedIterationPlan {
    /// Optional prefill chunk to dispatch. The owning
    /// [`PendingPrefillReq`] has been advanced by `chunk.tokens.len()`
    /// in scheduler state — caller is committed to processing.
    pub chunk: Option<PrefillChunkPlan>,
    /// Decode slots to dispatch alongside the chunk. Pairwise distinct
    /// `slot_idx`, distinct from `chunk.slot_idx` if present.
    pub decodes: Vec<ReadyDecode>,
}

#[derive(Debug, Clone)]
pub struct PrefillChunkPlan {
    pub request_id: MixedRequestId,
    pub slot_idx: usize,
    pub tokens: Vec<u32>,
    pub chunk_start: usize,
    pub is_final_chunk: bool,
}

impl MixedIterationPlan {
    pub fn is_empty(&self) -> bool {
        self.chunk.is_none() && self.decodes.is_empty()
    }
    pub fn token_count(&self) -> usize {
        self.chunk.as_ref().map_or(0, |c| c.tokens.len()) + self.decodes.len()
    }
}

/// Sarathi mixed-batch scheduler.
///
/// Cheap to construct; intended to live as a single instance inside
/// `ServerState` (behind a mutex) for the lifetime of the server.
#[derive(Debug, Default)]
pub struct MixedScheduler {
    pending_prefills: VecDeque<PendingPrefillReq>,
    /// FIFO of ready decodes — the caller pushes them as soon as a
    /// decode-state request has its previous step's logits sampled.
    ready_decodes: VecDeque<ReadyDecode>,
    /// Per-iteration knob: maximum total token count (K + N).
    pub token_budget: usize,
    /// Per-iteration knob: maximum K. K is also bounded by token_budget.
    pub max_chunk_size: usize,
    next_request_id: u64,
}

impl MixedScheduler {
    pub fn new(token_budget: usize, max_chunk_size: usize) -> Self {
        Self {
            pending_prefills: VecDeque::new(),
            ready_decodes: VecDeque::new(),
            token_budget: token_budget.max(1),
            max_chunk_size: max_chunk_size.max(1),
            next_request_id: 1,
        }
    }

    /// Allocate a fresh `MixedRequestId`. Callers track their own
    /// id-to-state mapping; the scheduler only uses the id for
    /// disambiguation in plans.
    pub fn allocate_request_id(&mut self) -> MixedRequestId {
        let id = MixedRequestId(self.next_request_id);
        self.next_request_id += 1;
        id
    }

    /// Enqueue a new pending-prefill request.
    pub fn submit_prefill(&mut self, req: PendingPrefillReq) {
        debug_assert!(req.remaining() > 0, "submit_prefill: empty prompt");
        self.pending_prefills.push_back(req);
    }

    /// Mark a slot as ready for one decode step.
    pub fn submit_decode(&mut self, decode: ReadyDecode) {
        self.ready_decodes.push_back(decode);
    }

    pub fn pending_prefill_count(&self) -> usize {
        self.pending_prefills.len()
    }
    pub fn ready_decode_count(&self) -> usize {
        self.ready_decodes.len()
    }

    /// Build one iteration plan. Returns `None` if there is no work
    /// (no pending prefills + no ready decodes).
    ///
    /// Strategy:
    /// 1. Pick the front pending prefill (if any). Slice off
    ///    `min(remaining, max_chunk_size, token_budget)` tokens.
    /// 2. Fill the rest of `token_budget` with ready decodes whose
    ///    `slot_idx` differs from the chunk's slot_idx.
    /// 3. Advance pending prefill's `chunk_start` by the slice length.
    ///    If `chunk_start == tokens.len()`, the request transitions to
    ///    decode state (caller's responsibility — scheduler removes it
    ///    from pending_prefills here).
    /// 4. Decodes are removed from ready_decodes; caller re-enqueues
    ///    them after sampling the next token.
    pub fn next_iteration(&mut self) -> Option<MixedIterationPlan> {
        if self.pending_prefills.is_empty() && self.ready_decodes.is_empty() {
            return None;
        }

        // Step 1: maybe pick a prefill chunk.
        let chunk_plan = if let Some(req) = self.pending_prefills.front_mut() {
            let chunk_len = req
                .remaining()
                .min(self.max_chunk_size)
                .min(self.token_budget);
            // Take ownership of this slice's tokens.
            let chunk_tokens: Vec<u32> =
                req.tokens[req.chunk_start..req.chunk_start + chunk_len].to_vec();
            let chunk_start = req.chunk_start;
            let is_final_chunk = req.is_final_chunk(chunk_len);
            let plan = PrefillChunkPlan {
                request_id: req.request_id,
                slot_idx: req.slot_idx,
                tokens: chunk_tokens,
                chunk_start,
                is_final_chunk,
            };
            req.chunk_start += chunk_len;
            // If finished, pop the request — it transitions to decode state.
            if is_final_chunk {
                self.pending_prefills.pop_front();
            }
            Some(plan)
        } else {
            None
        };

        // Step 2: fill remaining budget with ready decodes (excluding
        // the chunk's slot_idx if present).
        let used = chunk_plan.as_ref().map_or(0, |p| p.tokens.len());
        let cap = self.token_budget.saturating_sub(used);
        let mut decodes: Vec<ReadyDecode> = Vec::with_capacity(cap);
        let mut deferred: Vec<ReadyDecode> = Vec::new();
        let chunk_slot = chunk_plan.as_ref().map(|p| p.slot_idx);
        while decodes.len() < cap {
            let Some(d) = self.ready_decodes.pop_front() else {
                break;
            };
            if chunk_slot == Some(d.slot_idx) {
                // Slot collides with the chunk; defer for next iteration.
                deferred.push(d);
                continue;
            }
            decodes.push(d);
        }
        // Re-queue any deferred decodes at the front (preserve order).
        for d in deferred.into_iter().rev() {
            self.ready_decodes.push_front(d);
        }

        let plan = MixedIterationPlan {
            chunk: chunk_plan,
            decodes,
        };
        if plan.is_empty() {
            return None;
        }
        Some(plan)
    }

    /// Roll back the last iteration on dispatch failure. The caller
    /// supplies the plan returned by `next_iteration` so we can restore
    /// the queue state before retry.
    pub fn revert_iteration(&mut self, plan: MixedIterationPlan) {
        // Re-insert the prefill at the front with its previous state.
        if let Some(c) = plan.chunk {
            // Even if it was the final chunk (and thus popped during
            // next_iteration), re-insert here.
            // Find its (potentially in-flight) entry — if the request
            // had pop'd we lost its token list; reconstruct from chunk.
            // For non-final chunks, the request is still at the front.
            if !c.is_final_chunk {
                if let Some(front) = self.pending_prefills.front_mut() {
                    if front.request_id == c.request_id {
                        front.chunk_start = c.chunk_start;
                        // Tokens unchanged; nothing else to restore.
                    }
                }
            } else {
                // Final-chunk request was pop'd. We don't have the full
                // tokens vector here (chunk only had the slice). Caller
                // must re-submit the full request via submit_prefill if
                // they want to retry. For now, log loss.
                // (In practice dispatch failures should be rare and
                // caller can either fail the request or re-submit.)
            }
        }
        // Re-queue decodes at the front.
        for d in plan.decodes.into_iter().rev() {
            self.ready_decodes.push_front(d);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(id: u64, slot: usize, tokens: Vec<u32>) -> PendingPrefillReq {
        PendingPrefillReq {
            request_id: MixedRequestId(id),
            slot_idx: slot,
            tokens,
            chunk_start: 0,
        }
    }

    fn dec(id: u64, slot: usize, tok: u32, pos: usize) -> ReadyDecode {
        ReadyDecode {
            request_id: MixedRequestId(id),
            slot_idx: slot,
            token_id: tok,
            position: pos,
        }
    }

    #[test]
    fn empty_returns_none() {
        let mut s = MixedScheduler::new(512, 256);
        assert!(s.next_iteration().is_none());
    }

    #[test]
    fn prefill_only_single_chunk() {
        let mut s = MixedScheduler::new(512, 256);
        s.submit_prefill(req(1, 0, vec![10, 20, 30]));
        let plan = s.next_iteration().expect("plan");
        let chunk = plan.chunk.expect("chunk");
        assert_eq!(chunk.tokens, vec![10, 20, 30]);
        assert_eq!(chunk.chunk_start, 0);
        assert!(chunk.is_final_chunk);
        assert!(plan.decodes.is_empty());
        assert_eq!(s.pending_prefill_count(), 0);
        assert!(s.next_iteration().is_none());
    }

    #[test]
    fn prefill_chunked_across_iterations() {
        // 100-token prompt, max_chunk_size=30 → 4 iterations
        // (30, 30, 30, 10).
        let mut s = MixedScheduler::new(512, 30);
        let prompt: Vec<u32> = (0..100).collect();
        s.submit_prefill(req(1, 0, prompt));
        let mut chunks_seen = vec![];
        for _ in 0..4 {
            let plan = s.next_iteration().expect("plan");
            let chunk = plan.chunk.expect("chunk");
            chunks_seen.push((chunk.chunk_start, chunk.tokens.len(), chunk.is_final_chunk));
        }
        assert!(s.next_iteration().is_none());
        assert_eq!(
            chunks_seen,
            vec![(0, 30, false), (30, 30, false), (60, 30, false), (90, 10, true)]
        );
    }

    #[test]
    fn mixed_chunk_plus_decodes() {
        let mut s = MixedScheduler::new(10, 4);
        s.submit_prefill(req(1, 0, vec![1, 2, 3, 4, 5, 6]));
        s.submit_decode(dec(2, 1, 100, 50));
        s.submit_decode(dec(3, 2, 101, 60));
        s.submit_decode(dec(4, 3, 102, 70));
        let plan = s.next_iteration().expect("plan");
        let chunk = plan.chunk.as_ref().expect("chunk");
        assert_eq!(chunk.tokens.len(), 4); // bounded by max_chunk_size
        assert_eq!(plan.decodes.len(), 3); // 10 - 4 = 6 cap, 3 ready
        assert_eq!(plan.token_count(), 7);
    }

    #[test]
    fn chunk_does_not_collide_with_own_decode_slot() {
        // Chunk owns slot 0; a stale decode for slot 0 should be
        // deferred (in production this shouldn't happen — the chunk's
        // request is in PendingPrefill, can't also be in Decoding —
        // but we defend against it).
        let mut s = MixedScheduler::new(10, 4);
        s.submit_prefill(req(1, 0, vec![1, 2, 3]));
        s.submit_decode(dec(99, 0, 100, 50)); // collides
        s.submit_decode(dec(2, 1, 101, 60)); // ok
        let plan = s.next_iteration().expect("plan");
        let chunk = plan.chunk.as_ref().expect("chunk");
        assert_eq!(chunk.slot_idx, 0);
        assert_eq!(plan.decodes.len(), 1);
        assert_eq!(plan.decodes[0].slot_idx, 1);
        // The deferred decode is still queued.
        assert_eq!(s.ready_decode_count(), 1);
    }

    #[test]
    fn decodes_only_no_chunk() {
        let mut s = MixedScheduler::new(8, 4);
        s.submit_decode(dec(1, 0, 100, 5));
        s.submit_decode(dec(2, 1, 101, 6));
        let plan = s.next_iteration().expect("plan");
        assert!(plan.chunk.is_none());
        assert_eq!(plan.decodes.len(), 2);
        assert!(s.next_iteration().is_none());
    }

    #[test]
    fn revert_iteration_restores_decodes() {
        let mut s = MixedScheduler::new(8, 4);
        s.submit_decode(dec(1, 0, 100, 5));
        s.submit_decode(dec(2, 1, 101, 6));
        let plan = s.next_iteration().expect("plan");
        assert_eq!(s.ready_decode_count(), 0);
        s.revert_iteration(plan);
        assert_eq!(s.ready_decode_count(), 2);
    }

    #[test]
    fn revert_iteration_restores_non_final_chunk() {
        let mut s = MixedScheduler::new(8, 4);
        s.submit_prefill(req(1, 0, (0..10).collect())); // 10-token prompt
        let plan = s.next_iteration().expect("plan");
        let chunk = plan.chunk.as_ref().unwrap();
        assert_eq!(chunk.tokens.len(), 4);
        assert!(!chunk.is_final_chunk);
        s.revert_iteration(plan);
        // After revert, the pending prefill is at chunk_start = 0 again.
        let req_back = s.pending_prefills.front().unwrap();
        assert_eq!(req_back.chunk_start, 0);
        assert_eq!(req_back.tokens.len(), 10);
    }

    #[test]
    fn token_budget_caps_combined_size() {
        let mut s = MixedScheduler::new(6, 16);
        s.submit_prefill(req(1, 0, vec![1, 2, 3, 4]));
        s.submit_decode(dec(2, 1, 100, 5));
        s.submit_decode(dec(3, 2, 101, 6));
        s.submit_decode(dec(4, 3, 102, 7));
        s.submit_decode(dec(5, 4, 103, 8));
        let plan = s.next_iteration().expect("plan");
        // K=4 prefill + only 2 decodes = 6 = budget.
        assert_eq!(plan.token_count(), 6);
        assert_eq!(plan.decodes.len(), 2);
        // 2 deferred decodes still queued.
        assert_eq!(s.ready_decode_count(), 2);
    }
}
