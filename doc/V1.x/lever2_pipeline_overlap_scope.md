# Next session — Lever 2: GPU-pipeline overlap between dispatch iterations

**Status:** scoped, not implemented.
**Projected gain:** +20–40% aggregate throughput at N=6/Qwen3.5-9B/pp2tp2.
**Estimated effort:** 2–3 sessions.
**Prereqs:** commits up to `743606e` (lock-collapse lever 1).

## Problem statement

Current `dispatch_mixed_iteration` is fully synchronous per iteration:

```
iter N:  enqueue kernels  →  GPU runs (~5 ms)  →  CPU waits for completion
                                                  →  CPU sends results
iter N+1: enqueue kernels  →  GPU runs        →  CPU waits  →  send results
                              (waste — GPU idle while CPU was sending iter N
                               results & next iter was being assembled)
```

Live measurement on Qwen3.5-9B-Q4_1 / pp2tp2 / N=6:
- Per iteration wall: ~50 ms
- GPU% mean: **14%** (rocm-smi 200ms polls)
- Implied breakdown: ~5–7 ms GPU compute + ~43–45 ms host coordination
  + small-kernel launch latency between layers (200+ kernels/iter)

**The GPUs are idle ≥ 80% of wall time on this workload.** Closing
even half of that closes the difference between a 134 t/s aggregate
ceiling and ~250+ t/s.

## Target architecture

Decouple GPU enqueue from result delivery. Two independent threads of
control:

```
[Dispatch thread]  pulls plan from scheduler
                   enqueues kernels on stream
                   records a HipEvent at end
                   pushes (event, plan, output buffers) to "in-flight" queue
                   immediately picks up next plan ← KEY

[Completion thread] polls in-flight queue
                    waits on next event (with HipEvent::synchronize)
                    sends per-slot logits via channels
                    drops output buffers
                    consumes next entry
```

While the dispatch thread is enqueueing iteration N+1, iteration N's
kernels are still running on the GPU. The GPU never has to wait for
host-side result delivery.

This is the same pattern as vLLM's two-stage async engine loop
(RFC #11945), adapted to flambeau's rust + std-mpsc world.

## Specific changes

### 1. New: `MixedDispatcher` long-lived thread pair

```rust
// in MixedBatchCtx, add:
pub dispatcher_thread: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
pub completion_thread: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
pub in_flight: std::sync::Mutex<VecDeque<InFlightIteration>>,
pub wake: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,

struct InFlightIteration {
    completion_event: HipEvent,  // recorded after last kernel
    plan: MixedIterationPlan,
    decode_logits: Vec<Vec<f32>>,
    prefill_final: Vec<f32>,
    pool_to_vec: Vec<Option<usize>>,  // small fixed-size remap
}
```

Spawn both threads from `serve.rs` after the model loads (lazy-init
on first mixed dispatch is also fine; dispatcher needs an
`Arc<ServerState>`).

### 2. Dispatch thread loop

```rust
loop {
    let plan = scheduler.lock().next_iteration();
    match plan {
        None => wait_on_wake(),  // condvar
        Some(plan) => {
            // 1. Lock referenced inflight slots (same as today).
            // 2. Lock scratch.
            // 3. Enqueue kernels via forward_decode_mixed_hybrid.
            //    DO NOT synchronize streams.
            // 4. Record HipEvent on the last-touched stream.
            // 5. Push InFlightIteration to in_flight queue.
            // 6. notify completion thread.
            // 7. RELEASE scratch + slot guards (carefully — see #5).
        }
    }
}
```

### 3. Completion thread loop

```rust
loop {
    let entry = in_flight.lock().pop_front();
    match entry {
        None => wait_on_wake(),
        Some(it) => {
            it.completion_event.synchronize();  // blocks until GPU done
            // Now safe to read decode_logits / prefill_final.
            // Send results to per-request channels (batched, as today).
            // Drop entry → drops scratch refs / output buffers.
        }
    }
}
```

### 4. Submitters (prefill_via_mixed_scheduler / decode_via_mixed_scheduler)

Unchanged — submit + wait on rx. The wake signal goes to the
dispatch thread.

### 5. **The critical correctness invariant: scratch lifetime**

Today's dispatch holds `mb.scratch.lock()` during `forward_decode_mixed_hybrid`,
and the scratch's device buffers are written by kernels enqueued
inside. Today's *synchronous* dispatch waits for kernels to complete
before dropping the scratch lock — so reuse on the next iteration is
safe.

In the async architecture, **iteration N+1 wants to start using the
scratch BEFORE iteration N's kernels finish.** Two options:

**Option A** — pool of scratches (e.g., 2 or 3, double-buffered).
Each iteration takes one from the pool, returns it after the
completion event fires. Cost: 2-3× scratch VRAM (today: ~1 GB at
budget=512). Acceptable on 16 GB cards.

**Option B** — single scratch, but the dispatch thread waits on the
*previous* iteration's event before enqueueing into the same scratch.
This serializes scratch use but still lets host enqueue overlap with
GPU execution of OLD iterations. Less parallelism than A; same VRAM.

**Recommended: A**. Pre-allocate `[ScratchSlot; 2]` at server boot.
The cost is modest VRAM (extra ~600 MB at budget=512), the simpler
correctness story is worth it.

### 6. Inflight slot mutex lifetime

Today's dispatch holds the slot mutex from "lock all sessions" until
"send results", entire iter. In async, the kernels write to the KV
caches inside the slot's `HybridSession`. The slot mutex must be
held until the kernels actually FINISH (not just enqueue).

→ Inflight guards must be carried into `InFlightIteration` and
released only on the completion thread after `event.synchronize()`.

This means N+1 dispatch can't start until iter N's slot guards are
RELEASED. Alternative: per-session per-iteration completion flags or
event-gated locking. Simpler is to just hold N≤2 iterations'
worth of guards in flight (max 2 × inflight_pool.len() guards
held).

Actually even simpler with **option A scratch pool**: limit
in_flight depth to N_scratch (e.g., 2). Once 2 iterations are in
flight, dispatch waits for the oldest to complete before starting
the third. Slot guards count is bounded. Backpressure is natural.

### 7. Race fix carries over

The leader-release race fixed in `3a0d898` doesn't apply because
the dispatch thread is always alive — there's no "leader handoff."
Submitters always use the wake/notify pattern.

## Validation steps

### Correctness — must pass before any perf claim

1. **Existing parity tests still green**:
   - `mixed_batch_parity_pp2tp2_k32_n1`
   - `mixed_batch_parity_pp2tp2_k32_n2`
   - `mixed_batch_parity_pp2tp2_k128_n4`
   - `mixed_scheduler_drives_multi_chunk_pp2tp2`

   Async architecture must produce bit-identical (or tolerance-equal)
   outputs vs the synchronous path.

2. **New stress test**: 8 concurrent users / 6-inflight-pool / 30 sec
   sustained traffic. Verify:
   - All requests return coherent text
   - No deadlock / hang
   - No GPU error (HIP error 2/700)
   - VRAM steady (no leak)

3. **Backpressure test**: submit 12 prefills + 6 decodes simultaneously.
   With in_flight depth = 2, dispatcher should naturally backpressure
   — submitters block on rx until oldest completes.

### Performance — once correctness gates green

A/B against `743606e` baseline on the same workload
(`/tmp/timing_bench.sh` with N=6 inflight + 6 concurrent).

Target: **180+ t/s aggregate** (1.34× over the current 134 t/s
ceiling). Non-negotiable acceptance:
- GPU% mean ≥ 30 (vs current 14)
- Wall ≤ 11 sec (vs current 14.6)
- Output coherent

If the bench shows < 1.2× improvement, roll back — the engineering
cost isn't justified.

## Risks + mitigations

| risk | likelihood | mitigation |
|---|---|---|
| Kernel double-write to scratch (race) | high if not careful | Option A (pooled scratch) + per-pool-slot ownership tracking |
| Slot KV cache concurrent reads/writes between iterations | medium | Hold inflight guard until completion; max 2 iters in-flight |
| Channel send before kernel actually wrote logits | high if missed | event.synchronize() before any logits read or send |
| HipEvent leak / not recycled | low | RAII on InFlightIteration drop |
| Deadlock if completion thread blocks on a mutex held by submitter | medium | Submitters never hold mutex while waiting on rx; completion thread drops guards before sending channels |

## Where to start

1. Plumb `Arc<ServerState>` (or weak ref) into MixedBatchCtx so
   threads can dispatch.
2. Implement the dispatcher + completion threads as a `MixedThreadPair`
   that owns the wake condvar + in_flight queue.
3. Refactor `dispatch_mixed_iteration` into:
   - `enqueue_iteration(plan, scratch_slot) -> InFlightIteration` (no sync)
   - `complete_iteration(InFlightIteration)` (sync + send results)
4. Add scratch pool (size 2) to `MixedBatchCtx`.
5. Update `try_drive_mixed_dispatch` → just `notify_one()` on wake
   condvar. Remove the leader try-lock pattern.
6. Run parity tests. Iterate until clean.
7. Run perf bench. Compare to baseline.

## References

- Current dispatch: `crates/server/src/routes.rs::dispatch_mixed_iteration`
- Current bench: `/tmp/timing_bench.sh`, `/tmp/n4_bench.sh`
- vLLM v1 RFC #11945 — same pattern
- Cycle-3 cert: `certs/perf/p29b_i2_F_throughput/qwen35_9b_optimization_cycles_2026_05_04.md`
- Lever-1 commit: `743606e`
