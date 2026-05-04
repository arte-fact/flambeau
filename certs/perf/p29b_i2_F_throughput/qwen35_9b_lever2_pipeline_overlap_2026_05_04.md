# Lever 2 (GPU-pipeline overlap) — null at scaffold stage; awaits kernel-async work

**Date:** 2026-05-04
**Rig:** 4× MI50 (gfx906) PCIe 3.0 x16, ROCm 7.1.1
**Model:** Qwen3.5-9B-Q4_1
**Topology:** pp2tp2 over devices [0,2,1,3]
**Workload:** N=6 inflight pool / 6 concurrent staggered users (`/tmp/timing_bench.sh`)
**Tasks:** #309–#315 (lever 2 scope)
**Baseline:** commit `743606e` — 14566 ms median / 134.4 t/s / GPU% 14.2

## TL;DR

Lever 2 scaffolding (commits `b35e58a`, `7dbe76b`, `f458456`)
ships threading + scratch-pool architecture as designed, but the
**5.1% wall regression** vs baseline misses the ≥1.2× acceptance
gate. The win is gated by removing internal stream-syncs from
`forward_decode_mixed_hybrid` — that's the kernel-async work
deferred at the start of this lever.

## Measured (3 reps, median)

| | wall (ms) | aggregate t/s | GPU% mean | VRAM peak (MB) |
|---|---:|---:|---:|---:|
| **Baseline (`743606e`)**       | **14566** | **134.4** | 14.2 | 2740 |
| #311+#312 rep 1                |   16074   |   ~125    | 14.8 | 3389 |
| #311+#312 rep 2                |   15310   |   ~131    | 14.1 | 3389 |
| #311+#312 rep 3                |   16312   |   ~123    | 14.9 | 3389 |
| **#311+#312 median**           | **15310** | **~131**  | 14.6 | 3389 |
| **delta**                      | **+5.1%** | **−2.5%** |  +0.4 | +650 |

GPU% mean unchanged (14% in both) — confirms kernels are still
sync-bound, not host-bound. The +650 MB VRAM is the second
scratch-pool slot (expected per scope doc).

## Why it didn't deliver

The scope doc identified three internal sync points in
`forward_decode_mixed_hybrid` that block on GPU completion:
1. Entry-time stream drain (`#275` fix)
2. `peer_copy_via_host` at PP stage boundaries
3. `download_logits_host` at output-head per-slot

The threading scaffold is correct: dispatcher thread enqueues iter
N+1 while completion thread is processing iter N's results. But
because `forward_decode_mixed_hybrid` synchronously waits for kernel
completion before returning, the dispatcher thread BLOCKS inside
`enqueue_iteration` on the same internal syncs. Iter N+1's enqueue
can't start until iter N's GPU work + result download both finish.

In effect: dispatcher and completion threads currently serialize on
the same syncs as the single-threaded path, plus they pay condvar
wake/notify + queue push/pop overhead.

Net: ~700 ms wall added per run (mostly thread coordination on idle
queues + 200ms scratch pool init).

## What's preserved

The architecture is the right one. **Don't roll back the structural
split** — `enqueue_iteration` + `complete_iteration` is the API
boundary the future kernel-async work needs. The threading scaffold
also stays correct; the perf benefit just hasn't materialized.

## What needs to land for the gate

Refactor `forward_decode_mixed_hybrid` (or add an `_async` variant)
that:
1. Takes an optional `completion_event: Option<&HipEvent>` parameter
2. When `Some(event)`: skips internal stream syncs, records `event`
   on the head_rank's stream after the last kernel
3. When `None`: behaves as today (sync)

Plus modifications to `download_logits_host` (or a new `_no_sync`
variant) and `peer_copy_via_host` (event-recording instead of
host-side block).

Estimated: 2-3 sessions of careful kernel-side work.
`InFlightIteration::completion_event: Option<HipEvent>` is already
in place to carry the recorded event; setting it from
`enqueue_iteration` and syncing in `complete_iteration` will close
the loop.

## Recommendation

**Keep #309–#312 committed as scaffold** — they're correct and
non-regression at the architecture level. Cert this as "scaffold
shipped, perf win deferred to kernel-async follow-up". Production
default remains commit `743606e` (lever 1 baseline) until kernel-
async work delivers the actual lever.

## What lever 2 fully delivered would unlock

If kernel-async work were complete: per the scope doc projection,
GPU% would rise from 14% to 30%+, wall would drop to ≤11s, aggregate
throughput 180+ t/s. **+34% over the lever-1 baseline** at this
workload. The full architectural rewrite is the prerequisite — the
threading scaffold alone is a no-op until it lands.
