# Sarathi mixed-batch v1 — production wiring delivers 1.115× on staggered traffic (post profiling fixes)

**Date:** 2026-05-04
**Rig:** 4× MI50 (gfx906) PCIe 3.0 x16, ROCm 7.1.1
**Model:** Qwen3.5-9B-Q4_1 (qwen35 — hybrid GDN+full-attn+MoE)
**Topology:** pp2tp2 over devices [0,2,1,3]
**Task #:** #306 + #308 production cert
**Commits:** `73c9e06` (initial wiring), `3a0d898` (perf fixes)

## TL;DR

After profiling identified two production-blocking issues, the v1
mixed-batch dispatch path **wins by 10.4% wall** on the staggered
4-concurrent-user workload (the actual Sarathi lever).

| workload                  | legacy median | mixed median | ratio  |
|---------------------------|--------------:|-------------:|-------:|
| N=3 decode-only (no chunks) |       13056  |       14534  | 0.90×  |
| **N=4 staggered (lever)** |    **16944**  |   **15190**  | **1.115×** |

3 reps each, taking median of 3.

## Initial null result + diagnosis

First production A/B (commit `73c9e06`) showed mixed **slower** than
legacy across every workload tested:

| workload                            | legacy ms | mixed v0 ms | ratio |
|-------------------------------------|----------:|------------:|------:|
| 2 conc K=487/N=2 simultaneous       |    10763  |      10808  | 1.00× |
| 2 conc staggered (r2 mid-r1-decode) |     7085  |       8104  | 0.87× |
| 4 conc 3 decoders + 1 long-prefill  |    16903  |      22876  | 0.74× |
| 4 conc, mixed_chunk=64              |    16903  |      26608  | 0.64× |
| **N=3 pure decode (no chunks)**     |    13056  |      21372  | **0.61×** |

The decode-only run showed mixed was 64% slower **without any chunks
to dispatch** — the overhead was intrinsic to the dispatch path, not
the chunk processing. Profiling pointed to two distinct issues:

### Issue 1: leader-release race (legacy #276 pattern)

`try_drive_mixed_dispatch` exited the inner loop after
`next_iteration()` returned None, dropping the scheduler-lock guard
*before* the dispatcher-lock guard. A submitter landing in that
window observed (a) scheduler lock free, (b) dispatcher lock still
held → tried to be leader, got Err, fell through to wait on its rx.
Then the original leader released the dispatcher; the submitter's
queued entry was stranded until *some other* thread submitted next
and picked it up.

**Symptom**: one specific slot lagged the others by ~7 seconds in
N=4 staggered tests. The slot's decode loop kept missing the leader
window.

**Fix** (commit `3a0d898`, mirrors legacy #276):

```rust
let plan = s.next_iteration();
match plan {
    None => {
        drop(dispatcher_guard_opt.take());  // release WHILE holding s
        drop(s);
        break;
    }
    Some(plan) => {
        drop(s);
        if let Err(e) = self.dispatch_mixed_iteration(&plan) { ... }
    }
}
```

After race fix: N=4 staggered improved 22876 → 20879 ms (**+9%**).
Still slower than legacy (16903), but less catastrophically.

### Issue 2: decode-only path overhead

Live A/B on **N=3 pure decode** (no chunks) showed mixed 21372 ms
vs legacy 13056 ms = 64% slower. The kernel work is identical
(mixed delegates to `forward_decode_batched_hybrid` when chunk =
None), so the gap is host-side overhead per iteration:

- channel-map locks (insert at submit, remove at dispatch)
- scratch double-lock (lazy-init then use)
- HashMap pool→vec remap allocation
- intermediate Vec allocations (slot_indices, decode_logits, logits_refs)

Per iteration this added ~27 ms over legacy's `dispatch_batched_pending`.
Across ~300 iterations × 3 decoders = 8 sec extra wall.

**Fix** (commit `3a0d898`): when `pending_prefill_count() == 0`,
`decode_via_mixed_scheduler` delegates to `decode_via_scheduler_into`
(legacy path). The mixed path engages only when there's a chunk to
co-batch.

After both fixes:
- N=3 decode-only: 14534 ms (0.90× legacy — 10% remaining overhead
  in the prefill entry path)
- N=4 staggered: 15190 ms median (**1.115× legacy** — wins)

## Why staggered N=4 wins

Staggered N=4 = 3 ongoing decoders + 1 long-prompt arrival mid-flight
(r4 enters at t=2s with 487-token prompt + 64-token decode).

In legacy: r4's `forward_prefill_hybrid_logits` blocks the GPU for
~500 ms of prefill compute, freezing the 3 decoders. Wall time
includes this stall.

In mixed: r4's prefill is chunked into K=256 + K=231 = 2 chunks.
Each chunk co-batches with the 3 decoders' next step. The 3 decoders
keep advancing (slowly) during r4's prefill. Net win: the 500 ms
freeze becomes ~600 ms of mixed iterations during which decoders
also progress.

Aggregate throughput: total tokens 1545 / wall ≈ 102 t/s (mixed) vs
91 t/s (legacy) — **+12% aggregate t/s**.

## Knobs

```
FLAMBEAU_BATCHED_DECODE=1   # required (gates the scheduler path)
FLAMBEAU_MIXED_BATCH=1      # opt in to mixed dispatch
FLAMBEAU_MIXED_BUDGET=320   # K + N cap per iteration (default 512)
FLAMBEAU_MIXED_CHUNK=256    # max prefill chunk size  (default 256)
FLAMBEAU_MIXED_WINDOW_US=1500  # batching window      (default 1500µs)
FLAMBEAU_INFLIGHT_SLOTS=4   # multi-slot pool
```

## Recommendation

Mixed-batch v1 ships **opt-in via `FLAMBEAU_MIXED_BATCH=1`**. On
realistic mixed-traffic workloads (concurrent decoders + occasional
long-prompt arrivals) it delivers **+10–12%** wall throughput over
legacy on this rig. Default remains legacy until the lever is
profile-validated on more model + topology combinations.

## What remains

- The 10% remaining overhead on decode-only is the
  `prefill_via_mixed_scheduler` path (still goes through mixed even
  when no other chunks pending). Could fast-path that too.
- The `forward_decode_mixed_hybrid` driver makes 2 attn + 2 GDN calls
  per layer (prefill + decode regions). #303 (varlen-attn kernel)
  would halve this and likely deliver another +5–10%.
- Higher N (e.g. 8 concurrent decoders) should give bigger wins —
  the microbench at N=16 showed 1.174× per-call. Untested at server
  level due to N=4 inflight pool default.

## Methodology

```bash
FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_MIXED_BATCH=1 \
  FLAMBEAU_MIXED_BUDGET=320 FLAMBEAU_MIXED_CHUNK=256 \
  FLAMBEAU_INFLIGHT_SLOTS=4 FLAMBEAU_CTX_CAP=2048 \
  ./target/release/flambeau serve \
  --model /artefact/models/Qwen3.5-9B-Q4_1.gguf \
  --devices hip:0,2,1,3 --mesh-mode pp+tp \
  --pp-size 2 --tp-size 2 --port 8089

# Workload: 3 concurrent decode-heavy requests (300 tokens each)
# starting at t=0; one long-prompt request (487 prompt + 64 decode)
# starting at t=2s. Total tokens processed: 1545.
```

Bench script: `/tmp/n4_bench.sh`. 3 reps; report median wall.
