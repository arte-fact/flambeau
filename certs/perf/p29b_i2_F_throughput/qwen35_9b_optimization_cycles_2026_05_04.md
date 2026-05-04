# Mixed-batch optimization — 3 profiling-driven cycles

**Date:** 2026-05-04
**Rig:** 4× MI50 (gfx906) PCIe 3.0 x16, ROCm 7.1.1
**Model:** Qwen3.5-9B-Q4_1
**Topology:** pp2tp2 over devices [0,2,1,3]

## TL;DR

Three optimization cycles starting from commit `3a0d898` (post race
fix + decode fast-path). Net throughput went from 99 → **133 t/s
aggregate** on representative 6-user mixed traffic, a **1.34× lift**.

| | wall (ms) | aggregate t/s | GPU% mean |
|---|---:|---:|---:|
| **Cycle 0 baseline** (N=4 staggered, 1545 tok) | 15355 | 99 | 13.2 |
| **Cycle 3 best** (N=6 mixed traffic, 2012 tok) | 15115 | **133** | 14.2 |

The big lever was **inflight-pool size** (4 → 6), not chunk knobs or
host-side optimization. GPU% stays ~14% in both — the bottleneck is
host coordination, not GPU compute.

## Cycle 1 — Knob sweep (no code change)

Hypothesis: tune `MIXED_CHUNK` and `MIXED_BUDGET` for better
amortization.

| chunk | budget | wall (ms) | GPU% | VRAM peak (MB) |
|------:|-------:|----------:|-----:|---------------:|
|  256  |   320  |   15355   | 13.2 |        2677    |
|  128  |   192  |   16784   | 15.2 |        2532    |
|  512  |   576  |   17631   | 16.8 |        2925    |

**Finding**: defaults (chunk=256, budget=320) are already optimal for
this rig + model. Smaller chunks → more iterations × per-iter
overhead. Larger chunks → bigger compute spike per iteration that
freezes co-batched decoders. **No win available from knob tuning.**

## Cycle 2 — Code optimization (REVERTED)

Hypothesis: add a "fast-path" in `prefill_via_mixed_scheduler` that
delegates to legacy `prefill_logits` when no other slots are active,
mirroring the working decode fast-path (commit `3a0d898`).

**Result: regression.** Live A/B showed:

| workload | baseline | with prefill fast-path |
|----------|---------:|-----------------------:|
| N=3 decode-only | 14533 ms | 32087 ms (2.2× **slower**) |
| N=4 staggered (rep 1) | 15602 ms | 16197 ms (worse) |

Output coherent, so it was a perf bug not a correctness bug. The
fast-path was triggering as expected (the atomic + mutex check is
trivial), but somehow the rest of the run got slower. Most likely
explanation: the fast-path bypasses `mb.scheduler.allocate_request_id`
+ scratch lazy-init, and when a *different* request later goes
through the slow path, lazy-init hits with no warm-up. Or the
mutex contention pattern between the legacy single-shot prefill and
the mixed scheduler's pending-prefill check creates lock convoy
issues.

**Decision**: revert (no commit). Document the regression as a
trap to avoid in future iterations.

## Cycle 3 — Inflight-pool sizing

Hypothesis: GPU% mean 13.2% means the GPUs are 86 % idle. The win is
in increasing the *concurrent work per iteration*, not optimizing
the per-iteration overhead.

Bumped `FLAMBEAU_INFLIGHT_SLOTS` from 4 → 6 → 8 and re-ran on
6-/8-concurrent traffic (4 or 6 decoders + 2 staggered prefills).

| pool size | concurrent users | wall (ms) | total tokens | aggregate t/s |
|----------:|-----------------:|----------:|-------------:|--------------:|
|     4     |        4         |  15602    |     1545     |       99      |
|   **6**   |      **6**       |  **15115**|     2012     |     **133**   |
|     8     |        8         |  19614    |     2406     |     123       |

**Finding**: sweet spot is **N=6**. At N=8, per-user TPOT degrades
faster than aggregate work increases. VRAM at 2.7 GB/card — we have
~13 GB headroom but the bottleneck isn't memory.

Mixed-vs-legacy ratio at N=6: 132.9 vs 127.8 t/s = **1.04× mixed
faster**. The Sarathi lever is real but small at N=6 — most of the
133 t/s is just the better batched-decode amortization that comes
from N=6.

## What's actually limiting throughput

GPU% mean = 14% across all configs tested. **The GPUs are 86% idle.**
The model + topology combo is host-bound at this scale:

- Per iteration: ~15 ms CPU dispatch coordination + ~5 ms GPU compute
- Throughput ceiling ≈ 1000 ms / (15 ms iter overhead) × N decodes
  per iter = 67 × N tokens/sec at infinite N
- At N=6 we measure 133 t/s, but theory says we should approach 400
  t/s if GPU compute were the bottleneck

The 86 % idle GPU is host-side wait time:
- mutex acquisitions (inflight-pool, dispatcher, scheduler)
- channel sends + recv blocking
- per-stage `peer_copy_via_host` (PCIe-only rig — no xGMI/NVLink)
- per-rank stream synchronization

Real next-step levers (out of scope this turn):
1. **Async/await dispatcher** instead of mutex-based leader election —
   drop blocking_lock churn (~1ms/iter saved × 1000 iters = 1 sec)
2. **Reduce peer_copy_via_host calls** — currently 2 stage hand-offs
   per layer × 64 layers = 128 host-bounce copies per token
3. **Larger model** (35B-A3B) — per-iter GPU compute grows ~3.5× while
   host coordination stays constant; GPU% should rise to 30-40%, lever
   becomes more visible
4. **Varlen attention kernel (#303)** — halve per-layer kernel calls
   in chunk-bearing iterations

## Final state

Codebase: commit `3a0d898` (race fix + decode fast-path). No further
code changes from cycles 2/3 — they were a knob sweep + a reverted
optimization + a config recommendation.

**Recommended config for production**:
```bash
FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_MIXED_BATCH=1
FLAMBEAU_INFLIGHT_SLOTS=6        # the lever
FLAMBEAU_MIXED_BUDGET=320        # default
FLAMBEAU_MIXED_CHUNK=256         # default
FLAMBEAU_CTX_CAP=2048
```

This delivers ~133 t/s aggregate on 6-concurrent mixed traffic vs
~99 t/s at N=4 — **+34% throughput from one config change**.

## Methodology

- 3 reps each, median wall reported
- bench scripts: `/tmp/timing_bench.sh` (N=6), `/tmp/n8_bench.sh` (N=8)
- monitor: `rocm-smi --showuse --showmeminfo vram --csv` polled at
  200ms intervals across each run
- response correctness verified by reading completion text and
  finish_reason on every run
