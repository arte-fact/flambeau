# 27B mixed-batch A/B — heavier model didn't unlock the lever

**Date:** 2026-05-04
**Rig:** 4× MI50 (gfx906) PCIe 3.0 x16, ROCm 7.1.1
**Model:** Qwen3.6-27B-Q4_1 (qwen36moe — hybrid GDN+full-attn+MoE)
**Topology:** pp2tp2 over devices [0,2,1,3]
**Workload:** N=6 inflight pool / 6 concurrent staggered users (`/tmp/timing_bench.sh`)
**Task:** #317
**Code state:** lever-1 baseline (commit `743606e`); lever-2 rolled back

## TL;DR

Hypothesis from cycle-3 cert: heavier model means more GPU work per
iter, host coordination becomes a smaller fraction of wall, mixed-batch
ratio improves. **Falsified.** Mixed is ~2% slower than legacy on
27B too — same null verdict as 9B.

## Measured (3 reps, median)

| | wall median (ms) | aggregate t/s | GPU% mean | VRAM peak (MB) |
|---|---:|---:|---:|---:|
| **Legacy** (`FLAMBEAU_BATCHED_DECODE=1` only) | **35887** | **53.5** | 26.6 | 6960 |
| **Mixed**  (`+FLAMBEAU_MIXED_BATCH=1`)        |   36682   |   52.5   | 26.7 | 6246 |
| **delta**                                     |   +2.2%   |   −1.9%  |  ~0  | −10% |

Output coherent on both paths (octopus story, BST description, etc).
Both paths use the same kernels through `forward_decode_mixed_hybrid`
when chunks fire.

## What scaled with model size

| | 9B | 27B | ratio |
|---|---:|---:|---:|
| Per-call wall   | ~14.5s | ~36s   | ~2.5× |
| Aggregate t/s   | ~134   | ~53    | 0.40× |
| **GPU% mean**   | **14%**| **27%**| **2×** |
| VRAM peak       | 2.7GB  | 6.5GB  | ~2.4× |
| Host coord/iter | ~50ms  | ~50ms  | 1.0×  |
| GPU work/iter   | ~5ms   | ~13ms  | 2.6×  |

GPU% doubled as predicted — kernels do fill more of the wall on
27B. But mixed-vs-legacy ratio stayed flat because:

- Both paths run the same forward_decode_mixed_hybrid + same
  internal stream syncs.
- The lever's premise (iter N+1 enqueue overlapping iter N GPU work)
  never activates without an async peer_copy primitive and lockless
  output-head sync.
- Lever-2 rolled back per cert
  `qwen35_9b_lever2_pipeline_overlap_2026_05_04.md` — threading
  scaffold ships ~700 LOC of overhead without delivering overlap on
  this rig.

## Why GPU% 26% still isn't enough

Per-iter math at 27B / pp2tp2 / N=6:
- GPU work: ~13 ms / iter
- Host coord: ~50 ms / iter (peer_copy_via_host, mutex churn,
  channel ops — unchanged from 9B)
- Wall per iter ≈ max(host, GPU) = ~50 ms → still host-bound

To flip the ratio, GPU work would need to exceed 50 ms / iter:
- Even bigger model (~100B → ~50ms) — out of VRAM range on 4×16GB
- Higher N (~N=18) — would OOM the KV cache pool
- Larger context (ctx > 8k) — KV reads grow per token, helps but
  bench is fixed at ctx=2048

## Confirmed bottleneck: PCIe-only host coordination

`peer_copy_via_host` at PP stage hand-offs is ~1ms host-bounce per
transition. With pp2tp2 + 64 layers split across 2 stages, that's
1 transition per token × 6 slots = 6 host-bounces per iter ≈ 6ms.
Plus the mutex/channel coordination ≈ 50ms total per iter.

On an H100 + NVLink rig, peer-copy is async at full GPU speed and
event-chained — no host involvement. Mixed-batch's projected 1.5–
2.5× win lives there, not here.

## Verdict

**This rig is fundamentally host-coordination-bound** at every
(model, N) combo we can fit on 4×16GB cards. The lever-1 mixed-batch
baseline is the practical throughput ceiling without:

1. Async peer-copy primitive (would require xGMI/NVLink hardware)
2. Custom async dispatch + event chaining (architectural rewrite,
   not yet justified by perf data)

Production state: `743606e` lever-1 mixed-batch v1 stays the default.
134 t/s on 9B / 53 t/s on 27B at N=6 staggered.

## Methodology

```bash
# Both runs:
FLAMBEAU_BATCHED_DECODE=1 [+FLAMBEAU_MIXED_BATCH=1] \
  FLAMBEAU_INFLIGHT_SLOTS=6 FLAMBEAU_CTX_CAP=2048 \
  ./target/release/flambeau serve \
  --model /artefact/models/Qwen3.6-27B-Q4_1.gguf \
  --devices hip:0,2,1,3 --mesh-mode pp+tp \
  --pp-size 2 --tp-size 2 --port 8089

# Workload: 4 short-prompt long-decode + 2 staggered long-prompt
#   short-decode (timing_bench.sh). 3 reps, median.
```
