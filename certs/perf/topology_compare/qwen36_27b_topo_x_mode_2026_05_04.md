# Comparative bench: 3 topologies × 2 modes (no-batched vs batched)

**Date**: 2026-05-04
**Model**: Qwen3.6-27B-Q4_1
**Hardware**: 4× MI50 (gfx906), 16 GB each
**Build**: post-#295 (PP=4 hybrid hang fix) + post-CTX_CAP-PP-arm fix
**Knobs**: `FLAMBEAU_CTX_CAP=8192` (clamps GGUF default 128k → 8k for KV
budget); `FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_INFLIGHT_SLOTS=4` for
"batched" runs; nothing for "nobatch" runs.

## Headline data

| config            | idle MB | peak MB | gpu%_avg | gpu%_max | pp tok/s | tg tok/s | 4-conc wall |
|-------------------|--------:|--------:|---------:|---------:|---------:|---------:|------------:|
| pp4 / nobatch     |  36 020 |  36 021 |     25.3 |      100 |     18.8 |     20.2 |           — |
| pp4 / batched     |       — |       — |        — |        — |     18.9 |     20.3 |     25.04 s |
| tp2 / nobatch     |  20 147 |  20 355 |     49.3 |      100 |     35.1 |     26.0 |           — |
| tp2 / batched     |  22 269 |  23 105 |     46.6 |      100 |     35.0 |     26.0 |     18.08 s |
| pp2tp2 / nobatch  |  20 599 |  21 015 |     45.6 |      100 |     35.5 |     28.4 |           — |
| pp2tp2 / batched  |  22 731 |  24 403 |     44.9 |      100 |     35.3 |     28.8 |     16.17 s |

(pp4_batched VRAM/gpu% not in the original sampler trace — re-ran the
bench after the FLAMBEAU_CTX_CAP-PP-arm fix landed; values pending.)

## End-to-end speedup analysis

Per-request time derived from single-slot pp/tg + actual prompt size
(~70 tokens) + 64-token decode. 4 sequential time = 4 × per-request.

| topology | mode      | t_single  | 4 sequential | 4 concurrent | speedup  |
|----------|-----------|----------:|-------------:|-------------:|---------:|
| pp4      | nobatch   |   6.87 s  |     27.50 s  |          —   |       —  |
| pp4      | batched   |   6.86 s  |     27.43 s  |     25.04 s  | **1.10×**|
| tp2      | nobatch   |   4.46 s  |     17.85 s  |          —   |       —  |
| tp2      | batched   |   4.46 s  |     17.86 s  |     18.08 s  | **0.99×**|
| pp2tp2   | nobatch   |   4.23 s  |     16.93 s  |          —   |       —  |
| pp2tp2   | batched   |   4.20 s  |     16.81 s  |     16.17 s  | **1.04×**|

## Findings

### 1. Batched-decode delivers MARGINAL wall speedup at N=4

The batched-decode infrastructure (slot pool + scheduler + per-layer
batched kernels + #266c batched-attn + #287 batched-GDN) gives 0.99–
1.10× over 4-sequential at N=4 across all three topologies. Far below
the original 3× cert target.

- **tp2: 0.99× — slightly NEGATIVE.** Batching infrastructure overhead
  exceeds the GPU-pipelining benefit. Single-slot tg=26 t/s, 4-conc
  agg = 14.2 t/s → per-slot drops to 3.5 t/s under contention.
- **pp2tp2: 1.04× — break-even.** Same pattern, slightly positive.
- **pp4: 1.10× — modest positive.** PP=4's idle stages have headroom
  for batched-decode to fill.

### 2. VRAM overhead of batched-decode is small (+2–3 GB)

Comparing nobatch vs batched at the same topology:
- tp2: idle 20.1 → 22.3 GB (**+2.1 GB**), peak +2.7 GB
- pp2tp2: idle 20.6 → 22.7 GB (**+2.1 GB**), peak +3.4 GB

The +2 GB idle is the slot pool allocating 4 KV caches at SLOTS=4
(at FLAMBEAU_CTX_CAP=8192). The peak grows by another ~1 GB when the
prefill scratch is sized for n_tokens=4 (the batched-decode driver
uses `ShardedForwardPrefillScratchHybrid::new(model, max_slots)` for
its dispatch buffer).

### 3. Single-slot pp/tg ranking confirms prior topology certs

- **pp prefill**: tp2 / pp2tp2 ≈ 35 tok/s; pp4 = 18.8 tok/s.
  - PP-only has no parallel prefill (each token serially through
    n_stages → no batched matmul width). TP / hybrid get the batched
    prefill kernel which packs prompt tokens through wide matmuls.
- **tg decode**: pp2tp2 (28.8) > tp2 (26.0) > pp4 (20.3).
  - pp2tp2 wins single-slot decode despite +1 hop of cross-stage
    peer_copy: the per-stage TP=2 splits each FFN's matmul across 2
    GPUs which dominates decode wall.

### 4. GPU utilization tells the bottleneck story

- **pp4: 25 % mean GPU%.** Each of 4 stages idles 3/4 of the time
  (single-slot serial pipeline). Pure-PP needs N≥4 in flight with
  pipelining to fill — but at N=4 with current infrastructure, only
  marginal improvement (1.10×).
- **tp2 / pp2tp2: 45–49 % mean.** Both stages busy in parallel for
  TP-collective workloads. Headroom remains (peaks at 100% indicate
  bursty kernels but average is half).

### 5. Bug found + fixed: `FLAMBEAU_CTX_CAP` was ignored at MeshMode::Pp

The PP-mode loader (`crates/server/src/serve.rs:138-176`) didn't
re-apply the `FLAMBEAU_CTX_CAP` clamp like the TP / Hybrid arms do.
Fix: clamp `m.config.context_length` after `Qwen3MoEShardedModel::load`.
Without this, pp4 with SLOTS=4 OOMs because each layer's KV cache is
sized for the GGUF-embedded 128k ctx (~512 MB/layer/slot at TP=1).

## What this means for the 3× cert gate

The combined ceiling argued in part-3 / part-4 handoffs (1.05× attn ×
1.6× pipelining × 1.15× wave64 = 1.93×) is matching reality on
pp2tp2: actual 1.04× speedup. The pipelining ceiling isn't realized
because:

1. **Per-slot host-side launch overhead** dominates at N=4 — the
   scheduler's drain+dispatch + per-slot kernel-launch loop is
   competitive with raw GPU work at this batch size.
2. **Batched-decode kernels share the GPU** with each slot's KV-cache
   read/write (per-slot indexed K/V append). At N=4 the activation
   HBM traffic stays per-slot (no amortization), and the per-slot
   kernel launches still happen.

Path to >2×:
- Larger N (SLOTS=8 with smaller-quant model or smaller ctx).
- Pure-PP=4 pipelining with > 1.0× delivery (needs investigation —
  this session's #295 fix unblocked correctness but speedup is 1.0×).
- v3 wave64-decode kernel (FMA-symmetric — preserves bit-id-within-batch).

## Reproduce

```bash
# Bench script (3 topologies × 2 modes, ~12 min total):
/tmp/bench_topo.sh
# Parse:
python3 /tmp/parse_results.py
# Raw outputs in /tmp/bench_topo_results/<config>/{single_slot.json,
#   concurrent_4.json,smi_trace.jsonl,vram_idle.json,server.log}
```

## Diagnostic toggles

- `FLAMBEAU_CTX_CAP=<N>` — clamps model.context_length post-load.
  After this session's fix, applies to PP / TP / Hybrid uniformly.
- `FLAMBEAU_BATCHED_DECODE=1` + `FLAMBEAU_INFLIGHT_SLOTS=N` — enables
  the batched-decode dispatch path.
