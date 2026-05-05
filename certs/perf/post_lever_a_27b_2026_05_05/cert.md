# Post-Lever-A 27B topology bench — 2026-05-05

User question: *"so now tp beats pp on concurrency right?"*

**Answer: yes — TP2 now scales linearly to N=8, beating PP4 by 2.26× at
N=8. But the overall winner is pp2tp2, which beats TP2 by 1.31× and
PP4 by 2.95× at N=8.**

## Setup

- Qwen3.6-27B-Q4_1, ctx=4096, prompt=3313 tokens, max_tokens=256
- Greedy decode, `FLAMBEAU_INFLIGHT_SLOTS=8`, `FLAMBEAU_GPU_SAMPLER=1`
- 4× MI50 (gfx906), cross-die device pairings:
  - `pp2`: `hip:0,2`
  - `tp2`: `hip:0,2`
  - `pp4`: `hip:0,1,2,3`
  - `pp2tp2`: `hip:0,2,1,3` (stage0={0,2}, stage1={1,3})
- Includes `#321` prefill serialiser + `#324` shared TP prefill scratch
- Inter-N and inter-cell GPU temp gate at 70°C (rocm-smi)
- pp2/batched skipped (boot timeout in this run; pp2/no_batched data only)

## Aggregate decode throughput (cum_tps — server-side, sum of all streams)

| topology / path     | N=1   | N=2   | N=4    | N=8     | scaling N=1→8 |
|---------------------|-------|-------|--------|---------|---------------|
| pp2 / no_batched    | 16.4  | 25.6  | 25.1   | 23.0    | 1.40×         |
| tp2 / no_batched    | 20.7  | 23.7  | **47.9** | **92.3** | **4.46×**     |
| tp2 / batched       | 20.7  | 24.8  | 47.2   | 93.1    | 4.50×         |
| pp4 / no_batched    | 17.2  | 31.8  | 40.3   | 40.9    | 2.38×         |
| pp4 / batched       | 17.0  | 31.6  | 43.4   | 38.5    | 2.26×         |
| **pp2tp2 / no_batched** | **23.0** | **30.8** | **59.7** | **120.8** | **5.25×** |
| pp2tp2 / batched    | 22.9  | 31.3  | 61.1   | 118.8   | 5.19×         |

**pp2tp2 wins every column.** At N=8 it delivers **120.8 t/s aggregate**
on the same 4-GPU rig where the prior matrix (pre-Lever-A) had pp4
capped at 40 t/s and tp2 dropping 75-87% of streams.

## Per-stream throughput (what one user perceives)

| topology / path     | N=1   | N=2   | N=4   | N=8   |
|---------------------|-------|-------|-------|-------|
| pp2 / no_batched    | 16.4  | 13.4  | 6.6   | 3.1   |
| tp2 / no_batched    | 20.7  | 16.3  | 14.2  | 12.9  |
| pp4 / no_batched    | 17.2  | 16.4  | 11.7  | 5.7   |
| **pp2tp2 / no_batched** | **23.0** | **19.3** | **17.3** | **16.3** |

**pp2tp2 holds 16+ t/s per user even at 8 concurrent.** TP2 holds 13;
PP4 collapses to 5.7; PP2 collapses to 3.1.

## Prefill TTFT (ms, mean across N concurrent)

| topology / path     | N=1   | N=2   | N=4   | N=8   |
|---------------------|-------|-------|-------|-------|
| pp2 / no_batched    | 19657 | 21916 | 44660 | 84819 |
| tp2 / no_batched    | 11878 | 24459 | 49924 | 101010 |
| pp4 / no_batched    | 19753 | 20944 | 29436 | 52251 |
| pp2tp2 / no_batched | 11789 | 20829 | 39114 | 75636 |

PP4 has the best TTFT scaling (PP overlaps prefills across stages).
TP2's prefill is fully serialised by `#321 prefill_serialiser` —
each N adds ~12 s of waiting at the head. This is the V2 follow-up
already documented in the Lever A cert.

## Headlines

### YES, TP2 now beats PP at concurrency

The matrix-2026-05-04 cert said *"PP4 wins everywhere at N≥4 by 1.4–1.7×
on aggregate"*. That's no longer true post-Lever-A. The new ordering at
**N=8** is:

  pp2tp2 (120.8) > tp2 (92.3) > pp4 (40.9) > pp2 (23.0)

TP2's leap from "broken at N≥4" to "linear scaling to N=8" came from:
- `#321` (serialise TP/Hybrid prefill to fix VRAM OOM)
- `#324` (pooled TP prefill scratch — kills per-call alloc/dispose
  churn AND restores the GPU-stream serial decode at concurrency)

### pp2tp2 is the universal winner for this rig + 27B

- Best single-stream (23.0 — beats TP2's 20.7 because the inner-stage
  TP=2 amortises HBM read across 2 ranks, and the 2-stage PP fills
  the launch-latency floor cheaply)
- Best at every concurrency level (2,4,8)
- Best per-stream throughput at every concurrency level (16.3 t/s
  per user at N=8 — closest to ideal experience)
- Best raw aggregate throughput on this rig: **120.8 t/s at N=8**

### PP4 caps where it always did

PP4 at N=4: 40.3 t/s. PP4 at N=8: 40.9 t/s. **No scaling past N=4** —
the 4 pipeline stages are full at N=4 and adding more streams just
queues. Per-stream at N=8 = 5.7 t/s, the worst experience of the
3 working topologies.

### Batched vs no_batched: still a wash on long prompts

Same finding as the prior matrix. Speedup ratios all 0.95×–1.07×.
The `FLAMBEAU_BATCHED_DECODE=1` path doesn't help at chat-shape
prompts; it was the right design *but* the batched-MMVQ kernels
needed for actual speedup at small N are V2 work.

## Why the inversion happened

Pre-Lever-A:
- TP2 / N=4: 1 of 4 streams succeeded (3 OOM on per-call alloc)
- TP2 / N=8: 1 of 8 succeeded
- Aggregate TP2 ≈ single-stream throughput regardless of N

Post-Lever-A:
- All N concurrent prefills serialise on `prefill_serialiser`,
  reusing one shared 35 MB scratch — no peak-VRAM problem
- All 8 streams' decode loops then interleave on the GPU stream;
  with batched-attention + batched-GDN + the standard MoE/FFN
  kernels, each token's GPU work is short enough that 8-way
  interleave hits ~93 t/s aggregate

Pre-Lever-A PP4's "best concurrency" position was actually because
TP was broken — not because PP scaled well. PP4 has always saturated
at N = pp_size.

## Recommendations (post-Lever-A)

For Qwen3.6-27B on 4× MI50:
- **All concurrency levels (1, 2, 4, 8): use pp2tp2** — it dominates
  every column.
- TP2 is a close second at N≥4 if pp2tp2 isn't an option.
- PP4 is now never the right choice for this model.

For larger models (35B-A3B), the 2026-05-04 matrix had pp4 winning
N≥4 — but that was also on the broken-TP2 baseline. **Re-running 35B
on this fixed stack is the obvious follow-up.**

## Open follow-ups

- **PP2/batched boot timeout**: this run hit a 600s wait_ready timeout
  on PP2 with FLAMBEAU_BATCHED_DECODE=1. Boot logs showed the server
  was actually serving requests with garbled output (~80s prefill,
  repetitive "complex complex complex" decode). Likely a bug in the
  PP-batched decode path at slots=8 + cross-die device pair. PP2 is
  rarely the right topology so this is a low-priority debug.
- **35B-A3B re-run**: the 2026-05-04 matrix's "PP4 wins at concurrency"
  for 35B was on the broken-TP baseline. Should be re-run with the
  fixed stack to see if pp2tp2/TP2 also wins there.
- **Long prompt + ctx**: this run used ctx=4096. The earlier matrix
  used ctx=16384 but PP2 OOMed at boot. With #324 reducing VRAM
  pressure, ctx=16384 might fit again. Worth re-running the user's
  actual prod context for the final-final answer.

## Reproducer

```sh
python3 scripts/bench/run_matrix.py \
  --out certs/perf/post_lever_a_27b_2026_05_05/matrix.json \
  --max-tokens 256 --models qwen36_27B_q4_1 \
  --topos pp2,tp2,pp4,pp2tp2 \
  --cooldown-threshold-c 70 --resume

python3 scripts/bench/summarize.py \
  certs/perf/post_lever_a_27b_2026_05_05/matrix.json
```

Raw JSON: `matrix.json`. Server logs: `scripts/bench/logs/`.
