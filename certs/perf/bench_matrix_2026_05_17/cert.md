# flambeau vs llama.cpp — qwen36-35b-a3b-q4_0 TP2 + PP retry (2026-05-17)

Follow-up to `certs/perf/bench_matrix_2026_05_16/cert.md` after fixing
the qwen3-moe shared-expert router F32→F16 OOB bug
(commit `7c9a8e5` — see header for diagnosis).

Same single-stream chat-completion bench harness:
`scripts/bench/llamacpp_vs_flambeau_matrix.py`. One long prompt
(~360 tokens), `max_tokens=192`, `temp=0.7 top_p=0.9 seed=0`.

## Results (filling in the previously-failing flambeau cells)

| model | topo | engine | ptok | ctok | prefill (tok/s) | decode (tok/s) |
|---|---|---|---:|---:|---:|---:|
| qwen36-35b-a3b-q4_0 | tp2 | flambeau | 360 | 192 | **859.0** | 35.66 |
| qwen36-35b-a3b-q4_0 | tp2 | llama.cpp | 354 | 192 | 237.3 | **43.63** |
| qwen36-35b-a3b-q4_0 | pp2 | flambeau | 360 | 192 | **686.3** | 34.49 |
| qwen36-35b-a3b-q4_0 | pp2 | llama.cpp¹ | 354 | 192 | 299.1 | **61.37** |
| qwen36-35b-a3b-q4_0 | pp4 | flambeau | 360 | 192 | **699.9** | 32.16 |
| qwen36-35b-a3b-q4_0 | pp4 | llama.cpp¹ | 354 | 192 | 206.5 | **58.81** |
| qwen36-35b-a3b-q4_0 | pp2tp2 | flambeau | 360 | 192 | **510.1** | 30.26 |
| qwen36-35b-a3b-q4_0 | pp2tp2 | llama.cpp¹ | 354 | 192 | 217.5 | **58.99** |

¹ llama.cpp numbers from the 2026-05-16 cert — unchanged.

## Ratios (flambeau / llama.cpp)

| topo | prefill | decode |
|---|---:|---:|
| tp2 | **3.62×** | 0.82× |
| pp2 | 2.30× | 0.56× |
| pp4 | 3.39× | 0.55× |
| pp2tp2 | 2.35× | 0.51× |

## Key observations

1. **Flambeau decode is flat across topologies** (30-36 t/s); llama.cpp
   gets a large boost from `-sm layer` PP (43 → 60 t/s).
2. **Smallest decode gap is on TP2** (`0.82×`, ~18% lag). The largest
   is on `-sm layer` PP topologies (`0.51-0.56×`, ~45% lag).
3. **Flambeau prefill is consistently 2.3-3.6× faster** than
   llama.cpp on every topology. TP2 is the strongest at `3.62×`.

## Implication

For 35B-A3B Q4_0 single-stream chat:
- **Best flambeau decode topology = TP2** (35.66 t/s) — also the
  topology where flambeau's relative position is best.
- **Best llama.cpp decode topology = PP** (`-sm layer`, 60 t/s).
- The TP/PP asymmetry on llama suggests per-expert kernel-launch
  overhead is hidden under PP (sequential layers, fewer concurrent
  micro-launches) but exposed under TP (every rank fires its sliced
  MMVQ at every layer). Flambeau doesn't get that PP benefit —
  hypothesis: our per-expert MMVQ launches dominate decode wall
  regardless of topology.

## tg-improvement leverage analysis

Recommended next step: **rocprofv3 kernel-time breakdown on flambeau
35B-A3B / pp2 / decode-only**, with llama.cpp on the same cell as
reference. If the breakdown shows expert MMVQ kernel-launch overhead
or per-expert dispatch as a top consumer, the lever is a fused/indexed
batched-MoE-MMVQ kernel at decode-1 (one launch for all top_k experts
instead of K). Existing `mmvq_*_batched` kernels are for batch-N
slots, not across-expert at decode-1, so this is a separate kernel.

If the breakdown shows AllReduce or HBM bandwidth as the cap, the
lever is different — TP2's 0.82× ratio (vs PP's 0.55×) is consistent
with the launch-overhead hypothesis but doesn't prove it.

Raw: `certs/perf/llamacpp_vs_flambeau_matrix_qwen36_tp2.json`,
`certs/perf/llamacpp_vs_flambeau_matrix_qwen36_retry.json`.
