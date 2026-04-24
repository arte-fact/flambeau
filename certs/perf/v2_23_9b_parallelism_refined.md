# 9B Q4_1 parallelism head-to-head — refined analysis

Post-V2.23 comparison of flambeau against `llamacpp-turbo` on Qwen3.5-9B-Q4_1
across Mesh<1>, Mesh<2>, Mesh<4>. Initial analysis (see
`v2_23_9b_parallelism.md` draft in session) contained two errors corrected
here via documentation check.

## Bench (unprofiled, tok/s, median of 1-3 reps)

| | flambeau M<1> | flambeau M<2> | flambeau M<4> | turbo M<1> | turbo M<2> | turbo M<4> |
|---|---:|---:|---:|---:|---:|---:|
| pp 8 | 95.80 | 95.41 | 92.93 | — | — | — |
| pp 64 | 98.82 | 98.77 | 99.88 | — | — | — |
| pp 128 | 542.52 | 539.29 | 534.27 | — | — | — |
| pp 512 | **757.05** | 583.00 | 749.90 | **1033** | 1026 | 949 |
| pp 1024 | **772.08** | 768.87 | 766.07 | **1030** | **1319** | **1569** |
| tg 64 | **66.35** | 47.81 | 53.06 | **74.44** | 69.23 | 66.46 |

Observations:
- Turbo prefill L=1024 scales Mesh<1>→<4> **+52 %** (1030 → 1569).
- Flambeau prefill L=1024 across Mesh<1>→<4> is **flat** (772 → 766).
- Decode tok/s both drop with more ranks, Mesh<1> is fastest for both.
- Flambeau Mesh<1> `decode tg=64 = 66.35` beats Mesh<2> / Mesh<4>.

## Profile (decode tg=64, rocprofv3 kernel-trace)

| config | kernel_sum | calls | per-rank ms (min/avg/max) | max/min | per-rank busy % of measurement wall |
|---|---:|---:|---:|---:|---:|
| flambeau M<1> | 1023 | 59,025 | 1023 | 1.00 | 106 % (incl. warmup) |
| flambeau M<2> | 1144 | 59,157 | 525 / 572 / 618 | 1.18 | ~43 % (of wall/rank, ideal 50 %) |
| flambeau M<4> | 1223 | 59,421 | 283 / 306 / **372** | **1.32** | ~25 % (ideal 25 %) |
| turbo M<1> | 887 | 62,001 | 887 | 1.00 | 103 % (incl. warmup) |
| turbo M<2> | 970 | 62,393 | 478 / 485 / 492 | 1.03 | ~52 % (ideal 50 %) |
| turbo M<4> | 1050 | 63,177 | 228 / 263 / 287 | 1.26 | ~27 % (ideal 25 %) |

## Two corrections to the initial draft

### Correction 1: decode PP does NOT support cross-rank overlap (single request)

Initial draft observed `wall < kernel_sum` on turbo Mesh<2>/<4> and called it
"cross-rank overlap". This was misattributed: the profile captures all HSA
calls from process start (including warmup + weight upload tail), while the
measurement wall is just the 64-token decode phase. The excess kernel time is
warmup/init, not overlap.

Confirmed by the TD-Pipe paper (*Zhang et al., arXiv 2506.10470*, §2.1):

> inter-decode-step data dependency exists as the execution of generating the
> next token cannot start until the previous iteration completes

Single-request decode is *structurally* serial. With PP of depth N, each GPU
runs ≈ 1/N of wall; aggregate sum ≈ 100 % of wall. Both runtimes hit this
ideal at Mesh<4>:
- flambeau: 306/1206 per rank = 25.4 % (ideal 25 %)
- turbo: 263/964 per rank = 27.3 % (ideal 25 %)

Both saturated. The **decode gap is per-rank kernel time**, not overlap.

| | per-rank kernel_ms | wall ms | tok/s |
|---|---:|---:|---:|
| flambeau M<4> | 306 | 1206 | 53.1 |
| turbo M<4> | 263 | 964 | 66.5 |
| Δ | **+16 %** | **+25 %** | **−20 %** |

Per-rank flambeau is 16 % more kernel work; wall is 25 % slower (the extra
~9 % is peer-copy + PP bubble overhead that both runtimes pay but flambeau
pays slightly more — 1 more layer on Mesh<2>, imbalance at Mesh<4>).

This is a **big revision** vs the V2.28 position. The V2.28 doc measured
flambeau at `device/wall per-GPU = 0.56` on 27B-Q8_0, i.e. 44 % idle. Today
on 9B Q4_1 we're at 0.25/0.25 = **1.00**, essentially saturated. The V2.23
launch-fusion cycle has closed the host-idle gap.

### Correction 2: prefill scaling is about microbatching, not a flambeau bug

Initial draft claimed flambeau has a "fixed serial component" blocking
prefill scaling. The deeper truth (per llama.cpp PR #6017 and
`ggml-org/llama.cpp` discussion #20252) is:

llama.cpp's `llama_context: pipeline parallelism enabled` applies to prefill
only when `n_batch > n_ubatch`. The scheduler splits the prefill batch into
micro-batches and issues them in sequence to rank 0; while rank 0 works on
ubatch k, rank 1 works on ubatch k-1, etc. This is classic 1F1B pipeline
fill. Maintainer ggerganov: *"If you configure it correctly, the PP
performance scales nearly linear with the number of devices, even for single
request."*

Without microbatching, PP on a single prefill batch is **strictly
serial**: rank 0 processes all its layers, hands off to rank 1, rank 1
processes, hands off to rank 2, etc. Per-layer compute is unchanged vs
single-GPU; wall doesn't improve vs single-GPU.

Flambeau's `forward_prefill_pp` submits the whole L-token batch as one
logical unit — no microbatching. That's why:
- per-rank kernel time scales linearly with N ranks (work per rank halves)
- but wall doesn't scale (layers still sequential)

Turbo's prefill wall does scale (1030 → 1319 → 1569) because it
microbatches. Our per-rank kernel time scales fine; our **wall doesn't
because we don't micro-batch**.

### Revised lever list

| lever | urgency | action |
|---|---|---|
| per-kernel speed on 9B decode | top (16 % gap at saturation) | tune Q4_1 MMVQ, F16 attention kernels — 9B Q4_1 doesn't hit MMQ at decode |
| prefill microbatching | **top, architectural** | split `forward_prefill_pp` batch into ubatches of size ~256-512; schedule across ranks with event-based async waits |
| rank balance | moderate (M<4>=1.32) | model-specific `FLAMBEAU_PP_LAYERS`; 32-layer Qwen3.5-9B wants 9/8/8/7 or similar |
| launch overhead | **closed** for 9B decode | V2.23 cycle closed the V2.28 host-idle gap |

## Data sources

- TD-Pipe paper: Zhang et al., *TD-Pipe: Temporally-Disaggregated Pipeline
  Parallelism Architecture for High-Throughput LLM Inference*,
  arXiv:2506.10470v1, June 2025 — documents inter-decode-step data dependency
  and why pure PP on single-request decode has no overlap.
- llama.cpp PR #6017 (`f30ea47`): adds `n_ubatch` parameter and graph
  duplication (`LLAMA_SCHED_MAX_COPIES`) — enables prefill microbatch PP.
- llama.cpp discussion #20252 (Mar 2026): maintainer confirms PP requires
  `n_batch > n_ubatch` to be active.
- `ggml-org/llama.cpp` DeepWiki §3.5 "Batch Processing Pipeline" — describes
  `llama_batch → llama_ubatch` split.
- ik_llama.cpp `-sm graph` mode (Medium article, Jan 2026) — tensor
  parallelism at GGML graph level; achieves 3-4× vs layer split. **Not
  pipeline parallelism** — orthogonal mechanism. Requires NCCL (or RCCL
  equivalent on AMD). A future option for flambeau once RCCL ships in
  decode hot path (currently deferred per CLAUDE.md V1 posture).

## Concrete next-cycle plan (V2.24)

**V2.24.a prefill microbatching** — the biggest single win available.
Structural but scoped:
1. Extend `ShardedForwardPrefillScratch` to track a ubatch cursor.
2. In `forward_prefill_pp`, chunk `tokens[0..L]` into ubatches of size
   `FLAMBEAU_UBATCH` (default 256) and drive the stage pipeline one ubatch
   at a time.
3. Replace the blocking `stream.synchronize()` at stage boundaries with
   `hipEventRecord` / `hipStreamWaitEvent` so rank k+1 can start ubatch i
   while rank k starts ubatch i+1.
4. Target: L=1024 M<4> ≥ 1100 tok/s (73 % of turbo) — per the ideal
   `1 + (N-1)/N × ubatches_per_batch` scaling for depth-N pipeline.

**V2.24.b per-kernel tune on 9B path** — close the 16 % per-rank gap.
Diagnostic first: rocprofv3 per-kernel breakdown at flambeau M<1> decode
vs turbo M<1> decode, look for the dominant single-kernel delta.

**V2.24.c Mesh<1> KV size cap** — add `FLAMBEAU_KV_MAX_CONTEXT` so 9B Q4_1
fits on one MI50 when user requests it. Today's KV alloc = 512 MB ×
32 layers = 16 GB is too big for a 16 GB card.

## Gate / validation

- UD-Q4_K_S 8-token parity: **preserved** (not touched by this cycle).
- cert-check hip/gfx906: **48 rows, 0 failures**.
- Numbers reproducible via:
  ```
  for m in 1 2 4; do
    FLAMBEAU_MESH_RANKS=$m FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
      ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b --nocapture
  done
  # turbo:
  for sm in "-ts 1 -mg 0" "-ts 1/1 -mg 0" "-sm layer"; do
    LD_LIBRARY_PATH=... llama-bench -m Qwen3.5-9B-Q4_1.gguf -p 512,1024 -n 64 \
      -ngl 99 $sm -fa 1 -r 1
  done
  ```
