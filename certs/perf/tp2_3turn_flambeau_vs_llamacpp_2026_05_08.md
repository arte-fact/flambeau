# flambeau vs llama.cpp — TP2, 3-turn technical conversation

- **Date:** 2026-05-08
- **Hardware:** 2× AMD MI50 (gfx906, 16 GB VRAM each), PCIe 3.0 x16
- **Topology:** TP2 (per-tensor row split across 2 GPUs)
  - flambeau: `--mesh-mode tp --tp-size 2 --devices 0,1`
  - llama.cpp: `--split-mode row --tensor-split 1,1 -ngl 999`,
    `HIP_VISIBLE_DEVICES=0,1`
- **Models:** Qwen3.6-27B-Q4_0 (dense), Qwen3.6-35B-A3B-Q4_0 (MoE, 8/128 active)
- **Workload:** 3-turn conversation, each turn requests a long technical
  response (`max_tokens=512`, temperature=0.7, top_p=0.9, seed=0,
  `/no_think` to suppress Qwen3 think-blocks). Prompt accumulates.
- **Engines:**
  - flambeau commit `76e0330` (post flash-tile-Q8 + dp4a)
  - llama.cpp build-mi50 (b8911 era, ROCm 7.1.1, `-fa on --no-mmap`)
- **Common runtime:** F16 KV cache, 8192 ctx cap, 8 worker threads.

## Headline (averaged across 3 turns)

| | Qwen3.6-27B-Q4_0 (dense) | Qwen3.6-35B-A3B-Q4_0 (MoE) |
|---|---|---|
| flambeau decode | **24.34 tok/s** | **55.46 tok/s** |
| llama.cpp decode | 21.36 tok/s | 41.07 tok/s |
| **flambeau decode advantage** | **+14.0%** | **+35.1%** |
| flambeau prefill (cold turn 1) | **80 tok/s** | **412 tok/s** |
| llama.cpp prefill (cold turn 1) | 56 tok/s | 65 tok/s |
| **flambeau cold-prefill advantage** | **+43%** | **+6.3×** |

flambeau wins decode on both models; the MoE win is largest because
flambeau's MoE expert dispatch + multi-row MMVQ outperforms llama.cpp's
indexed-MoE-MMVQ on PCIe-only TP2.

## Per-turn breakdown

### Qwen3.6-27B-Q4_0 (dense)

| turn | ptok | ctok | engine | TTFT | prefill rate | decode rate |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 77   | 512 | **flambeau** | 961 ms | **80.1 tok/s** | **24.87 tok/s** |
| 1 | 77   | 512 | llama.cpp    | 1382 ms | 55.7 tok/s | 21.74 tok/s |
| 2 | 674  | 512 | **flambeau** | 2194 ms | **307.2 tok/s** | **24.17 tok/s** |
| 2 | 674  | 512 | llama.cpp    | 3098 ms | 217.6 tok/s | 21.41 tok/s |
| 3 | 1265 | 147* | **flambeau** | 4047 ms | 312.6 tok/s | **23.17 tok/s** |
| 3 | 1265 | 512 | llama.cpp    | 2809 ms | **450.4 tok/s**† | 20.95 tok/s |

*flambeau turn 3 hit EOS at 147 tokens (model-side stop, not max).
†llama.cpp turn 3 prefill exceeds flambeau because **llama-server has
prompt-cache enabled by default**, so it skips re-prefilling tokens
shared with the prior turn. flambeau in this bench had prefix-cache
disabled (`--prefix-cache 0` default). Cold-prefill (turn 1) is the
fair comparison: flambeau 80 vs llama 56 tok/s = +43%.

### Qwen3.6-35B-A3B-Q4_0 (MoE)

| turn | ptok | ctok | engine | TTFT | prefill rate | decode rate |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 77   | 512 | **flambeau** | 187 ms  | **412.3 tok/s** | **57.84 tok/s** |
| 1 | 77   | 512 | llama.cpp    | 1184 ms | 65.0 tok/s | 39.92 tok/s |
| 2 | 669  | 512 | **flambeau** | 727 ms  | **919.9 tok/s** | **55.82 tok/s** |
| 2 | 669  | 512 | llama.cpp    | 1285 ms | 520.6 tok/s | 40.17 tok/s |
| 3 | 1250 | 512 | **flambeau** | 1324 ms | **943.9 tok/s** | **52.93 tok/s** |
| 3 | 1250 | 512 | llama.cpp    | 1242 ms | 1006.4 tok/s† | 43.27 tok/s |

†llama.cpp prompt-cache hit. Cold turn 1: flambeau 412 vs llama 65
= **6.3× faster prefill**. Even at turn 3 with llama's cache hit,
flambeau decode wins by **+22.3%** (52.93 vs 43.27 tok/s).

## Methodology notes

- **TTFT vs prefill rate.** TTFT = time from `POST` to first SSE
  content chunk. Prefill rate = `prompt_tokens / TTFT`. Both engines
  agree on TTFT clock; llama.cpp does not surface `usage.prompt_tokens`
  in the streaming `chat-completion-chunk` schema (it does in the
  non-streaming path), so prefill rates for llama.cpp are computed
  using flambeau's reported prompt-token counts (the prompts are
  identical so the counts match).
- **Prompt cache.** llama-server enables prompt-cache by default; on
  multi-turn chats, turns 2/3 reuse the prefix KV from prior turns
  and the "prefill" they pay covers only the new tokens. flambeau in
  this run was launched without `--prefix-cache 1`. For an apples-
  to-apples cold-prefill comparison, **turn 1 is the only fair
  prefill data point**; turn 2/3 understates llama.cpp's actual
  per-token prefill cost. Decode comparisons are unaffected — both
  engines pay full per-token cost.
- **Decode rate variance across turns** (within each engine) is small
  (≤5%) on both models: the conversation never exceeds ~1700 tokens
  of context, well below split-K thresholds and the regime where
  attention-time dominates wall.
- **Sampling.** Both engines run with the same sampler config
  (temperature=0.7, top_p=0.9, seed=0). Output text differs between
  engines (different sampler implementations), but token counts and
  per-token rates are directly comparable.

## What this measures

- **Cold prefill rate** (turn 1, no cache) — model loading, first-
  forward warm-up included.
- **Steady-state decode rate** under realistic chat ctx (200-1700
  tokens, no extreme long-ctx).
- **Multi-turn throughput** — the regime real chat workloads run in.

## What this does not measure

- **Long-context decode** (n_kv > 4k). flambeau's split-K attention
  pays off there; this bench's max ctx is ~1.7k.
- **Concurrent multi-slot decode.** Single inflight request only.
- **Quality.** Both engines produce on-distribution output; no
  benchmark perplexity or reasoning evals here.
- **Q8 KV path.** flambeau ran F16 KV (default). Q8 KV would change
  the picture again — see `q8_flash_tile_24k_2026_05_08.md` for
  long-ctx Q8 numbers.

## Files

- `scripts/bench/llamacpp_vs_flambeau_tp2.py` — bench harness
- `certs/perf/tp2_3turn_flambeau_vs_llamacpp_2026_05_08.json` — raw
  per-turn timings
- Server logs (gitignored): `/tmp/llamacmp_bench_{flambeau,llamacpp}_{port}.log`

## Bottom line

On 2× MI50 PCIe TP2:
- flambeau decode beats llama.cpp by **+14%** on dense 27B and
  **+35%** on MoE 35B-A3B (averaged over a 3-turn long-response
  conversation).
- flambeau cold prefill beats llama.cpp by **+43%** on dense and
  **6.3×** on MoE. With prompt-cache enabled on both sides, the
  steady-state prefill gap narrows.
- The MoE-decode advantage (+35%) is the most valuable production
  number for the user's primary 35B-A3B workload.
