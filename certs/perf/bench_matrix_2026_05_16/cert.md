# flambeau vs llama.cpp — single-stream chat completions matrix (2026-05-16)

Single-stream chat-completion bench via OpenAI-compatible
`/v1/chat/completions` (streaming SSE) on both engines. One long
prompt (~360-389 tokens after tokenization), `max_tokens=192`,
`temperature=0.7 top_p=0.9 seed=0`.

- **Rig**: 4× MI50 (gfx906, 16 GB each)
- **Engines**:
  - flambeau commits `510189e` (HEAD) + serve.rs fixes shipped during the bench:
    - gemma4 serve: lift `MeshMode::Hybrid` `unreachable!` (today's #102 driver had no serve wiring)
    - gemma4 serve: pass `cfg_g4.context_length` instead of `prefill_ubatch` as KV `max_tokens` (KV was capped at 512)
  - llama.cpp: `build-mi50/bin/llama-server` (build 5d2b52d80 / 8911)
- **llama.cpp topology mapping**: tp2 → `-sm row -ts 1,1`; pp2/pp4 → `-sm layer -ts ...`; **pp2tp2 has no native equivalent — approximated as `-sm layer` on 4 GPUs (= pp4 for llama.cpp)**
- **prompt**: ~600+ chars (~360 tokens) on gfx906/dp4a internals — verbose, technical
- **runner**: `scripts/bench/llamacpp_vs_flambeau_matrix.py`

Numbers are `tok/s` (prefill = `ptok / ttft_s`; decode = `1000 / decode_ms_per_tok`).

## Results

| model | topo | engine | ptok | ctok | prefill (tok/s) | decode (tok/s) |
|---|---|---|---:|---:|---:|---:|
| qwen35-9b-q4_1 | tp2 | flambeau | 360 | 192 | **841.8** | 44.55 |
| qwen35-9b-q4_1 | tp2 | llama.cpp | 354 | 192 | 447.5 | **54.09** |
| qwen35-9b-q4_1 | pp2 | flambeau | 360 | 192 | **571.4** | 42.56 |
| qwen35-9b-q4_1 | pp2 | llama.cpp | 354 | 192 | 529.4 | **59.32** |
| qwen35-9b-q4_1 | pp4 | flambeau | 360 | 192 | **582.4** | 38.73 |
| qwen35-9b-q4_1 | pp4 | llama.cpp | 354 | 192 | 529.6 | **59.39** |
| gemma4-31b-q4_0 | tp2 | flambeau | 389 | 192 | 19.4 | 18.64 |
| gemma4-31b-q4_0 | tp2 | llama.cpp | 381 | 192 | **178.0** | **21.37** |
| gemma4-31b-q4_0 | pp2 | flambeau | 389 | 192 | 177.1 | 5.16 |
| gemma4-31b-q4_0 | pp2 | llama.cpp | 381 | 192 | **181.3** | **19.31** |
| gemma4-31b-q4_0 | pp4 | flambeau | 389 | 192 | 177.7 | 5.27 |
| gemma4-31b-q4_0 | pp4 | llama.cpp | 381 | 192 | **186.1** | **20.88** |
| gemma4-31b-q4_0 | pp2tp2 | flambeau | 389 | 192 | 21.7 | **21.45** |
| gemma4-31b-q4_0 | pp2tp2 | llama.cpp¹ | 381 | 192 | **184.6** | 20.84 |
| qwen36-35b-a3b-q4_0 | pp2 | flambeau | – | – | – | – |
| qwen36-35b-a3b-q4_0 | pp2 | llama.cpp | 354 | 192 | 299.1 | 61.37 |
| qwen36-35b-a3b-q4_0 | pp4 | flambeau | – | – | – | – |
| qwen36-35b-a3b-q4_0 | pp4 | llama.cpp | 354 | 192 | 206.5 | 58.81 |
| qwen36-35b-a3b-q4_0 | pp2tp2 | flambeau | 360 | 192 | **510.1** | 30.26 |
| qwen36-35b-a3b-q4_0 | pp2tp2 | llama.cpp¹ | 354 | 192 | 217.5 | **58.99** |
| gemma4-26b-a4b-q8_0 | pp2 | flambeau | – | – | – | – |
| gemma4-26b-a4b-q8_0 | pp2 | llama.cpp | 381 | 192 | 484.3 | 65.83 |
| gemma4-26b-a4b-q8_0 | pp4 | flambeau | – | – | – | – |
| gemma4-26b-a4b-q8_0 | pp4 | llama.cpp | 381 | 192 | 484.7 | 64.18 |
| gemma4-26b-a4b-q8_0 | pp2tp2 | flambeau | 389 | 192 | 45.5 | 42.39 |
| gemma4-26b-a4b-q8_0 | pp2tp2 | llama.cpp¹ | 381 | 192 | **482.3** | **63.03** |

¹ llama.cpp pp2tp2 is `-sm layer` on 4 GPUs (= pp4 for llama.cpp; no native 2D mesh).

## Ratios (flambeau / llama.cpp)

| model | topo | prefill | decode |
|---|---|---:|---:|
| qwen35-9b-q4_1 | tp2 | **1.88×** | 0.82× |
| qwen35-9b-q4_1 | pp2 | **1.08×** | 0.72× |
| qwen35-9b-q4_1 | pp4 | **1.10×** | 0.65× |
| gemma4-31b-q4_0 | tp2 | 0.11× | 0.87× |
| gemma4-31b-q4_0 | pp2 | 0.98× | 0.27× |
| gemma4-31b-q4_0 | pp4 | 0.95× | 0.25× |
| gemma4-31b-q4_0 | pp2tp2 | 0.12× | **1.03×** |
| qwen36-35b-a3b-q4_0 | pp2tp2 | **2.35×** | 0.51× |
| gemma4-26b-a4b-q8_0 | pp2tp2 | 0.09× | 0.67× |

## Highlights

### Where flambeau wins

- **Qwen3.5-9B-Q4_1 prefill across all topologies** — `1.08-1.88×` faster
  than llama.cpp. Strongest at TP2 (`1.88×`).
- **Gemma4-31B-Q4_0 pp2tp2 decode** — `1.03×` faster than llama.cpp's
  `-sm layer` 4-GPU split (today's #102 hybrid driver shipping value
  immediately measurable).
- **Qwen3.6-35B-A3B-Q4_0 pp2tp2 prefill** — `2.35×` faster (MoE batched
  prefill engages cleanly via the qwen3-moe hybrid driver).

### Where flambeau loses

- **All dense decode** — llama.cpp consistently 20-40% faster on
  single-stream decode for 9B (54 vs 45 t/s at TP2). Known gap; sampler
  work since Sampler-D4 closed the chat case at 35B but the per-stream
  decode kernel itself is still ~20% behind on dense Qwen.
- **Gemma4 prefill on TP** — `0.11×` (TP2: 19.4 vs 178 t/s). Gemma4 has
  no batched-prefill kernel on TP; falls back to per-token forward.
  Same root issue causes the pp2tp2 prefill at `0.12×` (45 vs 482 t/s).
  PP path uses the batched-prefill kernel and matches llama.cpp (`0.95-0.98×`).
- **Gemma4-31B PP decode** — `0.25-0.27×`. Pure pipeline serialisation
  with no TP-style weight split; llama.cpp's `-sm layer` apparently
  overlaps more aggressively. TP/hybrid paths recover (`0.87×` / `1.03×`).

## Failed cells (flambeau)

All failures are **pre-existing** issues outside today's scope:

| model | topo | failure |
|---|---|---|
| qwen36-35b-a3b-q4_0 | pp2 | `peer_copy DtoH sync: an illegal memory access was encountered` (qwen3-moe PP path) |
| qwen36-35b-a3b-q4_0 | pp4 | same |
| gemma4-26b-a4b-q8_0 | pp2 | `MoE prefill not supported in S8-B-A; see #23` |
| gemma4-26b-a4b-q8_0 | pp4 | same |

The gemma4-MoE PP failures are the known `forward_layer_prefill` MoE
gap (`#23`); hybrid path avoids it by per-token prefill (works at
pp2tp2).

## Fixes shipped during the bench

Two real bugs surfaced and patched mid-run (in commits to follow):

1. **`server/serve.rs`** — `Gemma4{Pp,Tp,Hybrid}Driver::upload` was
   getting `prefill_ubatch` (default 512) as its `max_tokens` arg.
   That sized the KV cache at 512 tokens regardless of `--ctx-cap`,
   so any prompt+decode > 512 errored mid-stream
   (`kv_cache.append: capacity exceeded`). Fix: pass
   `cfg_g4.context_length`.
2. **`server/serve.rs`** — `MeshMode::Hybrid` bailed with `unreachable!`
   for gemma4, even though today's `Gemma4HybridDriver::upload`
   (commit `81b3030`, fixes in `cc68970`) ships. Lift the bail + wire
   per-stage sub-cluster + global cluster construction (mirrors
   `qwen3-moe`'s hybrid arm).

Both surfaced ONLY because the matrix exercised cells outside the
hand-tested smoke set — same pattern as #94's parity harness catching
the hybrid composer regression.

## Raw data

- Part 1 (qwen35-9b, all topos): `/tmp/bench_part1_9b.log`
- Part 2 (gemma + qwen36 partial, KV-fix not yet applied): superseded
- Part 3 (post-fix re-run for gemma4 + qwen36): `/tmp/bench_full_out.log`
  + `certs/perf/bench_matrix_2026_05_16/part3.json`

Bench source: `scripts/bench/llamacpp_vs_flambeau_matrix.py`.
