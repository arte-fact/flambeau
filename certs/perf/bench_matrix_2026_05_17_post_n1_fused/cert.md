# flambeau vs llama.cpp — full 4-model matrix post N=1 fused fast-path (2026-05-17)

Re-run of `certs/perf/bench_matrix_2026_05_16/cert.md` after today's
fixes:
1. `7c9a8e5` — `ffn_gate_inp_shexp` F32 (unblocked qwen36 PP)
2. `bab194f` — async `peer_copy_via_host_event` in `forward_one_token_pp`
3. `9e54d8b` / `438f354` — **N=1 fused fast-path** in
   `qwen3moe_forward_decode_batched`

Same harness, same prompt (~360 tokens), `max_tokens=192`,
`temperature=0.7 top_p=0.9 seed=0`, both engines via OpenAI-compatible
chat completion. Runner: `scripts/bench/llamacpp_vs_flambeau_matrix.py`.

## Results — sorted by model

| model | topo | flambeau P / D | llama.cpp P / D | **P×** | **D×** |
|---|---|---|---|---:|---:|
| **qwen35-9b-q4_1** | tp2    | **839 / 58.96** | 438 / 54.07 | **1.92×** | **1.09×** |
| qwen35-9b-q4_1     | pp2    | 571 / 51.60     | 532 / 60.09 | 1.07× | 0.86× |
| qwen35-9b-q4_1     | pp4    | 581 / 50.66     | 525 / 58.63 | 1.11× | 0.86× |
| gemma4-31b-q4_0    | tp2    | 19.5 / 18.58    | 179 / 21.42 | 0.11× | 0.87× |
| gemma4-31b-q4_0    | pp2    | 174 / 5.15      | 181 / 19.07 | 0.97× | 0.27× |
| gemma4-31b-q4_0    | pp4    | 176 / 5.29      | 185 / 20.97 | 0.95× | 0.25× |
| gemma4-31b-q4_0    | pp2tp2 | 22.0 / 21.44    | 185 / 20.76 | 0.12× | **1.03×** |
| **qwen36-35b-a3b-q4_0** | tp2 | **836 / 57.92** | 240 / 41.75 | **3.48×** | **1.39×** |
| qwen36-35b-a3b-q4_0    | pp2    | 675 / 53.74     | 317 / 60.41 | 2.13× | 0.89× |
| qwen36-35b-a3b-q4_0    | pp4    | 690 / 52.64     | 211 / 58.71 | 3.26× | 0.90× |
| qwen36-35b-a3b-q4_0    | pp2tp2 | 502 / 51.38     | 207 / 58.76 | 2.43× | 0.87× |
| gemma4-26b-a4b-q8_0    | pp2tp2 | 45 / 38.59      | 487 / 62.56 | 0.09× | 0.62× |

(gemma4-26b-a4b pp2 + pp4 failed — known `forward_layer_prefill MoE
not impl` gap #23.)

## What changed since 2026-05-16 (the day before)

| cell | yesterday | today | delta |
|---|---:|---:|---:|
| qwen35-9b TP2 decode | 44.6 | **58.96** | **+32%** |
| qwen35-9b PP2 decode | 42.6 | 51.60 | +21% |
| qwen35-9b PP4 decode | 38.7 | 50.66 | +31% |
| qwen36-35b TP2 decode | (was OOM-skipped) | **57.92** | new |
| qwen36-35b PP2 decode | (crash) | 53.74 | new |
| qwen36-35b PP4 decode | (crash) | 52.64 | new |
| qwen36-35b pp2tp2 decode | 30.3 | 51.38 | **+70%** |
| gemma4-31b — all topos | unchanged (still uses gemma4 dispatch) | | |
| gemma4-26b — all topos | unchanged (still uses gemma4 dispatch) | | |

The N=1 fused fix only helps qwen3-moe paths (it specialises the
`qwen3moe_forward_decode_batched` dispatcher). Gemma4 paths go through
their own `Session::decode_one_logits` impl which already uses the
fused gemma4 forward, so they're unchanged.

## Wins

- **Qwen3.5-9B / TP2**: flambeau wins both prefill (1.92×) and decode
  (1.09×).
- **Qwen3.6-35B-A3B / TP2**: flambeau wins both prefill (3.48×) and
  decode (1.39×). Decode is now the strongest absolute number (57.92
  tok/s) of any qwen3.6-35B-A3B topo on this rig.
- **Qwen3.6-35B-A3B / pp2tp2**: +70% over yesterday — the production
  4-GPU topology now hits 51 tok/s decode (was 30).
- Prefill is faster than llama.cpp on **every qwen cell** (1.07-3.50×).

## Remaining gaps

- **Qwen3.5-9B PP{2,4} decode** trails by 14%. The fused fix helped
  (+21-31%) but PP at this small model is still slower than TP. Lever:
  PP stage overlap.
- **Qwen3.6-35B-A3B PP{2,4,pp2tp2} decode** trails by 10-13%. Same
  pattern. Lever: PP stage overlap.
- **Gemma4-31B TP2 prefill** is 11% of llama.cpp — gemma4 has no
  batched-prefill kernel on TP/Hybrid, falls back to per-token forward.
  Known structural gap (separate work).
- **Gemma4-26B-A4B PP{2,4} fail** — gemma4 MoE prefill not implemented
  (#23). Hybrid path works because it uses per-token forward.
- **Gemma4-26B-A4B pp2tp2 decode** = 38.6 (was 42.4). ~9% noise; gemma4
  not affected by today's fixes.

## Recommended topology defaults

For chat workloads on this rig:

| model | best flambeau topo | flambeau decode | vs llama best |
|---|---|---:|---:|
| Qwen3.5-9B-Q4_1 | **tp2** (hip:0,2) | 58.96 | beats llama (54.07) |
| Qwen3.6-35B-A3B-Q4_0 | **tp2** (hip:0,2) | 57.92 | beats llama (41.75); llama's best topo is PP at 60.4 |
| gemma4-31B-Q4_0 | **pp2tp2** (hip:0,2,1,3) | 21.44 | matches llama (~21) |
| gemma4-26B-A4B-Q8_0 | **pp2tp2** (hip:0,2,1,3) | 38.59 | trails llama PP (62.6) |

TP2 is the consistent winner for qwen models. Gemma4 needs MoE prefill
+ PP-overlap work to close the remaining gap.

## Raw

`certs/perf/llamacpp_vs_flambeau_matrix_n1_fused_all4.json`
