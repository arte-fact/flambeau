# pp2tp2 4-model matrix (long prompt + long response)

Topology: pp2tp2 on devices `hip:0,2,1,3` (pp_size=2, tp_size=2).
Features: FLAMBEAU_BATCHED_DECODE=1, GPU_SAMPLER=1, PREFIX_CACHE=1 (4 GiB),
4 inflight slots, 32 max-queue depth, 512-token prefill ubatch.
Prompt: ~2 k-token long-prompt fixture (chunked to 4 prefill chunks).
Sampling: greedy (temperature=0, seed=42), max_tokens=512.
VRAM: peak per-GPU MiB observed via rocm-smi during prefill+decode.
GPU%: average rocm-smi GPU-use% sample across the prefill+decode window.

| model | load (s) | prompt tok | gen tok | prefill tok/s | decode tok/s | peak VRAM (MiB) GPU 0/1/2/3 | avg GPU% 0/1/2/3 |
|---|---:|---:|---:|---:|---:|---|---|
| qwen35_9B_q4_1 | 4.2 | 4062 | 512 | 681.7 | 48.7 | 3078/3266/3074/3262 | 27/62/26/56 |
| qwen36_27B_q4_1 | 18.4 | 4062 | 512 | 254.3 | 23.1 | 6226/6465/6222/6461 | 27/65/26/63 |
| qwen36_35B_a3b_q4_0 | 38.8 | 4062 | 512 | 744.5 | 43.7 | 6012/6099/6009/6095 | 53/33/55/33 |
| qwen3_coder_next_q4_0 | 98.0 | 4074 | 512 | 543.8 | 34.1 | 12111/12091/12109/12089 | 38/45/40/44 |

Raw JSON: `certs/perf/matrix_pp2tp2_4models/raw.json`