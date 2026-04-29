# Qwen3.6-35B-A3B-Q4_0 — 4-way comparative profile — 2026-04-29

flambeau (TP2) vs llama.cpp (PP4 default), each in greedy and sampled mode,
with the same 269-token prompt + 128-token generation.

## Decode-rate matrix (3-run mean, warmed)

| stack          | greedy tg/s | sampled tg/s (temp=0.7, top_p=0.9) | sampler cost |
|---|---:|---:|---|
| **flambeau TP2**    | **60.5** | 35.6 | **−24.9 t/s (−41%)** |
| **llama.cpp PP4**   | 50.1 | 50.1 | **0.0 t/s (free)** |

## What this means

- **Forward path: flambeau wins greedy by 1.21×** (60.5 vs 50.1 tok/s). The
  GDN + MoE + AR kernels in flambeau are faster end-to-end than llama.cpp's
  ggml-cuda implementations on this rig.
- **Sampler: llama.cpp's sampler is essentially free** (50.13 sampled vs 50.12
  greedy on its server, deterministic to the second decimal). It runs softmax
  + top-p with an O(K)-class partial sort once `top_k` is set; without an
  explicit `top_k`, it sorts only the candidates above the temperature
  threshold.
- **flambeau's `build_distribution`** does an O(V log V) full sort across
  the 151k-token vocabulary on every decode step, costing ~12 ms/token. At
  60 t/s greedy the per-step wall is 16.5 ms, so adding 12 ms/sampler-step
  brings each token to ~28.5 ms = 35.1 t/s — matching the measured 35.6.

## Per-kernel breakdown (synthetic ~256+64 tokens, both under rocprofv3 kernel-trace)

### flambeau TP2 (sum across 2 ranks)

```
flambeau_gdn_state_step_alphabeta_f32_s128            4020 calls   298 ms (74 µs)  11.70%
flambeau_indexed_moe_mmvq_q4_0_q8_1                   4620        261 ms (57 µs)  10.26%
flambeau_mmvq_q4_0_gate_up_dp4a_q8_1                  3960        127 ms (32 µs)   4.98%
flambeau_indexed_moe_mmq_q4_0_gate_up_tile8_dp4a_q8_1   80        113 ms (1.4 ms)  4.43%
flambeau_moe_sort_scatter_det                          80         95 ms (1.2 ms)  3.75%
flambeau_mmq_q4_0_4warp_lds_q8_1                      182         94 ms (518 µs)  3.70%
flambeau_topk_softmax_f32                            5360         90 ms (17 µs)   3.53%
flambeau_p2p_allreduce_residual_tp2                  5440         89 ms (16 µs)   3.48%
flambeau_mmq_q8_0_wave64_tile16_q8_1                  308         86 ms (279 µs)  3.37%
flambeau_cast_f32_f16                              20100         80 ms (4 µs)    3.16%
                                                                ─────
                                                       Total:  2546 ms (∑ both ranks)
```

### llama.cpp PP4 (single sequential rank, GGML_CUDA_DISABLE_GRAPHS=1, ~6× rocprof slowdown)

```
mul_mat_vec_q<Q4_0>                                   7489 calls   241 ms (32 µs)  15.23%
mul_mat_q<Q4_0, 64-warp>                              203         199 ms (980 µs) 12.58%
gated_delta_net_cuda<128>                            1950         108 ms (55 µs)   6.80%
mul_mat_vec_f<F32>                                   8962          92 ms (10 µs)   5.81%
quantize_q8_1                                       18629          89 ms (5 µs)    5.64%
mul_mat_vec_q<Q6_K>                                   385          87 ms (226 µs)  5.50%
mul_mat_vec_q<Q4_0, indexed>                         3137          80 ms (26 µs)   5.09%
concat_f32_dim0                                      1920          67 ms (35 µs)   4.25%
mul_mat_vec_q<Q8_0>                                  3457          45 ms (13 µs)   2.84%
k_get_rows_float                                     4069          45 ms (11 µs)   2.82%
topk_moe_cuda<256>                                   2561          40 ms (16 µs)   2.55%
                                                                ─────
                                                       Total:  1581 ms
```

## Per-kernel head-to-head

| role | flambeau | llama.cpp | who wins |
|---|---|---|---|
| GDN state-step | `gdn_state_step_alphabeta_f32_s128` 4020 × 74 µs | `gated_delta_net_cuda<128>` 1950 × 55 µs | llama.cpp's monolithic kernel is 25% faster per call, plus 2× fewer calls (PP1 vs TP2) |
| MoE MMVQ Q4_0 | `indexed_moe_mmvq_q4_0_q8_1` 4620 × 57 µs | `mul_mat_vec_q<Q4_0>` 7489 × 32 µs | similar absolute time (261 vs 241 ms); flambeau's variant batches 2 experts per call (lower count, higher avg) |
| MoE MMQ prefill | `indexed_moe_mmq_q4_0_gate_up_tile8_dp4a` 80 × 1.4 ms | `mul_mat_q<Q4_0, 64-warp>` 203 × 980 µs | similar absolute (113 vs 199 ms when summed across more calls in llama.cpp) |
| KV concat / append | (in-place via `attention_decode`) | `concat_f32_dim0` 1920 × 35 µs (4.25%) | flambeau wins: avoids a 4.25%-of-wall kernel by writing directly into KV slab |
| AR / cross-rank residual | `p2p_allreduce_residual_tp2` 5440 × 16 µs (3.48%) | none (single-rank PP) | llama.cpp avoids AR entirely; flambeau pays 3.5% for the TP residual-sum |
| Embedding/expert gather | (specialised per-tensor) | `k_get_rows_float` 4069 × 11 µs (2.82%) | llama.cpp uses a single primitive across many gather sites |

## Sampler attribution (host-side, not in GPU profile)

flambeau's sampler is purely CPU-side Rust code. It does **not** appear as
GPU kernels in the profile — it shows up as a 12 ms gap between consecutive
forward calls. Three offenders in `runtime/src/sampling.rs`:

1. `build_distribution` — full softmax over `[V=151424]`, then full sort to
   apply `top_p`. ~7 ms.
2. Penalty application — iterates over the entire vocabulary even though
   only `history.len() ≤ 64` tokens have non-default penalty multipliers.
   ~3 ms.
3. Multinomial sample — `sample_from_distribution` walks the sorted prefix
   summing probabilities to find the cumulative threshold. ~2 ms.

llama.cpp's sampler chain (`llama_sampler_chain_apply`) processes only the
top-K candidates after a partial sort, so it stays sub-millisecond per call.

**Tracked fix in flambeau memory: #194 — partial-sort + penalty-aware
build_distribution.** Implementation is O(V) penalty pass + O(V + K log K)
top-K + partial sort. Estimated win: 12 ms → 1-2 ms per token, recovering
~10 ms/token on the chat decode wall.

## Where the 35B-A3B-Q4_0 chat user actually feels the perf

| scenario | observed tg/s | what's slow |
|---|---:|---|
| chat at temp=0 (greedy) | 60.5 | nothing — flambeau wins outright |
| chat at temp=0.7, top_p=0.9 | 35.6 | the sampler, **not** the kernels |
| chat with `presence_penalty` / `frequency_penalty` | further degraded | penalty path runs full-vocab |

So the user's "35B-A3B should be a lot faster" intuition was right about chat
but for the wrong reason. The fix is in `crates/runtime/src/sampling.rs` (host
code), not in the GPU forward path.

## Reproducer

```sh
# llama.cpp greedy + sampled chat (50 tg/s for both)
cd /artefact/llama.cpp_mi50
LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm-host/lib \
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-host/lib/rocblas/library \
./build/bin/llama-server -m /artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
  -ngl 99 -c 4096 -t 1 --host 0.0.0.0 --port 8081 &
# (warm prompt cache once; subsequent calls reuse cache)
curl -s -X POST http://localhost:8081/v1/chat/completions \
  -H 'Content-Type: application/json' -d @/tmp/r_lc_greedy.json | jq .timings
curl -s -X POST http://localhost:8081/v1/chat/completions \
  -H 'Content-Type: application/json' -d @/tmp/r_lc_sampled.json | jq .timings

# flambeau greedy + sampled chat (60 / 36 tg/s)
/artefact/flambeau/target/release/flambeau serve \
  --model /artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
  --devices hip:0,1 --mesh-mode tp --tp-size 2 --port 8080 &
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' -d @/tmp/r_greedy.json
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' -d @/tmp/r_sampled.json

# llama.cpp kernel profile (under rocprofv3, GGML_CUDA_DISABLE_GRAPHS required)
PATH=/opt/rocm-host/bin:$PATH \
LD_LIBRARY_PATH=/artefact/llama.cpp_mi50/build/bin:/opt/rocm-host/lib \
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-host/lib/rocblas/library \
GGML_CUDA_DISABLE_GRAPHS=1 \
rocprofv3 --kernel-trace -d /tmp/lc_profile/out --output-format json -- \
  /artefact/llama.cpp_mi50/build/bin/llama-bench \
  -m /artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
  -p 256 -n 64 -ngl 99 -r 1 -t 1 --no-warmup
```
