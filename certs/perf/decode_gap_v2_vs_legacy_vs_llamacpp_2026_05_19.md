# Decode-gap profiling: flambeau-v2 vs flambeau-legacy vs llama.cpp

Hardware: 2× MI50 (gfx906), TP=2 on hip:0,1. Prompt 287 tokens, 128 decode tokens.

## Wall-rate summary

### Qwen3.6-27B-Q4_0 (16 FullAttn + 48 GDN, dense FFN)

| stack                 | prefill (t/s) | decode (t/s) |
|-----------------------|--------------:|-------------:|
| **flambeau-v2 (TP2)** | **307**       | **23.3**     |
| flambeau-legacy (TP2) | 300           | 25.3         |
| llama.cpp `-sm row` (TP) | 214        | 21.9         |
| llama.cpp `-sm layer` (PP) | 200      | 20.8         |

v2 / legacy: prefill 1.02× / decode 0.92×.
v2 / llama.cpp `row`: prefill 1.43× / decode 1.06×.
v2 / llama.cpp `layer`: prefill 1.54× / decode 1.12×.

### Qwen3.6-35B-A3B-Q4_0 (16 FullAttn + 48 GDN, MoE FFN + shared expert)

| stack                 | prefill (t/s) | decode (t/s) |
|-----------------------|--------------:|-------------:|
| **flambeau-v2 (TP2)** | **952**       | **57.6**     |
| flambeau-legacy (TP2) | 841           | 67.6         |
| llama.cpp `-sm row` (TP) | 577        | 42.2         |
| llama.cpp `-sm layer` (PP) | 687      | 63.0         |

v2 / legacy: prefill 1.13× / decode 0.85×.
v2 / llama.cpp `row`: prefill 1.65× / decode 1.37×.
v2 / llama.cpp `layer`: prefill 1.39× / decode 0.91×.

## GPU kernel trace (decode-only, 128 tokens)

Sum across both TP ranks (so multiply per-rank by 2):

### Qwen3.6-27B

| metric                   | v2     | legacy | delta     |
|--------------------------|-------:|-------:|----------:|
| launches / token         | 3334   | 3014   | **+320**  |
| GPU kernel time (ms/tok) | 71.93  | 64.17  | **+7.75** |

Top per-kernel deltas (v2 minus legacy, 128 decode tokens):

| kernel                                  | v2 #   | leg #  | delta # | v2 ms | leg ms | delta ms |
|-----------------------------------------|-------:|-------:|--------:|------:|-------:|---------:|
| `mmvq_q4_0_q8_1` (single-row)           | 63 488 | 18 432 | +45 056 | 3 239 |    792 | **+2447** |
| `mmvq_q4_0_gate_up_t128_dp4a_q8_1` (fused) | 0  | 16 384 | -16 384 |     0 |   1616 |   -1616 |
| `mmvq_q4_0_kv_f16dst_dp4a_q8_1` (F16-out) | 0    |  4 096 |  -4 096 |     0 |    227 |    -227 |
| `mmvq_q4_0_q8_1_f16` (F16-out)          |     0  |  4 096 |  -4 096 |     0 |    159 |    -159 |
| `cast_f32_f16`                          | 45 056 | 32 768 | +12 288 |   168 |    110 |     +58 |
| `__amd_rocclr_copyBuffer`               | 33 024 | 25 088 |  +7 936 |   143 |     92 |     +51 |

### Qwen3.6-35B-A3B

| metric                   | v2     | legacy | delta     |
|--------------------------|-------:|-------:|----------:|
| launches / token         | 2926   | 2766   | **+160**  |
| GPU kernel time (ms/tok) | 34.20  | 27.27  | **+6.93** |

| kernel                                  | v2 #   | leg #  | delta # | v2 ms | leg ms | delta ms |
|-----------------------------------------|-------:|-------:|--------:|------:|-------:|---------:|
| `mmvq_q8_0_t128_vdr2_q8_1`              | 24 064 |  8 704 | +15 360 |   137 |     83 |     +54 |
| `quantize_row_f16_q8_1`                 | 35 840 | 23 040 | +12 800 |   126 |     77 |     +50 |
| `dense_gemv_f16_f16` (legacy F16 router) |  0    | 10 240 | -10 240 |     0 |     61 |     -61 |
| `moe_combine_no_residual_f32` (v2)      | 10 240 |  0    | +10 240 |    41 |      0 |     +41 |
| `moe_combine_no_residual_f16` (legacy)  |     0  | 10 240 | -10 240 |     0 |     45 |     -45 |
| `mmvq_q6_k_dp4a_q8_1`                   |  1 536 |    128 |  +1 408 |   230 |     97 |    +134 |

## HIP API trace (host-side, Qwen3.6-27B, 128 decode tokens)

| API                       | v2 calls | legacy # | delta # | v2 ms | leg ms | delta ms |
|---------------------------|---------:|---------:|--------:|------:|-------:|---------:|
| `hipStreamSynchronize`    | 35 056   | 1 901    | **+33 155** | **8 389** | 1 081 | **+7 308** |
| `hipMemcpyAsync`          | 36 198   | 28 301   | +7 897  | 12 232 |  6 205 | +6 027 |
| `hipStreamWaitEvent`      |       0  | 32 768   | -32 768 |     0 |    293 |    -293 |
| `hipEventRecord`          |       0  | 32 768   | -32 768 |     0 |    136 |    -136 |
| `hipModuleLaunchKernel`   | 404 132  | 371 010  | +33 122 |  1 513 |  1 338 |   +175 |

## Diagnosis — three distinct effects

**1. v2 picks slower mmvq kernel variants** (the dominant gap)

For 27B: legacy fires 16 384 `mmvq_q4_0_gate_up_t128_dp4a_q8_1` (the `_t128_dp4a` variant of the fused gate+up kernel) that v2 never dispatches to. Plus legacy uses F16-output variants (`mmvq_q4_0_kv_f16dst_dp4a_q8_1`, `mmvq_q4_0_q8_1_f16`) so the F32→F16 cast is integrated into the matmul. v2 fires the F32-output mmvq then explicit `cast_f32_f16` calls.

Accounts for: most of the +7.75 ms/tok GPU kernel time delta on 27B.

**2. v2 quantizes the router weight that legacy keeps F16**

35B: legacy converts the F32 `ffn_gate_inp` router weight to F16 at load time (`tp_target_dtype` in `qwen3-moe/src/tp_sharded.rs:518`) and fires `dense_gemv_f16_f16`. v2's `upload_quant_weight` host-quantizes F32→Q8_0 and dispatches a Q4_0/Q8_0 mmvq instead. Result: 10 240 extra `dense_gemv_f16_f16` in legacy (a fast F16 GEMV) vs v2's slower `mmvq_q8_0` per-router-call.

**3. v2 uses host-blocking `hipStreamSynchronize` for cross-rank AR ordering; legacy uses HIP events**

v2's `bar_ar_residual_f16` / `bar_ar_residual_rmsnorm_f16` does `Stream::synchronize(stream)` on every rank before publishing the partial pointer (to drain producer writes). 33k extra syncs for 128 decode tokens. Legacy publishes producer-done as a `hipEventRecord` + waiter ranks `hipStreamWaitEvent` — GPU-side ordering, host never blocks.

This is a more subtle effect: host time spent in `hipStreamSynchronize` is NOT pure overhead — it's literally measuring GPU completion time (the wait). But it BLOCKS the host thread, preventing CPU/GPU overlap. Legacy's event-based ordering lets the host queue ahead.

The first two effects account for the bulk of the GPU kernel time delta. The third effect explains why the wall gap is sometimes smaller than the GPU-time delta (legacy CPU/GPU overlap is better but legacy GPU does less work).

## Closable levers

Ranked by tractability:

1. **F16-output mmvq dispatch** (27B-class win, ~5 ms/tok GPU). Add `mmvq_q4_0_kv_f16dst` / `mmvq_q4_0_q8_1_f16` / `_t128_dp4a` variants to v2's dispatch table for the Q4_0 attention output_proj and similar shapes that produce hidden-sized outputs.

2. **F16 router weight conversion** (35B-class win, ~1 ms/tok). Mirror legacy's `tp_target_dtype` for the v2 loader — when `ffn_gate_inp` source is F32, convert to F16 instead of Q8_0. Composite already supports F16 router via `qmatmul`. Pure loader change.

3. **Event-based AR ordering** (universal, removes ~30k host-block syncs / 128 decode tokens). Replace `Stream::synchronize` in `bar_ar_residual_*` with `HipEvent` record/wait so producer-stream ordering goes through GPU events instead of host blocking. The legacy `ar_residual_rmsnorm` in qwen3-moe shows the shape.
