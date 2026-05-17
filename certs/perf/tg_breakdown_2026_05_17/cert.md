# tg breakdown: flambeau vs llama.cpp on qwen3.6-35B-A3B PP2 (2026-05-17)

rocprofv3 kernel-trace of pure-decode-dominated runs on
Qwen3.6-35B-A3B-Q4_0, PP2 layer split (hip:0,2). Tiny prompt + 128
decode tokens so prefill is negligible (~2 dispatches).

## Setup

- **flambeau**: `flambeau infer --max-tokens 128 --devices hip:0,2 --mesh-mode pp` on `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`
- **llama.cpp**: `llama-server -sm layer -ts 1,1 --flash-attn on` + single chat completion `max_tokens=128`, `HIP_VISIBLE_DEVICES=0,2`
- **profiler**: `/opt/rocm-host/bin/rocprofv3 --kernel-trace -f csv`
- **bench reference**: `certs/perf/bench_matrix_2026_05_17/cert.md` — pp2 decode flambeau 34.49 / llama.cpp 61.37 tok/s

## Top-level numbers

|  | flambeau | llama.cpp |
|---|---:|---:|
| total GPU kernel time | **1565.6 ms** | 2406.6 ms |
| total dispatches | 131 072 | 212 912 |
| dispatches / token | ~1 008 | ~1 638 |
| wall time (128 tok) | ~3 710 ms | ~2 090 ms |
| GPU-busy fraction | ~42 % | ~115 %¹ |
| **GPU-idle (wall - GPU)** | **~2 144 ms** | ~–315 ms |

¹ llama exceeds 100 % because the trace records both GPUs' kernels;
under `-sm layer` PP, the two stages overlap kernel execution → total
device-seconds > wall-seconds. Flambeau's PP does NOT overlap stages
(strict sequential).

## Bottom line

**Flambeau executes 35 % LESS GPU work than llama.cpp but is 78 %
slower end-to-end.** The decode bottleneck on this configuration is
NOT the GPU kernels — it is host-side latency (~2.1 s of wall time
where the GPU is idle waiting for the next launch / sync /
inter-stage hand-off).

This **invalidates** the earlier hypothesis (`bench_matrix_2026_05_17`)
that per-expert MMVQ launch count dominates. The MoE expert kernels
collectively account for ~30 % of GPU time but ~12 % of the wall-clock
gap.

## Top kernels — flambeau (% of GPU kernel time)

| % | count | kernel |
|---:|---:|---|
| 13.27 | 3 472 | `flambeau_indexed_moe_mmvq_q4_0_q8_1` |
| 9.19 | 2 980 | `flambeau_mmvq_q4_0_gate_up_dp4a_q8_1` |
| 7.24 | 3 972 | `flambeau_indexed_moe_mmvq_q4_0_gate_up_dp4a_q8_1` |
| 5.11 | 3 010 | `flambeau_gdn_state_step_alphabeta_f32_s128` |
| 4.94 | 993 | `flambeau_attention_decode_f16` |
| 4.95 | 105 | `flambeau_mmvq_q6_k_dp4a_q8_1` (LM head) |
| 4.42 | 14 052 | `flambeau_cast_f32_f16` (108 / token — high) |
| 4.35 | 6 015 | `flambeau_mmvq_q5_0_q8_1` |
| 4.26 | 4 012 | `flambeau_topk_softmax_f32` (router) |
| 3.75 | 4 073 | `flambeau_rmsnorm_q8_1_fused` |
| 3.48 | 10 924 | `flambeau_swiglu_f32_to_q8_1` |

## Top kernels — llama.cpp (% of GPU kernel time)

| % | count | kernel |
|---:|---:|---|
| 35.77 | 37 960 | `mul_mat_vec_q` (Q-quant GEMV — dominates) |
| 7.90 | 351 | `mul_mat_vec_q_moe` (fused MoE GEMV — note count!) |
| 7.56 | 18 338 | `mul_mat_vec_f` |
| 7.05 | 38 311 | `quantize_q8_1` |
| 6.16 | 29 180 | `k_bin_bcast` |
| 5.53 | 3 930 | `gated_delta_net_cuda` |
| 5.49 | 3 810 | `concat_f32_dim0` (KV append) |
| 4.23 | 5 161 | `topk_moe_cuda` |

## Key structural differences

1. **`mul_mat_vec_q_moe` fires only 351 times in llama** (vs flambeau's
   ~7 444 indexed-MoE MMVQ launches across 2 kernels) — llama batches
   all top-k experts per token into one launch. So the launch-count
   hypothesis IS true for llama's design — but **closing that gap on
   flambeau wouldn't move the needle** because GPU compute isn't the
   bottleneck; ~30 % of GPU time is already a fraction of the 58 %
   wall-time gap.

2. **Flambeau has 14 052 `cast_f32_f16` dispatches** (108 per token).
   At ~5 µs/launch host overhead, that's ~70 µs/token of pure launch
   overhead — small per token (negligible at our wall budget), but
   indicative of unfused compute kernels.

3. **llama's PP overlaps GPU 0 and GPU 1 stages** (per-device kernel
   time > wall). Flambeau's `forward_one_token_pp` is strictly
   sequential rank 0 → peer_copy → rank 1. Two MI50s sit idle 50 %
   of the time each.

## Recommended next levers (ranked)

1. **Overlap PP stages with double-buffering** — issue rank 1's
   kernels for token T while rank 0 processes token T+1. Llama's
   pipeline interleave gives them a free 2× from this alone. Risk:
   non-trivial driver restructure; KV-cache + peer-copy ordering
   needs careful event-based DAG. **Biggest lever** — estimated 1.5×
   decode if it works.

2. **Async peer_copy_via_host** — eliminate the DtoH-sync-HtoD-sync
   round-trip from the per-token critical path. Use
   `peer_copy_via_host_async_laned` (already exists in the code,
   per `backend-hip/src/cluster.rs:742`) on the decode path.

3. **Reduce dispatch count** — fuse `cast_f32_f16` into producers.
   At ~108 dispatches/token, an 8× fusion saves ~95 dispatches/token
   × 5 µs = ~475 µs/token = ~13 % of wall budget.

4. **Indexed batched-MoE-MMVQ at decode-1** — only ~30 % of GPU
   time, smaller payoff than 1-3. Worth doing AFTER closing the
   wall-time gap because then GPU compute starts mattering.

## Reproducer

```
# Flambeau:
OUT=/tmp/rocprof_fb_$(date +%s); mkdir -p $OUT
LD_LIBRARY_PATH=/opt/rocm-host/lib RUST_LOG=error \
  /opt/rocm-host/bin/rocprofv3 --kernel-trace -o trace -d $OUT -f csv -- \
  ./target/release/flambeau infer \
    --model /artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
    --prompt "Hi" --max-tokens 128 --devices hip:0,2 --mesh-mode pp

# llama.cpp (start under rocprofv3, then curl, then kill):
OUT=/tmp/rocprof_lc_$(date +%s); mkdir -p $OUT
LD_LIBRARY_PATH=/opt/rocm-host/lib:/artefact/llama.cpp/build-mi50/bin \
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-host/lib/rocblas/library \
HIP_VISIBLE_DEVICES=0,2 \
/opt/rocm-host/bin/rocprofv3 --kernel-trace -o trace -d $OUT -f csv -- \
  llama-server -m <model> --port 8080 -ngl 999 \
    --split-mode layer --tensor-split 1,1 --ctx-size 4096 \
    --flash-attn on --threads 8 --no-mmap

# Aggregate:
python3 /tmp/agg_kernels.py $OUT/trace_kernel_trace.csv
```

## Files (not committed; kept in /tmp during the session)

- Flambeau trace: `/tmp/rocprof_fb_1779007765/trace_kernel_trace.csv` (18 MB)
- llama.cpp trace: `/tmp/rocprof_lc_1779007837/trace_kernel_trace.csv` (78 MB)
- Aggregator: `/tmp/agg_kernels.py`
