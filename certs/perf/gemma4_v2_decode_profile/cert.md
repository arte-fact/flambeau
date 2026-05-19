# gemma4-v2 26B-A4B MoE — measured decode-gap profile vs llama.cpp

Date: 2026-05-19
Model: `gemma-4-26B-A4B-it-Q8_0.gguf`
Dims: hidden=2816, head_count=16, head_kv=8, head_dim=176,
expert_count=128, experts_per_tok=8, expert_ffn=704, 30 layers
Mesh: flambeau PP2 hip:0,1; llama.cpp PP2 hip:0,1 (`--split-mode layer`)
Workload: warmup + 128-token decode at temp=0
Profiler: `rocprofv3 --kernel-trace --stats --summary -d <out>`,
SIGTERM-shutdown via the tokio signal handler added to `serve_common.rs`.

## How the profiler now works

Both stacks were profiled in-process under `rocprofv3`. Two prereqs
were needed to make this work on this rig:

1. **Graceful shutdown on SIGTERM.** flambeau's `run_axum` now wires
   `axum::serve(...).with_graceful_shutdown(shutdown_signal())` where
   `shutdown_signal()` awaits either SIGINT or SIGTERM via
   `tokio::signal::unix`. Without this, the rocprofv3 wrapper got SIGINT
   on container shutdown and never finalised its output. SIGTERM now
   triggers a clean tokio runtime exit → rocprofv3 atexit fires →
   SQLite DB + stats summary are written.
2. **Wrapper LD_LIBRARY_PATH points at `/opt/rocm-host`.** The
   rocprofiler-sdk-tool library lives there; `/opt/rocm-7.1.1/lib` only
   has the register shim. Setting `LD_LIBRARY_PATH=/opt/rocm-host/lib`
   when invoking rocprofv3 lets the wrapped binary find the tool.

## Aggregate kernel time

Both runs were the same workload (warmup + 128-token decode-only
greedy chat), profiled end-to-end.

| stack       | total kernel time | top-1 kernel %       |
|-------------|------------------:|---------------------:|
| flambeau-v2 | 3754 ms           | 16.3 % (prefill mmq) |
| llama.cpp   | 2719 ms           | 35.3 % (mmvq_q8_0_1) |

Total v2 GPU work is **1.38× llama.cpp's**. That tracks the measured
end-to-end decode gap (0.65× = 1.54× wall) almost exactly — what's
slower at decode is GPU work, not host scheduling.

## Top kernel comparison (decode-dominant kernels)

| flambeau-v2 kernel                          | calls  | ms      | % of total |
|----------------------------------------------|-------:|--------:|-----------:|
| `flambeau_attention_decode_f16`              |  4 020 | **575** | **15.3**   |
| `flambeau_indexed_moe_mmvq_q8_0_dp4a_q8_1`   | 12 060 | **586** | **15.6**   |
| `flambeau_rmsnorm_f32`                       | 12 240 |     366 |     9.7    |
| `flambeau_mmvq_q8_0_t128_vdr2_q8_1`          |  8 176 |     300 |     8.0    |
| `flambeau_mmvq_q8_0_t128_vdr2_q8_1_f16`      | 11 390 |     202 |     5.4    |
| `flambeau_rmsnorm_f16`                       | 24 540 |     174 |     4.6    |
| `flambeau_mmvq_q8_0_gate_up_t128_vdr2_q8_1`  |  4 020 |     140 |     3.7    |
| **per-token attention_decode_f16 cost**      |        | **143 µs**|          |
| **per-token MoE MMVQ launches × 3 mats**     |        | **147 µs**|          |

| llama.cpp kernel                          | calls  | ms      | % of total |
|--------------------------------------------|-------:|--------:|-----------:|
| `mul_mat_vec_q<Q8_0, 1>`                   | 31 644 | **959** | **35.3**   |
| `flash_attn_tile<256, 256, 1, 2>`          |  3 350 | **241** |  **8.9**   |
| `mul_mat_vec_q_moe<Q8_0, 2>`               |    232 | **176** |  **6.5**   |
| `quantize_q8_1`                            | 36 708 |     162 |     5.9    |
| `mul_mat_vec_q<Q8_0, 1, true>`             |  4 024 |     155 |     5.7    |
| `rms_norm_f32<1024, true>`                 | 17 070 |     111 |     4.1    |
| `mul_mat_q<Q8_0, 32>`                      |    202 |     101 |     3.7    |
| `rms_norm_f32<1024, true, true>`           | 12 456 |      90 |     3.3    |
| **per-token flash_attn_tile<256,256> cost**|        |  **72 µs**|          |
| **per-token mul_mat_vec_q_moe**            |        |  **52 µs**|          |

## Where the 8 ms/tok gap actually lives

### Lever A — attention decode is 2× slower per call (biggest gap)

- v2 `attention_decode_f16`: **143 µs/call**, 4 020 calls, **575 ms**
- llama.cpp `flash_attn_tile<256,256,1,2>`: **72 µs/call**, 3 350 calls,
  **241 ms**

For head_dim=176 (between 128 and 256), llama.cpp uses the d=256 tile
kernel — padded but well-tuned. flambeau's decode-attn path is twice
as slow per call. This costs ~**2.4 ms/tok**.

### Lever B — MoE expert dispatch is 52× more launches (3.3× more time)

- v2 `indexed_moe_mmvq_q8_0_dp4a_q8_1`: **12 060 calls**, **586 ms**
  (49 µs avg)
- llama.cpp `mul_mat_vec_q_moe<Q8_0, 2>`: **232 calls**, **176 ms**
  (760 µs avg)

llama.cpp emits ~one MoE MMVQ launch per layer that handles ALL
{gate, up, down} × 8 experts in one shot. flambeau's indexed_moe_mmvq
batches across the 8 experts but still emits three launches per layer
(gate, up, down). Over 30 layers × ~134 decode tokens = 12 060 launches.
This costs ~**3.4 ms/tok**.

### Lever C — RMSNorm: many small launches, 2× slower

- v2 rmsnorm family (`f32 + f16 + q8_1_fused + f32_to_f16`): **655 ms**
  across 45 016 launches
- llama.cpp rms_norm family (multiple template instantiations):
  **~263 ms** across 42 124 launches

Call counts are similar; per-call time is 2× slower. Same launch
volume = same launch overhead, but the kernel itself is doing more
work per call (likely an unfused path that llama.cpp folds into the
matmul). This costs ~**3 ms/tok**.

### Levers D – F (smaller)

- D. `__amd_rocclr_copyBuffer` (26 031 calls, 119 ms) — 6.5 D2D
  copies per layer in v2. Many small. Possibly redundant — audit
  the buffer flow at attention output + MoE down-write.
- E. `flambeau_rmsnorm_f16` (24 540 calls, 174 ms) — ~6 per layer.
  Some redundant?
- F. `flambeau_quantize_row_f16_q8_1` (20 460 calls, 103 ms) — 5
  per layer; fewer would mean re-using Q8_1-quantised activations
  across consumers.

## Sum check

A + B + C alone ≈ 2.4 + 3.4 + 3 = **8.8 ms/tok**. Matches the measured
8.1 ms/tok gap. The 8 ms/tok decode gap is dominated by these three.

## Priority levers (ordered by ROI)

1. **Fused MoE MMVQ {gate, up, down} → one launch per layer** —
   biggest single win. ~3 ms/tok. Existing kernel
   `flambeau_indexed_moe_mmvq_q8_0_dp4a_q8_1` already batches across
   experts; extend it to emit all three projection outputs in one
   dispatch, mirroring llama.cpp's `mul_mat_vec_q_moe`. Or at minimum
   fuse gate+up like the dense path already does
   (`mmvq_q8_0_gate_up_t128_vdr2_q8_1`).
2. **Per-call attention decode cost cut to llama.cpp parity** — port
   the `flash_attn_tile<d=256>` shape to flambeau's decode path or
   audit the current kernel for tile size / wave occupancy on head_dim
   in {64, 128, 256} with padding for non-power-of-2. ~2 ms/tok.
3. **RMSNorm + downstream fusion** — fold the post-FF norms with
   their consumer (cast/quantize/matmul). ~1–2 ms/tok.
4. **D2D copy audit** — 26k buffer copies for 134 decode tokens is
   ~200 copies/token. Look for redundant scratch shuffles. ~0.5 ms/tok.

Hitting (1) alone closes the gap to ~0.85× of llama.cpp; (1) + (2)
reaches parity; (1) + (2) + (3) puts flambeau ahead.

## Files

- Raw kernel summaries: `flambeau_v2_kernels.txt`,
  `llamacpp_kernels.txt`, `flambeau_v2_post_lever_b_kernels.txt`
- rocprofv3 SQLite DBs (not committed):
  `/tmp/rocprof_v2/flambeau_v2_results.db`,
  `/tmp/rocprof_llamacpp/llamacpp_results.db`

## Lever B shipped (Q8_0 indexed-MoE gate+up fusion)

Ported `indexed_moe_mmvq_q4_0_gate_up_dp4a` to Q8_0:
`flambeau_indexed_moe_mmvq_q8_0_gate_up_dp4a_q8_1`. Same per-block
shape (256 threads, VDR=2 DP4A), reads the Q8_1 activation int32
words once and accumulates `gate` and `up` partial sums in parallel.
Halves the MoE MMVQ launch count for the gate+up branch (`down`
remains a separate launch — needs a 3-way fused kernel for full
llama.cpp parity).

Measured on the same 128-token Q8_0 decode workload:

| metric                              | before        | after (lever B) | delta    |
|-------------------------------------|--------------:|----------------:|---------:|
| `indexed_moe_mmvq_q8_0_*` launches  | 12 060        | 4 020 + 4 020   | -33 %    |
| MoE MMVQ GPU time                   | 586 ms        | 574 ms          | -2 %     |
| Total kernel GPU time               | 3754 ms       | 3499 ms         | -6.8 %   |
| Decode t/s (bench)                  | 43.21         | **44.85**       | +3.8 %   |
| v2 / llama.cpp decode ratio         | 0.65×         | **0.68×**       | +3 pp    |

The launch-count cut and the HBM saving on activation re-reads both
contribute. Remaining gap is dominated by lever A (attention decode
2× slower) and lever C (rmsnorm 2× per-call).

Next: A. attention_decode_f16 tile-size audit / port to
`flash_attn_tile<d=256>` shape.

## Lever A shipped (splitk_chunk tile-2 inner loop)

The 184 µs/call cost in `attention_decode_f16_splitk_chunk` is dominated
by the per-token sync overhead: each context-token iter does one
`__syncthreads()` for cross-warp reduce + one `__syncthreads()` between
iters. At chunk_size=128 that's 256 syncs per block.

Rewrote the inner loop to process two K/V tokens per iter:

- Both Q · K[t] and Q · K[t+1] dot products done in parallel
- Warp-reduce both via paired `__shfl_xor`
- Cross-warp reduce uses `score_parts[ATTN_SK_MAX_WARPS * 2]` — write
  both partials before a single `__syncthreads`
- Online-softmax update with both scores: `new_max = max(running, a, b)`,
  `out_shared[tid] = out_shared * scale_old + coeff_a*V[t] + coeff_b*V[t+1]`,
  `running_sum = running_sum * scale_old + coeff_a + coeff_b`
- Odd tail handled by a single-token loop after

Bench numbers (gemma-4-26B-A4B PP2, 725-tok prompt, 128-tok decode):

| metric                              | post-B (single)| post-A (tile-2) | delta    |
|-------------------------------------|---------------:|----------------:|---------:|
| splitk_chunk time                   | 739.4 ms       | **435.3 ms**    | **-41 %**|
| splitk_chunk per-call cost          | 184 µs         | **108 µs**      | **-41 %**|
| Total kernel GPU time               | 4117 ms        | 3713 ms         | -10 %    |
| Decode t/s (bench)                  | 44.85          | **49.53**       | +10.4 %  |
| v2 / llama.cpp decode ratio         | 0.68×          | **0.82×**       | +14 pp   |

Same chat output (coherence preserved). The single-pass kernel
(`attention_decode_f16`) was not modified — only the splitk_chunk path
which the long-prompt bench hits. The single-pass kernel only runs for
n_tokens_kv ≤ 256; if that path also needs the tile-2 pattern (short-
context bench shapes), it's a follow-up port.

llama.cpp's `flash_attn_tile<256,256>` is still ~1.5× faster per call
than our tile-2 splitk. The remaining gap likely lives in K/V cache
coalescing or in GQA-aware q_head batching (n_heads_q=16 sharing
n_heads_kv=8 means 2 q_heads per kv_head — each kv_head row is
currently read twice).

## Lever C shipped (rmsnorm_f32 float4 vectorisation)

`flambeau_rmsnorm_f32` at decode launches a single block per token-layer
(n_rows = 1, hidden = 2816 F32 elements). Per-call cost was 30 µs vs
llama.cpp's templated `rms_norm_f32<1024>` at ~6 µs/call — 5× per-call
gap.

Rewrote both phases (sum-of-squares + scale) to use `float4` (16-byte)
loads/stores when the row pointers are 16-byte aligned and k is a
multiple of 4. gemma4 hidden=2816, qwen hidden=2304/3072/5120 all
satisfy the alignment. Same launch shape, same correctness contract.

Measured:

| metric                              | post-A (tile-2)| post-C (float4) | delta    |
|-------------------------------------|---------------:|----------------:|---------:|
| rmsnorm_f32 time                    | 366 ms         | **69 ms**       | **-81 %**|
| rmsnorm_f32 per-call cost           | 30 µs          | **5.6 µs**      | **-81 %**|
| Decode t/s (bench)                  | 49.53          | 49.90           | +0.7 %   |

The kernel is 5.4× faster but the bench-level delta is noise-bound
(0.7 % ≈ run-to-run variance). The 297 ms saved at GPU is real but
only ~1 ms/decode-token, masked by other variance.

A parallel `rmsnorm_f16` vectorisation (uint4 packing 8 F16 elems)
was attempted and reverted — it caused a 22 % prefill regression at
n_rows = 725 (probably VGPR-budget driven occupancy drop on the
prefill grid). The F16 kernel is left as-is; tackling it cleanly
needs the kind of `<BLOCK_SIZE, K>` template-specialisation pattern
llama.cpp uses, which is a separate session.
