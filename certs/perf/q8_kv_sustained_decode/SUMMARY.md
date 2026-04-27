# V1-BENCH-#118 — Q8 KV sustained-decode bench (negative result)

**Verdict: Q8 KV is currently a perf regression at all tested ctx
sizes on this rig. Stay opt-in via `FLAMBEAU_KV=q8`; documented for
VRAM-bound use cases only.**

## Setup

- Model: Qwen3.6-35B-A3B-UD-Q4_K_S (4× MI50, 100 W cap, ROCm 7.1.1)
- Topology: pp4 (devices 0,1,2,3)
- Method: prefill `ctx_size` synthetic tokens, then measure decode
  wall-clock over 64 tg steps at position `ctx_size + 1 ..= ctx_size + 64`.
  Same model handle, same process — only `FLAMBEAU_KV` flips between
  runs at session ctor.

## Results — `forward_one_token_pp` decode tg/s (tg=64)

| ctx  | F16 tg/s | Q8 tg/s | Q8 / F16 |
|-----:|--------:|--------:|--------:|
|  512 |  59.53  |  43.11  |  0.72×  |
| 2048 |  45.15  |  21.69  |  0.48×  |
| 8192 |  43.52  |   7.70  |  **0.18×** |

The ratio worsens monotonically with ctx. F16 holds ~43 tg/s flat
between ctx 2 048 and 8 192 (decode is HBM-bound on the F16 KV fetch
but split-K bounds the wall); Q8 falls off a cliff (44 → 22 → 8 tg/s).

## Why Q8 KV loses today

Three compounding costs:

1. **No split-K Q8 attention kernel.** The F16 decode path at
   `n_tokens_kv > 256` routes to `attention_decode_f16_splitk` —
   V2.19.b measured ~7.78× speedup at ctx=2048 (340 µs vs 2 647 µs).
   Q8 has only the single-pass `attention_decode_q8_kv`. At ctx=8192
   the missing split-K is most of the gap.
2. **Per-step F16 → Q8_0 quantize launches.** Each full-attn layer
   adds 2 `quantize_f16_q8_0` launches per decoded token (K and V).
   On the 35B's 10 full-attn layers × 64 tg, that's 1 280 extra
   kernel launches per bench run.
3. **Inline dequant in the attention loop.** Each thread reads
   `k_block.d` and `v_block.d` from global memory per token, then
   dequantises before the dot-product / V-accum. The F16 path just
   reads two fp16s per token — no scale fetch, no decompress.

The HBM saving on the cache fetch side (Q8: 136 B vs F16: 256 B per
(token, kv-head) at head_dim=128, ~1.9× saving) does not overcome
these costs at any ctx point we tested.

## Where Q8 KV still wins

**VRAM footprint, not throughput.** Per-layer KV cache size scales
linearly with `bytes_per_row(head_dim)`, so a Q8 cache fits roughly
1.9× more ctx into the same VRAM budget than F16. For users who hit
"out of VRAM at ctx=N" with F16 KV, opting into Q8 lets them reach
~1.9 N before OOM — at the tg/s cost above.

## Path to making Q8 KV perf-positive (V2 follow-ups)

In rough priority order:

1. **`attention_decode_q8_kv_splitk` kernel.** Port the F16 split-K
   structure (chunk-then-combine over the KV axis) to the Q8 reader.
   Closes the bulk of the long-ctx gap. Estimate: 1–2 sessions.
2. **Fuse F16 → Q8_0 K/V quantize into the K/V projection epilogue.**
   The MMVQ kernel already writes F32 → cast_f32_f16. Adding a
   F32 → Q8_0 epilogue removes the standalone quantize launches.
   Estimate: 1 session.

## Already-tried levers — null results

3. **Cache per-block scales in LDS at attention-loop entry**
   (V1-BENCH-#118a, 2026-04-27). All 32 threads sharing a block read
   the same `block.d`; the prediction was ~halving global memory
   traffic on the cache-fetch side. Implemented + correctness sweep
   green. Measured Δ vs the pre-fix kernel: **null** (within ±3 %
   rep variance at every ctx point). Reverted because the extra
   `__syncthreads()` per token slightly hurts in some configs while
   buying nothing.

   Why null: the GPU's L1 cache absorbs the 32× redundant scale
   reads at near-zero cost. Global-memory pressure on the scale
   loads was never the bottleneck. The actual bottleneck remains
   the absence of split-K parallelism (item 1) and the per-step
   F16→Q8_0 K/V quantise launches (item 2).

After (1) + (2), Q8 KV should match or beat F16 on long-ctx decode.
Until then it stays opt-in for VRAM-bound configs.

## Cert files

- `certs/perf/q8_kv_sustained_decode/Qwen3.6-35B-A3B-UD-Q4_K_S.json` —
  raw timings.
- `certs/quality/Qwen3.6-35B-A3B-UD-Q4_K_S_q8_kv.json`,
  `certs/quality/Qwen3.5-9B-Q4_1_q8_kv.json` — V1-BENCH-#117 quality
  certs (PASS, mean KL ≤ 9e-4 on both).

## Closes

- #118 #108c — measured. Q8 KV stays opt-in; "promote to default" was
  always the wrong framing (rule #6); now also empirically wrong on
  raw throughput.
