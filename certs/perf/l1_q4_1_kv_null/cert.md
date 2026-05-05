# L1 — Q4_1 K+V fused MMVQ on gfx906 / pp2tp2 — NULL (2026-05-05)

Port of `mmvq_q4_0_kv_f16dst_dp4a` to Q4_1 weights. Kernel correct
(token-id bit-equal vs unfused fallback) but **null on wall** for the
two Q4_1 chat models. Reverted; one-line warning kept inline at
`crates/models/qwen3-moe/src/forward/attn_tp.rs:197`.

## Bench shape

- Topology: pp2tp2 on `hip:0,2,1,3` (matrix-bench standard).
- Features: `BATCHED_DECODE=1, GPU_SAMPLER=1, PREFIX_CACHE=1 (4 GiB),
  4 inflight slots, 32 max-queue depth, 512-token prefill ubatch`.
- Sampling: greedy (temperature=0, seed=42).
- Prompt: 4 062 tokens (matrix-bench long fixture). 512 decode tokens.
- Concurrent users: 1.

## Token parity (single-rank TP2 / 9B-Q4_1)

Fused output identical character-for-character to the unfused
fallback (`FLAMBEAU_KV_F16_DST=off`) at greedy temperature. ✓

## Kernel-time delta (rocprofv3 --kernel-trace, 1 warmup + 1 traced)

| model | kernel | calls baseline | calls fused | total ms baseline | total ms fused |
|---|---|---:|---:|---:|---:|
| 9B  | `mmvq_q4_1_t128`     | 28 224 | 24 192 | 633.8 | 594.8 |
| 9B  | `kv_f16dst` (new)    | —      | 2 016  | —      | 106.6 |
| 9B  | `cast_f32_f16`       | 22 704 | 18 672 | 119.6 | 104.5 |
| 9B  | **total**            |        |         | **6 288** | **6 353** (+1.0 %) |
| 27B | `mmvq_q4_1_t128`     | 56 448 | 48 384 | 1 648.9 | 1 579.6 |
| 27B | `kv_f16dst` (new)    | —      | 4 032  | —      | 202.7 |
| 27B | `cast_f32_f16`       | 45 408 | 37 344 | 243.2 | 213.0 |
| 27B | **total**            |        |         | **16 441** | **16 908** (+2.8 %) |

## Wall-clock A/B (matrix bench shape, fused vs FLAMBEAU_KV_F16_DST=off)

| model | unfused decode t/s | fused decode t/s | delta |
|---|---:|---:|---:|
| 9B  | 49.4 | 48.1 | **−2.6 %** |
| 27B | 23.3 | 22.9 | **−1.7 %** |

(Prefill numbers in the run log are corrupted by prefix-cache hits
between warmup and traced runs and are not included.)

## Diagnosis

The lever was supposed to win on three axes — none of them deliver:

1. **Activation re-read avoidance**: per fused call, the Q8_1 `y` is
   read once instead of twice. Total saving across the request: 8 MB.
   At 1 TB/s HBM = ~8 µs. Negligible.
2. **Launch overhead reduction**: 4 launches per layer-token-rank
   collapse to 1 launch. ~60 µs/pair × 2 016 fused pairs = 121 ms of
   CPU launch latency saved. But progressive dispatch (V1 inheritance
   from candle, see `feedback_hipgraph_null.md`) already overlaps
   launches with kernel execution; the saving doesn't reach the wall.
3. **F32 scratch round-trip**: `cast_f32_f16` calls drop ~18 % (visible
   in trace). Saves ~7 ms wall-equivalent. Real but small.

The structural reason: **weight HBM bytes are 99 % of K+V traffic and
they do not change with fusion**. Both K-tensor and V-tensor reads
still happen — the fused kernel just performs them inside one launch
with two F32 accumulators and two reductions. Per-call work roughly
doubles (52 µs vs 2×22 µs unfused), eating the launch saving.

## Don't re-attempt without

- A K/V quant whose weight bytes shrink (Q3_K K+V, Q2_K K+V).
- A redesign that lets K+V share the weight HBM stream (e.g. concat
  `[K|V]` into one tensor at GGUF-load time).
- A different rig where launch-overhead actually dominates (PCIe gen 2,
  no progressive dispatch).
