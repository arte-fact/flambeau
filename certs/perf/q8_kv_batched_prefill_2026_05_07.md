# Q8 KV batched prefill — perf cert

- **Model:** Qwen3.6-27B-Q4_0 (GGUF)
- **Topology:** hybrid pp2tp2 (4× MI50, gfx906)
- **Date:** 2026-05-07
- **Workload:** 214-token prompt + 80-token greedy decode (temp=0, seed=0)

## Headline

| arm | wall | decode ms/tok |
|---|---:|---:|
| F16 KV (baseline) | 3.76 s | 47 |
| Q8 KV pre-fix (per-token prefill fallback) | 10.5 s | n/a (prefill-bound) |
| **Q8 KV post-fix (batched prefill + split-K decode)** | **3.72 s** | flat 36–40 across n_kv = 25 → 425 |

Q8 KV is **1 % faster than F16** on this workload, matching theory:
Q8 reads ~half the KV bytes during attention. Pre-fix the per-token
prefill fallback (`is_q8_kv()` gated bail in `forward_prefill_*`)
dominated wall; replacing it with a real batched Q8 prefill kernel
restores Q8's expected silicon advantage.

## Pre-fix bottleneck

`crates/models/qwen3-moe/src/forward/hybrid.rs` and `pp.rs` both gated
`is_q8_kv()` and routed Q8 prompts through the **per-token decode
loop** (`forward_one_token_*` × n_prompt). On a 214-token prompt this
paid 214 × ~25 ms ≈ 5–6 s of hidden prefill cost — roughly the entire
Q8↔F16 wall gap. Decode itself was already fast (the split-K kernel
landed in commit ff21e2d holds decode flat across n_kv = 25 → 425).

## Fix (this commit)

Three coordinated pieces:

1. **`attention_prefill_q8_kv` kernel** — port of the F16 oracle prefill
   (`attention_prefill_f16` in `crates/kernels-hip/src/kernels/`) to
   Q8_0 KV. Same control flow + causal mask; K/V dequant on the fly
   during the score and V-accumulate. Block = head_dim, grid =
   (n_q_tokens, n_heads_q). Handles any `n_q_tokens`. Flash-tile-Q8
   (BR=4/8 LDS-tiled) is a follow-up; this oracle path is enough to
   beat the per-token fallback by ~3×.

2. **Generic prefill drivers** — `forward_full_attn_prefill_tp` (TP /
   hybrid) and `forward_full_attn_prefill` (PP-only) made generic over
   `L: CacheLayout`. Append branches on layout: Q8 quantises directly
   into the cache slot via `compute_append_dsts(n_tokens) →
   quantize_f16_q8_0(n × kv_elems) × 2 → bump_tail(n_tokens)` (no
   staged DtoD memcpy). Attention branches on layout to dispatch
   `attention_prefill_q8_kv` when `L::NAME == Q8Contig::NAME`.
   `quantize_f16_q8_0` is already n_tokens-agnostic — its kernel
   partitions over `blockIdx.x` per 32-element block, so passing
   `n_tokens × kv_elems` and a `n_tokens × kv_elems / 32`-sized
   destination is the batched form for free.

3. **Drop `is_q8_kv()` gates** — both `forward_prefill_hybrid_logits`
   and `forward_prefill_pp_logits` no longer bail on Q8 caches. The
   batched-prefill chain handles `LayerCache::FullAttnQ8` end-to-end
   via the layout-generic prefill drivers.

## Validation

- **Bit-identical output** vs Q8 single-pass-only (n_q_tokens = 1
  through forward_one_token loop): SHA-256 = `5fcf53291435cf91e0797d09
  381b963ae08357ee79010b78d2eda8718bb27785` matches the
  `phase3_q8_splitk_check` capture from before this fix.
- **Decode rate flat** across n_kv = 25 → 425 (validated under the
  same harness in commit ff21e2d): 44 → 36 → 39 → 40 ms/tok.
- **No correctness regressions** on F16 path (same dispatch branch,
  unchanged behaviour when `L::NAME == F16Contig::NAME`).

## Out of scope (V2 follow-ups)

- **Flash-tile-Q8** — port BR=4/8 LDS-tiled prefill to Q8 KV. Oracle
  path here is single-warp-per-block per (q_token, q_head); flash-tile
  would batch across BR=4-8 query rows per block with cooperative
  K/V tile loads. Would close any residual silicon gap at large
  prompts (n_prompt ≫ 256), but not on this workload's critical path.
- **Multi-slot batched-decode Q8** — `tp.rs:2760` bail still rejects
  FullAttnQ8 in the continuous-batching multi-slot decode driver.
  Single-slot Q8 decode (this work) lands first; multi-slot
  P2.9b-Q8 is V2.
- **Graph-captured Q8 append** — `kv_cache_append_hip_slot` is F16-only;
  Q8 append currently bypasses graph capture. Adds a
  slot-aware-quantize launch when needed (no current consumer).

## Files touched

- `crates/kernels-hip/src/kernels/attention_prefill_q8_kv.cu` (new, 130 lines)
- `crates/ops/src/hip/{attention.rs,mod.rs}` (Rust wrapper + module list)
- `crates/models/qwen3-moe/src/forward/{attn.rs,attn_tp.rs}` (generic over L)
- `crates/models/qwen3-moe/src/forward/{hybrid.rs,pp.rs}` (drop gate)
- `crates/models/qwen3-moe/src/forward/layer.rs` (handle FullAttnQ8 cache)
- `crates/models/qwen3-moe/src/forward/tp.rs` (batched-prefill callsite)
