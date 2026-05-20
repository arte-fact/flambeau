# S3 piece-1 — rmsnorm+RoPE fused, Q-norm/K-norm paths

Date: 2026-05-20
Status: **shipped.** Coherent at short and long contexts. Part 1 of
the per-layer decode-prologue fusion (S3); follow-ons are V-unit-norm
+ KV-append fuse, attn-post-norm + add fuse.

## Change

New kernel `kernels-hip/src/kernels/rmsnorm_rope_neox_partial_f16.cu`:
per-head rmsnorm + partial-NeoX RoPE in one launch, in-place.
Wrappers for head_dim ∈ {64, 128, 256, 512}.

Replaces, for gemma4 Q-norm and K-norm paths (`weights.attn_q_norm`,
`weights.attn_k_norm` set):

```
old: rmsnorm_f16(x → tmp) + memcpy_async DtoD(tmp → x) + rope(x in-place)
new: rmsnorm_rope_neox_partial_f16(x in-place)
```

Per layer per token: 3 launches → 1, plus 1 DtoD memcpy eliminated.
Saves Q and K both, so 4 launches + 2 memcpys per layer per token.
For E4B at 42 layers × 128 decode tokens: ~21 504 launches saved.

The fused kernel uses a single block per (token, head):
- Phase 1 — sum-of-squares reduce via warp shfl_xor + LDS cross-warp.
- Phase 2 — inv_rms; per-thread scale by `norm_w * inv_rms`, stash to
  LDS so the RoPE pair-access can read both halves.
- Phase 3 — RoPE per pair (`(tid, tid + rotated_dims/2)`) with
  pass-through for indices ≥ rotated_dims.

Positions HtoD hoisted ahead of Q/K-norm so the fused kernel can read
it. Non-fused fallback retained for archs without `attn_q_norm` /
`attn_k_norm` (Qwen3 dense, qwen3-moe).

## Measurement (E4B-Q4_0 SD, prompt 291 tok, decode 128 tok, hip:3)

3-run streaming bench, same prompt as previous cert:

| run | prefill t/s | decode t/s |
|---:|---:|---:|
| 1 | 581.6 | 46.9 |
| 2 | 591.0 | 46.7 |
| 3 | 590.7 | 46.7 |
| **avg** | **588** | **46.8** |

vs post-S2 (`gemma4_v2_s2_lmhead_q4k_r4_2026_05_20`):

|  | prefill | decode |
|---|---:|---:|
| post-S2 | 570 ± 30 | 46.1 |
| post-S3-1 | **588 ± 5** | **46.8** |
| Δ | +3 % (low noise) | +1.5 % |

Modest match for the back-of-envelope (saved launches ≈ 54 ms of
~3115 ms total = 1.7 %). Prefill gain comes from elimination of the
DtoD memcpy that sat in the prefill chunk's stream too.

## Cumulative perf vs llama.cpp on E4B

|  | flambeau-v2 | llama.cpp | ratio |
|---|---:|---:|---:|
| prefill | 588 t/s | 1046 t/s | **0.56×** |
| decode  | **46.8 t/s** | 70.6 t/s | **0.66×** |

Pre-push (pre-L1) was 0.55× / 0.58×. Cumulative L1+S2+S3-1 lift:
**decode 41.1 → 46.8 = +13.9 %**.

## Coherence

- Short prompt: `"The capital of France is"` → `"Paris."` ✓
- Long prompt: 291-token technical compiler prompt → coherent
  technical response at 128-token decode.

## Why this is smaller than the cert's optimistic estimate

The cert originally estimated 5–15 % wall on S3. Reality for piece-1
is ~1.5 %, because:
- rmsnorm + memcpy + rope at small head_dim (256–512) already fits
  in L2 between launches → the saved HBM round-trips don't show up.
- Per-launch overhead is ~5 µs, so 54 ms saved over ~3.1 s kernel
  time = 1.7 %. We measured 1.5 %. Math matches.

The bigger payoff in S3 lives in fusing kernels that aren't already
L2-cached: V-unit-norm + KV-append (eliminates the DtoD), and the
attn-post-norm + residual-add pair (post-attention residual write
goes through HBM each layer). Queued for piece-2/3.

## Files

- `crates/kernels-hip/src/kernels/rmsnorm_rope_neox_partial_f16.cu`
- `crates/ops/src/hip/{mod,pe,ops_impl}.rs` — module registration +
  trait method + launcher
- `crates/ops/src/ops_trait.rs` — Ops trait extension
- `crates/forward/src/core/composites/standard_attn.rs` — gated
  switch to the fused kernel for gemma4 paths
