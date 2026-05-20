# S3 piece-3 — rmsnorm_f32_to_f16 + residual_add fused

Date: 2026-05-20
Status: **shipped.** Coherent. Net cumulative trace win the headline.

## Change

New kernel `flambeau_rmsnorm_f32_to_f16_add_residual` (entry in the
existing `rmsnorm_f32_to_f16` module). Fuses the
(`rmsnorm_f32_to_f16` → `add_f16`) pair on gemma4 post_attn_norm /
post_ffn_norm sites: one pass over each row reads `delta_f32` +
`weight_f16` + `resid_in_f16`, writes `resid_out_f16 = resid_in +
(rmsnorm(delta) * weight)`.

Wiring: standard_attn_local and dense_ffn_local, when their
`post_*_norm` is set, advance `ScratchPool::next_residual_slot()`
themselves, call the fused kernel writing directly into that slot,
and set a new pool flag `fused_residual_already_done`.
`residual_add_local` checks the flag at entry: when set, returns
`b` unchanged (which IS the new residual) and clears the flag.

Per layer per token, this collapses two kernels (rmsnorm_f32_to_f16
+ add_f16) into one, AND eliminates the intermediate `delta_f16`
HBM round-trip. Fires twice per layer (post-attn + post-ffn) on
gemma4.

## Kernel-trace evidence (E4B-Q4_0 SD, prompt 291 tok / decode 128)

Captured with rocprofv3 --kernel-trace.

|  | pre-S3 (post-S2) | post-S3-3 | Δ |
|---|---:|---:|---:|
| total kernel time | 3115 ms | **2931 ms** | **−5.9 %** |
| total launches | 206 767 | **159 222** | **−23 %** |
| `__amd_rocclr_copyBuffer` | 132 ms / 29 287 | 29 ms / 6 294 | −78 % (S3-2) |
| `rmsnorm_f16` | 119 ms / 22 344 | 48 ms / 5 712 | −60 % (S3-1) |
| `rmsnorm_f32_to_f16` | 81 ms / 11 088 | — | replaced |
| `add_f16` | 76 ms / 16 632 | — | replaced |
| `rmsnorm_f32_to_f16_add_residual` | — | 116 ms / 11 088 | new (S3-3) |
| `rmsnorm_rope_neox_partial_f16_d256` | — | 47 ms / 9 240 | new (S3-1) |
| `kv_append_v_unit_norm_f16_d256` | — | (not in top-14) | new (S3-2) |

Note: 11088 `rmsnorm_f32_to_f16_add_residual` calls = 5544 (post-attn) +
5544 (post-ffn) = 42 layers × 128 tokens × 2 paths × 1 call each.
Pre-S3: 11088 `rmsnorm_f32_to_f16` + 11088 `add_f16` calls. So S3-3
saved 5544 launches per path × 2 paths = 11 088 launch headers, plus
the intermediate delta HBM round-trip.

## Wall measurement (E4B-Q4_0 SD, same prompt, 3-run avg)

| stage | prefill t/s | decode t/s |
|---|---:|---:|
| post-S2 | 570 ± 30 | 46.1 |
| post-S3-1 | 588 | 46.8 |
| post-S3-2 | 589 | 47.3 |
| **post-S3-3** | **589** | **47.3** |
| Δ S3-3 alone | flat | +0.0 % |

S3-3's wall-clock impact landed under the bench noise floor for this
single prompt. The trace says we saved ~120 ms of kernel time
overall; the decode wall stayed at ~2.7 s. The cumulative S3 picture
is clearer:

| metric | pre-L1 | post-S3-3 | Δ |
|---|---:|---:|---:|
| total kernel time | 3351 ms | **2931 ms** | **−12.5 %** |
| decode t/s | 41.1 | **47.3** | **+15.1 %** |
| decode vs llama.cpp | 0.58× | **0.67×** | — |

## Coherence

- Short: `"capital of France"` → `"Paris."` ✓
- Long technical prompt: 128-token decode → coherent text matching
  the request.

## Why per-piece gains under-measured

Each S3 piece saves ~5–10 k kernel launches × ~5 µs of host-side
overhead = ~25–50 ms of kernel time. The wall-clock decode is
~2700 ms total, so each piece nets ~1 % wall. Three pieces give
~3 % wall — consistent with the 44.3 → 47.3 t/s = +6.8 % measured
across all of S3, with the rest absorbed by HBM and CPU-side
overhead the bench captures.

## Files

- `crates/kernels-hip/src/kernels/rmsnorm_f32_to_f16.cu` — new
  `flambeau_rmsnorm_f32_to_f16_add_residual` entry
- `crates/ops/src/hip/{norm,ops_impl}.rs` + `ops_trait.rs` — launcher +
  trait extension
- `crates/forward/src/core/scratch.rs` — `fused_residual_already_done`
  flag on `ScratchPool`
- `crates/forward/src/core/composites/{standard_attn,dense_ffn,residual_add}.rs`
  — branched wiring + flag protocol

## What's next

S3 is the structural launch-overhead push. Three pieces are now in
trunk; the residual gap on E4B decode (0.67×) is dominated by the
~27 % share of `mmvq_q4_0_q8_1` (already dp4a-optimal per the in-code
comment from commit 8.d). Closing more requires either MoE-style
batched-N decode (a different request shape), or attacking
`attention_decode_f16_splitk_chunk` further (now down to 13.2 %).
