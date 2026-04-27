# CN-80B-6 (perf iter 3) — F16 router weight

## Profile pick (iter-2 intra-layer marks)

```
section            total_ms   count   mean   % wall
layer_attn_gdn       259.49    1152   0.225   31.2%
layer_moe_ffn        203.04    1536   0.132   24.4%
layer_router         123.94    1536   0.081   14.9%   ← iter-3 target
layer_attn_full       93.90     384   0.245   11.3%
layer_shared_expert   64.14    1536   0.042    7.7%
layer_post_norm       20.31    1536   0.013    2.4%
```

GDN attention (#1, 31.2 %) is the biggest hot-spot but its body is a
multi-kernel chain that needs further instrumentation (deferred to
iter-4 if useful, otherwise the iter-3+4 chain runs out of cheap
levers). Router is #3 at 14.9 %, runs in every layer (1 536 calls in
the bench), and is structurally a single F32 GEMV — clean kernel-
level lever in one session.

## Lever: F16 router weight

Router is F32 in every Qwen3.x GGUF. F16 halves per-row HBM bandwidth
on `dense_gemv_*` (2 B/elem vs 4 B/elem) at the cost of one
fp16→float per FMA. Quality impact is negligible — router is a
coarse top-k discriminator over discrete experts; a 1e-3 logit shift
cannot flip top-1 unless the choice was already a coin-flip.

Implementation:
- New kernel `dense_gemv_f16_f16.cu` + batched sibling
  `dense_gemv_f16_f16_batched.cu` (mirror of the F32 variants, only
  the weight load differs).
- `flambeau_ops::hip::router::{dense_gemv_f16_f16, dense_gemv_f16_f16_batched}`
  + REQUIRED_MODULES wired.
- PP loader (`sharded.rs::upload_ffn`) converts F32 `ffn_gate_inp` →
  F16 at upload via the existing `up_f16` helper. Other dtypes pass
  through unchanged.
- `forward_router_decode` and `forward_router_prefill` accept either
  F32 or F16 ffn_gate_inp and dispatch on the dtype.

## Verification

- F16 parity preserved: 35B-A3B prefill L=1 → 11 ✓, L=2 → 271 ✓
  (bit-exact vs llama.cpp reference; F16 router weight didn't even
  shift L=2's near-tie argmax).
- Build clean; `dense_gemv_f16_f16*` kernels compile under the
  existing build script.

## A/B bench — Coder-Next-80B pp4

| metric  | baseline | iter-3 (F16 router) | Δ |
|---------|---------:|--------------------:|---:|
| pp128   | 426.4    | 431.4               | **+1.2 %** |
| pp512   | 537.6    | 567.3               | **+5.5 %** |
| pp2048  | 555.6    | 597.7               | **+7.6 %** |
| tg64    |  41.3    |  41.1               | −0.5 % (noise) |

Win scales with prefill L: at L=2048 with 48 layers, the batched router
runs 48 × 2048 = 98 K logical row × token computations per pass — the
HBM weight read halving lands meaningfully. At decode L=1 the saving
per call is tiny so the variant change is wash (within rep variance).

**Keep.** Net positive (+5–8 % prefill), no regression on decode.

## Closes

- CN-80B-6 #132 — first real iter ship with measured perf delta.
  Iter-4 (#133) will either dig into GDN attn (the 31 %) or batch
  the iter-3 lever's siblings (e.g. F16 `topk_f32` precursor — small
  gain) depending on whether intra-GDN instrumentation lands cleanly.
