# MTP-4-A — F32 residual additions: null lever

**Status:** clean null result. F32 residual additions in the MTP block
produce **identical acceptance** (9/16, 56.2%) to F16 residual
additions. The hypothesis was that F16's narrow exponent was
clipping precision in the residual `h0 + attn_proj` and
`h1 + mlp_out` adds; the experiment disproved it.

## Why it didn't help (post-hoc)

`crates/kernels-hip/src/kernels/add_f16.cu:23` reads:
```c
// Up-cast to F32 for the sum so F16 denormals / subtracts don't bite.
const float av = (float) a[i];
const float bv = (float) b[i];
y[i] = (fb_fp16_t) (av + bv);
```

The existing `add_f16` already internally up-casts to F32 for the
addition; only the final cast to F16 introduces precision loss. And
the magnitudes after MTP residuals are all in the F16-representable
range (max ~16 abs value), so the F32→F16 cast loses only ~1e-3
relative per element — far below what would shift argmax decisions.

The residual additions are not where MTP precision is being lost.

## Where the precision actually goes (revised hypothesis)

After ruling out residuals, the dominant precision-loss source in
MTP forward is most likely the **per-matmul Q8_1 activation
quantize**. Each of the 8 mmvq call sites runs:

```
F16 input → quantize_f16_q8_1 → Q8_1 (per-32-element block scale)
                                  → mmvq → F32 output → cast F16
```

Q8_1's per-block-of-32 scaling means a single large element in a
block dominates the per-block scale; smaller elements in that
block lose ~1/127 ≈ 0.78% relative precision. With 8 sequential
quant+matmul ops, cumulative noise ≈ √8 × 0.78% ≈ 2.2% per element.
Argmax flips become possible at the LM head where two nearby
candidates are within ~3% of each other.

vLLM/sglang's BF16-throughout path skips the activation quantize
entirely — F16/BF16 weight × F16/BF16 activation matmul, no
intermediate int8. That's the actual precision delta.

Fixing it for MTP requires either:
- **MTP-4-B (F32 hidden)** — keep activation in F32 through the
  mmvq using existing `mmvq` with F32 → Q8_1 only at edges. This
  partially helps if the per-block-Q8_1 noise comes from F16-cast
  amplitudes.
- **MTP-4-C (BF16 throughout)** — new BF16 kernel variants for
  the entire chain so matmul inputs stay BF16 (no Q8_1 in the
  middle). This is the actual fix.

In other words: the precision drag isn't where I expected. To
clear the 75% gate we likely need MTP-4-C, not MTP-4-B.

## Code state

- `crates/kernels-hip/src/kernels/add_f32.cu` — new tiny kernel.
- `crates/ops/src/hip/mlp.rs` — `add_f32` wrapper.
- `crates/ops/src/hip/mod.rs` — `"add_f32"` added to KERNEL_STEMS.
- `crates/models/qwen3-moe/src/mtp.rs::forward_mtp_step` — uses
  `add_f32` for the two residuals (h0+attn_proj, h1+down) instead
  of `add_f16` + cast.

The kernel + wrapper land in tree as a primitive; harmless and
useful for any future F32-residual lever even though it didn't move
acceptance.

## Decision

Mark MTP-4-A complete (null). Next attempt should be MTP-4-C
(BF16) directly — MTP-4-B's expected gain is bounded by the same
per-block-Q8_1 quantization that MTP-4-A failed to alleviate.
Hop straight to BF16 if we want to clear the strict 75% gate.

Or, more pragmatic: stop chasing the strict gate and ship MTP-5
at 56.2% for the practical K=2-3 throughput win.

## Sources

- [`add_f16.cu:23`](https://github.com/artefacts-musique/flambeau/blob/main/crates/kernels-hip/src/kernels/add_f16.cu) — prior F32 up-cast inside F16 add
- [vLLM `qwen3_next.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next.py) — BF16 throughout
