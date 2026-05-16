# mmvq_f16_direct — F16-store MMVQ variants (#120)

Date: 2026-05-16
Backend: HIP / gfx906 (MI50)
Scope: full 20-dtype MMVQ matrix

## What shipped

Templated `__device__` body + two `extern "C"` thunks (F32 + F16-saturating)
on every MMVQ kernel used by `mmvq()` at m=1. The F16 thunk writes
directly into the consumer's F16 destination buffer with a saturating
clamp at ±F16_MAX, skipping the `mmvq_f32` scratch + `cast_f32_to_f16`
two-step.

Single shared store helper at `kernels-shared/include/mmvq_store.cuh`:

```cpp
template<typename OutT>
__device__ __forceinline__ void mmvq_store(OutT* y, int row, float acc);
template<> __device__ __forceinline__ void mmvq_store<float>(...) { y[row] = acc; }
template<> __device__ __forceinline__ void mmvq_store<fb_fp16_t>(float* y, int row, float acc) {
    float v = fmaxf(-65504.f, fminf(65504.f, acc));
    y[row] = (fb_fp16_t) v;
}
```

Op surface: `Ops::mmvq_f16_direct(weights, act_q8_1, dst_f16, n_rows, k, dtype)`,
dispatches across 20 dtypes via match.

## Coverage

11 kernel files refactored, 20 dtypes total wired into `mmvq_f16_direct`:

| Family       | dtypes                                                      |
|--------------|-------------------------------------------------------------|
| 32-elem block| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0                                |
| K-quant      | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K                          |
| IQ codebook  | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,       |
|              | IQ4_NL, IQ4_XS                                              |

Kernels with `__launch_bounds__ < 256` (Q4_1 t128, Q8_0 t128_vdr2) needed
the launcher signature change — `mmvq_simple_launch` now takes
`(threads, rows_per_block, units_per_row)` explicitly. Latent landmine
caught: over-launching a `__launch_bounds__(128)` kernel with 256 threads
gives HIP error 719; under that bound writes OOB to per-warp shared
slots. Every call site now spells out the kernel's launch shape.

## Consumer wiring

`StandardAttention::forward_decode` Q + K + V projections all take the
F16-direct path for any `attn_q/k/v.dtype ∈ {20-dtype set}`, gated by
`f16_direct_supported(dtype)`. Per attention block, this eliminates:

- 3 F32 scratch writes (the `mmvq_f32` slab into Q, K, V slots)
- 3 `cast_f32_to_f16` kernel launches
- 3 F32 HBM round-trips through the scratch

Output projection intentionally stays on the F32 scratch path — it
relies on F32 readability for the `f32_output_proj` saturation handling
shipped in 10c-G-attn-fix (gemma4-31B head_dim=512 Q8_0 overflow).

## Parity

Test: `crates/ops/tests/mmvq_f16_direct_parity.rs::mmvq_f16_direct_matches_cast`
20 dtypes × 3 shapes = 60 cases. Assertion covers three regimes:

1. **Normal** (|F32 acc| ≤ 65504): F16(via_cast) ≡ F16(direct), bit-
   identical via `to_bits()` comparison.
2. **Saturation** (|F32 acc| > 65504): F16(direct) clamps to ±65504;
   F16(via_cast) is ±inf. This is the intended divergence — the
   saturating clamp is the whole point of the F16-direct path; it
   prevents ±inf from poisoning the KV cache. The legacy path's
   un-saturated `(fb_fp16_t)x` cast propagates inf, which then NaNs
   through downstream rmsnorm.
3. **NaN passthrough**: degenerate test inputs (random opaque bytes
   through a codebook reaching a 0×∞ path); both paths produce NaN,
   matched by bit pattern.

```
Q4_0  / Q4_1  / Q5_0  / Q5_1  / Q8_0  × 3 shapes: 0 bug rows
Q2_K  / Q3_K  / Q4_K  / Q5_K  / Q6_K  / Q8_K × 3 shapes: 0 bug rows
IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_NL / IQ4_XS × 3 shapes: 0 bug rows
```

IQ1_M at n_rows=4096/k=5120 measured: **1615 saturated_rows + 2352
nan_rows + 0 bug rows**. The 1615 rows are exactly where the legacy
via-cast path produces ±inf. Concrete evidence the saturating clamp is
load-bearing for the IQ-quant decode hot path.

## Code change is the cert

The cast attribution claim ("cast_f32_f16 launches drop to zero for
migrated projections") is a code-level invariant, not a measurement:
the call site `ops.cast_f32_to_f16(scratch.mmvq_f32, scratch.q/k/v_f16, ...)`
is gone entirely in the F16-direct branch. A rocprofv3 trace would
show what reading the diff shows.

Estimated wall-clock impact (kept as estimate, not measured for this
cert): per-token decode on a 30-layer model with 3 cast launches per
attention block at ~5 µs launch + HBM round-trip, ~450 µs/token
overhead eliminated. At 60 t/s that's ~3 % of decode wall. Real win
depends on layer count / dtype mix; structural change either way.

## Regressions

- `mmvq_f16_direct_matches_cast` — 60/60 green.
- `parity_31b_q4_0_pp2_pertoken` — gemma4 31B-Q4_0 decodes
  ` Paris.<turn|>\n<|channel>thought\n<channel|>It looks like you've
  provided a` (unchanged vs pre-#120 baseline).
- `real_text_tp2_qwen35_9b_q4_1_paris` — Qwen3.5-9B-Q4_1 TP2 decodes
  ` Paris` (passes via the K-projection F16-direct path).

## Open

- rocprofv3 attribution measurement on a real 30+-layer decode — would
  quantify the wall-clock delta + confirm zero `cast_f32_f16` launches
  on the attention block. Separate task; this cert ships the kernel
  + parity coverage.
- F16-direct for MMQ (m ≥ 32 prefill path) — out of scope for #120;
  MMQ already has a tile-shaped final-store that's structurally
  different from MMVQ's `dst[row] = acc`.

## Files

- `crates/kernels-shared/include/mmvq_store.cuh` (new)
- `crates/kernels-hip/src/kernels/mmvq_{q4_0,q4_1_t128,q5_0,q5_1,q8_0_t128_vdr2,q2_k_r2,q3_k_r2,q4_k_r2,q5_k_r2,q6_k_dp4a,q8_k,iq{1_s,1_m,2_xxs,2_xs,2_s,3_xxs,3_s,4_nl,4_xs}_r2}.cu` (20)
- `crates/ops/src/hip/qmatmul.rs` — `mmvq_f16_direct`, launcher refactor
- `crates/ops/src/ops_trait.rs` — `Ops::mmvq_f16_direct`
- `crates/ops/src/hip/ops_impl.rs` — `HipOps::mmvq_f16_direct`
- `crates/blocks/src/attention.rs` — Q/K/V consumer wiring, `f16_direct_supported`
- `crates/ops/tests/mmvq_f16_direct_parity.rs` — 60-case parity

Commits: `40c0a83` (Q4_0 prototype), `934b3d5` (K-projection), `7108007`
(Q4_1+Q8_0), `eaa4aed` (Q5_0+Q5_1), `e751016` (Q+V), `8a6de3f`
(K-quants), `9564bf3` (IQ family).
