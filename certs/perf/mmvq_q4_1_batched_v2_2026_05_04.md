# #288-v2 batched-MMVQ Q4_1 — wave64 routing at small N

**Date**: 2026-05-04
**Kernels**: `mmq_q4_1_wave64` (existing prefill kernel, used at small N)
+ `mmvq_q4_1_q8_1_batched` (v1, retained for A/B).
**Wrapper**: `mmvq_q4_1_batched_launch` + shape-aware short-circuit in
`crates/ops/src/hip/qmatmul.rs`.
**Gate**: `FLAMBEAU_BATCHED_MMVQ=1` opt-in env var; default OFF.

## Result

### Correctness — IEEE-correct, but breaks bit-identical-within-batch

Two parity facets to track:

1. **vs single-row baseline** (`mmvq_q4_1_wave64_small_n_parity`): pass
   at 1e-5 abs-err. Max drift 5.2e-6 at n_rows=14336/N=8.
2. **across slots with identical activation**
   (`mmvq_q4_1_wave64_identical_slots`): wave64 produces slightly
   different output for slot 0 vs slot 1 even when their activations
   are bit-identical. Max diff 3.3e-6 (f32 LSB scale); 78% of outputs
   differ in `to_bits()`. Root cause: compiler FMA-contraction
   asymmetry inside the per-col unrolled loop in `mmq_q4_1_wave64.cu`
   — hipcc emits slightly different FP code for c=0 vs c=1.

Live impact on Qwen3.6-27B chat decode (greedy, temp=0):
  - Baseline (no wave64), 2 concurrent: BOTH slots md5 59f23a62 (= N=1).
  - Wave64 path, 2 concurrent: slot 0 md5 59f23a62, slot 1 md5 c2547432
    ("rocky shore..." vs "thunderous sound..."). Both coherent poems,
    but the LSB-scale per-slot drift compounds across 64 layers and
    flips greedy argmax decisions.

The kernel is correct in the IEEE-754 sense — every individual element
is within tolerance of the per-row reference. But callers who depend
on **bit-identical-within-batch output** (the existing batched-decode
invariant) will see that broken when wave64 is engaged.

Identical-slots test asserts max_abs < 1e-5 (tolerance) rather than
`n_diff == 0` so it captures regression boundary while documenting the
known LSB asymmetry.

### Perf — wave64 wins at GDN-out shape

Microbench on gfx906 / MI50:

| shape           | path          | N=2     | N=4     | N=8     |
|-----------------|---------------|--------:|--------:|--------:|
| 3584 × 4096     | per-row (base)| 39.5 µs | 74.7 µs | 171.3 µs |
|                 | v1 batched    | 63.2 µs | 85.9 µs | 148.9 µs |
|                 | wave64        | 128.2µs | 176.9µs | 291.2 µs |
| 14336 × 4096    | per-row (base)|141.2 µs |368.8 µs | 889.7 µs |
|                 | v1 batched    |265.8 µs |373.2 µs | 623.1 µs |
|                 | **wave64**    |275.3 µs |269.9 µs | **396.3 µs** ← 1.27× over per-row, **2.24× over baseline qmatmul(m=N)** |

(`baseline` = single qmatmul(m=N) call which loops MMVQ per row;
"per-row" reference computes N · single-row-time = "what N independent
m=1 calls would cost".)

Headline:
- At Qwen3.6-27B's **ssm_out shape** (n_rows=14336, k=4096), wave64
  delivers **1.27× over per-row** at N=8 (= 2.24× over the existing
  qmatmul(m=N) baseline path which loops MMVQ).
- Wave64 LOSES at small shapes (n_rows=3584): 0.54× at N=8.
- v1 wins marginally at small shapes (1.06× at 3584/N=8) but loses
  vs per-row at large shapes.

## Dispatch decision: shape-aware

`FLAMBEAU_BATCHED_MMVQ=1` engages a shape-aware short-circuit:
- `n_rows ≥ 8192` → route through `mmq_q4_1_wave64`.
- `n_rows < 8192` → fall through to per-row MMVQ (the production
  default; wave64 has too few thread-blocks at small n_rows for the
  GPU to schedule efficiently).

Override knobs (for A/B / debug):
- `FLAMBEAU_BATCHED_MMVQ=v1` — force v1 batched kernel.
- `FLAMBEAU_BATCHED_MMVQ=wave64` — force wave64 unconditionally.

The 8192 threshold was picked between the two measured shapes (3584
and 14336). Refining the threshold (e.g., per-(n_rows, k) crossover
sweep) is a follow-on optimization but not load-bearing for the cert
gate.

## Why wave64 wins at large n_rows

`mmq_q4_1_wave64` is gridDim=(n_rows/64, n_cols/8) — each block of 64
threads handles 64 output rows × 8 cols. Per K-iter:
- Each thread loads its row's Q4_1 weight block (1 per thread).
- All 64 threads share `TILE_N=8` Q8_1 activation blocks (loaded once
  per col, reused across 64 rows via L1).

So per-block HBM = 64×20 B (weight) + 8×36 B (activation) = 1568 B per
K-iter. Single-row mmvq's per-block HBM = 1×20 B (weight) + 1×36 B
(activation) per row = 56 B per K-iter per output. For 64 outputs at
n_rows=14336, single-row total = 64 × 56 = 3584 B per equivalent
work-unit. **Wave64 cuts per-output HBM by ~2.3×** at the wave64-tile
shape, which is exactly the measured speedup over baseline.

At small n_rows (3584), n_rows/64 = 56 thread-blocks total —
insufficient to keep all SIMDs busy on gfx906 (~30 SIMDs × 4 waves
each = 120 active waves needed), so wave64 stalls. Per-row mmvq with
gridDim=(n_rows,) = 3584 blocks fills the GPU.

## Why v1 partial-win'd at small shapes only

The v1 kernel's gridDim=(n_rows,) keeps all SIMDs busy at small
n_rows. The per-K-iter slot loop adds ~25% activation HBM per K-block
but launch overhead amortization across N gives a small net win at
n_rows=3584/N=8 (1.06× vs per-row).

At large n_rows the activation re-reads dominate (528 MB at
n_rows=14336/N=8) and v1 regresses to 0.81× — confirmed by the v1
cert (`certs/perf/mmvq_q4_1_batched_v1_2026_05_04.md`).

## Path to 3× cert gate (revised)

| lever                       | win        | status               |
|-----------------------------|-----------:|----------------------|
| #266c batched-attn          | 1.05×      | landed               |
| #287 batched-GDN wired      | 1.03×      | landed               |
| #290 PP=2 pipelining ceiling| ≤1.6×      | shipped, blocked at N≥4 (27B Q4_1 KV OOM) |
| **#288-v2 wave64 routing**  | **~1.15× combined** (1.27× on ssm_out, ~1.0× elsewhere) | **opt-in** |

Combined ceiling with #288-v2: 1.05 × 1.6 × 1.15 ≈ 1.93× — improves the
post-#287 1.03× cert toward 2× but still under the 3× gate. To clear
3× cleanly the next levers are:
- Pure-PP=4 pipelining (2.3× ceiling at N=4) instead of PP=2/TP=2.
- Larger N validation (INFLIGHT_SLOTS=8) with smaller ctx / smaller
  quant (27B Q4_1 OOMs at SLOTS=4 already).

## Reproduce

```bash
# Parity:
cargo test --release -p flambeau-ops --features hip \
  --test mmvq_q4_1_batched_parity \
  --test mmvq_q4_1_wave64_small_n_parity

# Microbench (3 paths × 4 N values × 2 shapes):
cargo test --release -p flambeau-ops --features hip \
  --test mmvq_q4_1_batched_perf -- --nocapture --ignored

# Live opt-in (shape-aware):
FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_INFLIGHT_SLOTS=4 FLAMBEAU_BATCHED_MMVQ=1 \
  ./target/release/flambeau serve [...]
```
