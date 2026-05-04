# #288 batched-MMVQ Q4_1 v1 — partial win, gated opt-in

**Date**: 2026-05-04
**Kernel**: `mmvq_q4_1_q8_1_batched` in
`crates/kernels-hip/src/kernels/mmvq_q4_1_batched.cu`
**Wrapper**: `mmvq_q4_1_batched_launch` in
`crates/ops/src/hip/qmatmul.rs`
**Gate**: `FLAMBEAU_BATCHED_MMVQ=1` opt-in env var; default OFF.

## Goal

The `feedback_qmatmul_small_m_no_amortize` memory note documented that
`qmatmul(m=N, ...)` for Q4_0/Q4_1/Q5_0/Q5_1 below m=32 dispatches to a
per-row MMVQ loop, reading the full weight tile from HBM N times.
Combined with #266c batched-attention (1.05× contribution) and #290
pipelined-decode (1.6× ceiling at PP=2/N=4), batched-MMVQ projected
1.5–2× was the missing lever to clear the 3× cert gate.

## Result

### Correctness — PASS

11 parity cases sweep N ∈ {1,2,4,8} × n_rows ∈ {64, 4096} × k ∈ {1024, 4096}.
All cases below 1e-5 abs-err vs `qmatmul(m=1)` baseline (max 1.9e-6, f32
LSB scale). Cert: parity test in
`crates/ops/tests/mmvq_q4_1_batched_parity.rs`.

The kernel is NOT bit-exact at k≥4096 because surrounding slot-loop
code changes hipcc's FMA-contraction choices vs the single-row kernel.
Drift is at the same f32 LSB scale already accepted by the batched-GDN
cert. `#pragma clang fp contract(off)` was tested and made things
worse (single-row uses FMA, mine doesn't), so left at default.

### Perf — partial win, shape-dependent

Microbench (run with `cargo test --release -p flambeau-ops --features
hip --test mmvq_q4_1_batched_perf -- --nocapture --ignored`), 50 iters
post-3-warm-up, on gfx906 (MI50):

| shape           | N=1 single | N=2 batched      | N=4 batched      | N=8 batched      |
|-----------------|-----------:|-----------------:|-----------------:|-----------------:|
| 3584 × 4096     |    19.8 µs | 56.3 µs (0.70×)  | 76.6 µs (1.04×)  | 122.9 µs (1.29×) |
| 14336 × 4096    |    64.5 µs | 203.5 µs (0.63×) | 348.3 µs (0.74×) | 610.7 µs (0.84×) |

(speedup factor = N · single-row / batched; >1 means batched
amortizes; <1 means batched costs more than per-row × N).

Headlines:
- At Qwen3.6-27B's `qkv` shape (n_rows=3584, k=4096), N=8 wins 1.29×.
- At the `ssm_out` shape (n_rows=14336, k=4096), every N regresses
  (batched is slower than per-row).
- Below the design's 1.5–2× projection across the board.

## Diagnosis

VGPR profile via `llvm-objdump`: 44 VGPR, 39 SGPR, no scratch spills.
Comfortable budget; not register-bound.

The kernel grid is `(n_rows,)` — same as the single-row variant — and
the inner per-K-iter loop processes N slots' activations inside each
output-row block. **Each output-row block re-reads ALL N slots'
activations from HBM**:

- Per-thread per-K-block at N=8: 1 weight int (4 B) + N × 2 activation
  ints (64 B). Weight is 6% of per-iter HBM, activation is 94%.
- Total activation HBM scales as `n_rows × N × act_bytes_per_row`. At
  n_rows=14336, k=4096, N=8: ~528 MB activation reads — far exceeding
  the 35 MB of weight reads. Activation HBM dominates and the
  weight-amortization lever doesn't help.
- L2 (4 MB on gfx906) is too small to hold the activation strip
  re-fetched per row.

The kernel is activation-HBM-bound, not weight-HBM-bound, at small N.
The amortization design assumed weight HBM dominated.

### What the projected 1.5–2× would actually require

Row tiling: each block handles **R output rows × N slots**, weight
tile read once per (R, N) work-group, activation read once per N rows
× R outputs. R ≥ 4 likely needed to amortize activation reads.

This is the MMQ-style tile pattern (`mmq_q4_1_4warp_lds.cu`,
`mmq_q4_1_wave64.cu`) adapted to decode-friendly shapes. Non-trivial
re-design — the existing MMQ kernels target prefill (large m).

## Decision

The v1 kernel is correct (parity passes) but doesn't deliver the
projected win. Routing the batched-decode dispatch through it would
regress the production hot path at GDN-out shapes (the 14336 × 4096
case dominates wall time on Qwen3.6-27B).

Therefore the qmatmul short-circuit is **gated behind
`FLAMBEAU_BATCHED_MMVQ=1`** — opt-in only, default OFF.

The 3× cert gate remains open. The next lever is a row-tiled v2
batched-MMVQ (separate task) OR pivoting to the existing prefill
kernels at decode-N.

## Reproduce

```bash
# Parity (correctness):
cargo test --release -p flambeau-ops --features hip \
  --test mmvq_q4_1_batched_parity

# Perf microbench:
cargo test --release -p flambeau-ops --features hip \
  --test mmvq_q4_1_batched_perf -- --nocapture --ignored

# Live opt-in (when ready to re-cert end-to-end):
FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_INFLIGHT_SLOTS=4 FLAMBEAU_BATCHED_MMVQ=1 \
  ./target/release/flambeau serve [...]
```
