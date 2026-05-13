# mmvq_q5_k_row_tile_batched — perf cert vs K3

Date: 2026-05-13
Backend: HIP / gfx906 (MI50, 100 W cap)
Kernel: `mmvq_q5_k_row_tile_batched.cu`
Baseline: `mmvq_q5_k_r2_batched.cu` (K3; 2 rows per block via half-warp)

## Method

HIP event timing, 200 warmup + 1000 measured launches per shape,
`--release`, single device. Same Q5_K weights + Q8_1 activations reused
across both kernels. Repro:

```
cargo test --release -p flambeau-backend-hip \
  --test mmvq_q5_k_row_tile_perf -- --ignored --nocapture
```

## Results

```
 n_rows |    k | N | K3(µs) | RT(µs) | K3/RT
--------+------+---+--------+--------+------
   5120 | 5120 | 2 | 283.80 | 165.78 | 1.71×
   5120 | 5120 | 3 | 374.88 | 250.34 | 1.50×
   5120 | 5120 | 4 | 469.16 | 225.80 | 2.08×
   6912 | 5120 | 2 | 351.80 | 220.86 | 1.59×
   6912 | 5120 | 4 | 644.16 | 302.73 | 2.13×
   4096 | 4096 | 2 | 200.39 | 109.68 | 1.83×
   4096 | 4096 | 4 | 329.44 | 149.84 | 2.20×
    128 | 2048 | 2 |  25.30 |  11.30 | 2.24×
    128 | 2048 | 4 |  43.06 |  16.04 | 2.68×
```

## Interpretation

Row-tile wins 1.50×–2.68× across every shape, including the small-row
corner that Q4_0 lost. Q5_K is plain F32 FMA (no dp4a), so per-byte
activation re-reads were the dominant cost in K3's per-`(b, s, c)`
inner loop — each Q8_1 byte was HBM-fetched once per super-block s per
slot per row pair. Row-tile stages the full super-block strip (8 Q8_1
sub-blocks × N slots ≈ 1.2 KB at N=4) into LDS once per outer iter, then
reuses it across 8 rows × 8 sub-blocks × N slots.

At decode-class shapes (n_rows ∈ {5120, 6912}, k=5120, N=4) the win
is 2.08×–2.13×, the cleanest end of the lever the memory note
`feedback_mmvq_batched_activation_hbm` predicted (1.5–2× from a real
row-tile rewrite vs the per-N inner-loop approach).

Even small n_rows (128) wins 2.24×–2.68× because Q5_K's per-element
arithmetic is heavier than Q4_0's dp4a, so the LDS stage overhead is
amortized faster relative to compute.

## Parity

`tests/mmvq_q5_k_row_tile_batched.rs`: 4/4 green, **all bit-equal**
(max rel err 0.0). Q5_K uses plain F32 FMA in identical (b, s)
accumulation order in both kernels; only the activation-load path
changed. Includes a non-multiple-of-8 row count (n_rows=13) to exercise
the partial-tail block.

## Closes

Task #19 (T2.9 Q5_K batched MMVQ row-tile rewrite). Caller-side wiring
(replacing existing `mmvq_q5_k_r2_batched` calls in `qmatmul`'s small-m
fast path at `qmatmul.rs:137`) is a separate one-line dispatch swap and
sits behind this cert, ready for the next session.
