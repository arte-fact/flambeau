# mmvq_q4_0_gate_up_row_tile_batched — perf cert vs K5

Date: 2026-05-13
Backend: HIP / gfx906 (MI50, 100 W cap)
Kernel: `mmvq_q4_0_gate_up_row_tile_batched.cu`
Baseline: `mmvq_q4_0_gate_up_batched.cu` (K5; one row per block)

## Method

HIP event timing, 200 warmup + 1000 measured launches per shape, `--release`,
single device. Same Q4_0 weights + Q8_1 activations reused across both
kernels. Wall = `elapsed_ms / iters`. Repro:

```
cargo test --release -p flambeau-backend-hip \
  --test mmvq_q4_0_gate_up_row_tile_perf -- --ignored --nocapture
```

## Results

```
 n_rows |    k | N | K5(µs) | RT(µs) | K5/RT
--------+------+---+--------+--------+------
   7168 | 2304 | 2 | 101.26 |  76.44 | 1.32×
   7168 | 2304 | 3 | 149.42 |  96.28 | 1.55×
   7168 | 2304 | 4 | 199.41 | 110.63 | 1.80×
   4096 | 2048 | 2 |  51.41 |  38.05 | 1.35×
   4096 | 2048 | 4 |  98.23 |  56.20 | 1.75×
   4096 | 4096 | 2 |  80.58 |  69.61 | 1.16×
   4096 | 4096 | 4 | 144.60 |  99.31 | 1.46×
    128 | 2304 | 2 |   5.06 |   6.64 | 0.76×
    128 | 2304 | 4 |   7.02 |   8.13 | 0.86×
```

## Interpretation

Row-tile wins 1.16×–1.80× across all decode-shape rows. Win grows with N
because each LDS-staged activation strip amortises across more output
columns: at N=2 each strip is shared 4× across the 4-row tile; at N=4
the activation cache hit is 4× larger so reuse is denser, lifting the
win to 1.80× at the GDN-class shape (n_rows=7168, k=2304, N=4).

Pathological corner (n_rows=128) loses 0.76×–0.86× — too few output
rows to amortise the LDS-staging overhead; activation already fits L2
at that scale so the row-tile's value vanishes. Practically not an
issue: real GDN gate/up rows are in the thousands. If we ever wire a
shape predicate, gate on `n_rows >= 1024` or so.

## Parity

`tests/mmvq_q4_0_gate_up_row_tile_batched.rs`: 5/5 green.
Max relative error 1.0e-4 vs row-tile-baseline (K5) F32 reduction
noise tolerance 2.8e-3. Includes K=512 (exactly 1 outer iter, bit-equal)
and K=800 (partial-tail 9-block outer iter, F32 noise only).

## Closes

Task #23 (T2.13 Wire K5 + bench) kernel side. Caller-side wiring
(replacing per-slot GDN gate+up loops in `forward_decode_batched_*`
with a single fused row-tile launch) is a separate driver change and
remains a follow-up; current MoE-decode and m=1 GDN paths are
unaffected by this commit.
