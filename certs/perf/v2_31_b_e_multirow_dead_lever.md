# V2.31.b / V2.31.e — multi-row MMVQ for decode is a dead lever on gfx906

## Hypothesis (from V2.30.b profile)

- 27B decode: Q8_0 MMVQ eats 81 % of wall (2871 ms of 3543 ms at tg=64).
  Single-row kernels fire ~21 k calls for 64 tokens → "launch-overhead
  bound, r2/r4 multi-row packing will save 25–40 %".
- Coder decode: Q4_K MMVQ (`mmvq_q4_k_r2`) is 30 % of wall — similar
  argument, port to r4.

## What we built

- `mmvq_q8_0_r4_dp4a.cu` — 64 threads / block, 4 rows / block, 16 lanes
  per row, VDR=2 DP4A inner.
- `mmvq_q4_k_r4.cu` — 64 threads / block, 4 rows / block, 16 lanes per
  row, 2 bytes / lane / sub-block.

Both behind `FLAMBEAU_VARIANT=q8_r4` / `q4_k_r4` opt-in guards (mainline
default unchanged).

## What went wrong

### Correctness: F32 drift, not a bug

Direct kernel-vs-kernel test (`crates/ops/tests/mmvq_q8_0_r4_diff.rs`):

| shape                            | result                                 |
|----------------------------------|----------------------------------------|
| n_rows=8, n_blocks=2             | bit-exact                              |
| n_rows=12288, n_blocks=160 (27B) | **89 % rows differ by ~1e-4 rel-err** |

r4 accumulates 16 blocks per lane (16-deep tree) before the 16-wide
warp reduce. vdr2 accumulates ~2 blocks per lane (shallow) then reduces
across 256 threads (wider). Different F32 addition order on the same
integer dp4a inputs → mantissa noise. At n_blocks=2 the tree is too
shallow to surface drift; at production shapes (n_blocks=160) drift
is ~1e-4 rel but argmax rank is sensitive to per-token drift multiplied
across 16 layers × 64 tokens. Each logit drifts a few ulps; argmax
position flips on close candidates.

This is **not a correctness bug** — a CPU reference with the same
accumulation order as r4 would also produce these values. Both are
within F32 tolerance (both deviate from F64 reference by similar
magnitudes).

### Perf: -6 % regression on 27B decode (the shipping case)

Measured (100 W/GPU, Mesh<4>):

| variant   | decode tg=64 tok/s |
|-----------|-------------------:|
| vdr2 (default) | **18.75**      |
| r4 (`q8_r4`)   | 17.58 (-6.2 %) |

r4 gives each row 4× less compute throughput than vdr2:
- vdr2: 4 waves compute 1 row = full wave64 × 4 concurrent waves per row
- r4: 1 wave computes 4 rows = 16 lanes per row = 0.25 waves per row

At 147 µs / call for `mmvq_q8_0_gate_up_dp4a`, the launch-overhead
portion is ~10–20 µs (7–14 %). Best-case r4 savings from 4× fewer
launches: ~10 % of call time. Actual: regressed 6 %. The
compute-bottleneck per row (saturated DP4A issue rate) dominates the
win.

## Context: V2.29.c already called this on Q4_1

From `certs/perf/v2_29_c_q4_1_mmvq_ab.md`:

| impl_id                              | decode tg=64 | Δ      |
|--------------------------------------|-------------:|-------:|
| `qmatmul_q4_1_mmvq_t128_gfx906` (def) | 54.97       | —      |
| `qmatmul_q4_1_mmvq_nw1_r2_gfx906`     |  24.05       | **−56 %** |
| `qmatmul_q4_1_mmvq_r2_dp4a_gfx906`    | 47.17       | −14 %  |

The "fewer threads × more rows" family is reliably worse for decode on
gfx906. Occupancy + compute-per-row matter more than launch count.

## Implication

**Multi-row MMVQ packing is NOT a viable lever for Q-kernel decode on
gfx906.** The 81 % Q8_0 MMVQ wall on 27B decode is compute-bound, not
launch-bound. To move it we'd need:
- Wider parallelism per row (but 256 t/row is already near-max for the
  dispatch shape)
- Per-DP4A fewer cycles (hardware limit)
- Smaller per-call work (reduce k — not a knob at inference time)
- Batched tokens per call (spec decoding, not a kernel lever)

**Shipped deliverables**:
- Both r4 kernels in-tree as `FLAMBEAU_VARIANT=q8_r4` / `q4_k_r4`
  opt-in A/B references — zero regression on defaults.
- `mmvq_q8_0_r4_diff` unit test in ops crate confirms kernel
  correctness envelope.
- This cert documents the null verdict for future audits.

## Ranked update to V2.31 batch

- V2.31.b — **NULL** (dead lever per above)
- V2.31.e — **NULL** (same class as b)
- V2.31.c/d — still skipped (VGPR risk, independent of above)
- V2.31.a — SHIP (+25–50 % Coder prefill)
- V2.31.f — NULL (9B Q4_1 MMQ at 100 W confirmed)
- V2.31.g — SHIP (+4–7 % 35B prefill)
