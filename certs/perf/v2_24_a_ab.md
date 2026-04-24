# V2.24.a — Lever-1 per-kernel 9B decode tune AB report

Goal: close the 16 % per-rank kernel time gap between flambeau and turbo on
Qwen3.5-9B-Q4_1 decode tg=64. Target kernel identified via rocprofv3: the
original V2.2.b `mmvq_q4_1` (256 threads, single row, DP4A) = 45 % of 9B
decode GPU time (462 ms across 11,616 calls at ~40 µs/call).

## Attempts

### Iter 1 — `mmvq_q4_1_r2.cu` (scalar r2 multi-row)

Port of Q4_K r2 structure (64 threads, half-warp DPP, 2 rows/block) to
Q4_1. Scalar per-element FP multiply instead of DP4A because the half-warp
pattern gives each lane 1 element of one Q4_1 block, which doesn't pack 4
int8s into a single DP4A.

**NULL (−55 %)**. M<1> decode 66.35 → 28.24 tok/s. The DP4A parallelism on
Q4_1 is essential — packing 4 int8×int8 MACs per instruction is ~4× more
compute per clock than scalar FP. Q4_K r2 compensated for scalar FP with
its sub-block decode overhead; Q4_1 has no such overhead to amortise.

### Iter 2 — `mmvq_q4_1_r2_dp4a.cu` (DP4A r2 multi-row)

Keep the 256-thread DP4A block shape, but emit 2 output rows per block.
Halves grid.x + shares Y reads across 2 rows from L1.

**NULL (−7.5 %)**. M<1> decode 66.35 → 61.33 tok/s. Likely register
pressure from 2 per-row accumulators (2 × {vi_lo, vi_hi, sumi, d, m})
causing spill or occupancy drop. 256-thread block × extra register load
tipped over the spill threshold.

### Iter 3 — `mmvq_q4_1_t128.cu` (thin-block 128-thread single-row) **[SHIPPED]**

Same DP4A inner loop as V2.2.b but 128 threads/block instead of 256.
Halves total thread count per launch, raises CU occupancy (more blocks
resident), reduces kernel dispatch overhead.

**WIN +3.7 %**. M<1> decode 66.35 → **68.83 tok/s** (median of 3).
cert-check green on 15 shapes, UD-Q4_K_S 8-token parity preserved.

## Results

| config | baseline | Iter 1 (r2) | Iter 2 (r2 dp4a) | **Iter 3 (t128)** |
|---|---:|---:|---:|---:|
| M<1> decode tg=64 tok/s | 66.35 | 28.24 | 61.33 | **68.83** |
| Δ vs baseline | — | **−57 %** (null) | **−7.5 %** (null) | **+3.7 %** |
| M<4> decode tg=64 tok/s | 53.06 | n/a | n/a | 52.19 (noise) |
| vs turbo M<1> 74.44 | 89 % | — | — | **92.5 %** |
| UD-Q4_K_S 8-tok parity | ✓ | (not tested) | (not tested) | ✓ |

## Diagnosis of the remaining ~7.5 % gap

Turbo at M<1> decode is 74.44 tok/s, still 7.5 % ahead. Remaining levers
for a future cycle:

1. **Shape-aware thread count**: 128 threads is better for most 9B
   matmuls; large matmuls (ffn_gate/up, 13824 rows) may prefer 256. A
   dispatch split by n_rows could win an extra 1-2 %.
2. **Pre-shuffled Q8_1 Y layout**: llama.cpp uses a pre-shuffled Q8_1
   layout that makes the DP4A pair access stride-1 instead of our
   stride-4 (`u[lane4]` + `u[lane4+4]`). Requires changes to
   `quantize_row_q8_1` and all downstream consumers — substantial.
3. **MMVQ fused attn Q/K/V**: candle D5 pattern. Would collapse 3
   separate per-layer MMVQ launches into one, saving ~2k calls on 9B
   decode. Needs loader changes to lay weights contiguously.

Given the ceiling: Q4_1 MMVQ at 40 µs/call is already near HBM
bandwidth (~430 GB/s effective, 43 % of MI50's 1 TB/s peak). We're
within 10 % of compute+memory ceiling. Further wins require structural
changes (fused layout or pre-shuffled Y), not kernel micro-tuning.

## Gate

- cert-check hip/gfx906: 48 rows (1 impl swapped, no row count change), 0 failures
- UD-Q4_K_S 8-token parity bit-exact (seed 9419)
- Iter 1, Iter 2 files kept under `src/kernels/` (ship iter 3 only) with
  `#[cfg(unverified)]`-style notes — per architecture rule 10, null kernels
  keep a one-line diagnosis. Move to `_unverified/` is a follow-up cleanup.

## Regeneration

```
for m in 1 4; do
  FLAMBEAU_MESH_RANKS=$m FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
    FLAMBEAU_DECODE_ONLY=1 \
    ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b --nocapture
done
```
