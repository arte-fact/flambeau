# V2.31.d — Q8_0 MMQ TILE_N=32 is a null

## Hypothesis

`mmq_q8_0_wave64_tile16` (V2.7, default) is 87 % of 27B prefill wall.
V2.7 measured MemUnitBusy ≈ 83 % at TILE_N=16 → still memory-bound.
Doubling to TILE_N=32 would halve weight HBM bandwidth again
(decoded tile served to 32 cols per thread instead of 16).

## Outcome: ~1 % regression

Qwen3.6-27B-Q8_0 Mesh<4> async ub=128, 100 W/GPU:

| L    | tile16 (default) | tile32 (q8_tile32) | Δ       |
|------|-----------------:|-------------------:|--------:|
|  128 |             98.4 |               97.4 | −1.0 %  |
|  512 |            147.2 |              157.8 | (noisy) |
| 1024 |            229.7 |              227.1 | −1.1 %  |
| 2048 |            284.3 |              281.2 | −1.1 %  |
| 4096 |            315.5 |              312.1 | −1.1 %  |
| 8192 |            318.3 |              315.2 | −1.0 %  |
| decode |          18.4 |               18.4 |  0 %    |

Parity: bit-exact — all last_ids match. Kernel is correct.

## Why it didn't work

Per-thread FP32 accumulator count doubles 16 → 32. Register pressure
grows enough that either:
1. Compiler scratch-spills — any VALU issue pressure on tile32's
   inner loop (not `#pragma unroll`-ed for this reason) still comes
   with loads from scratch.
2. Or the kernel is now actually compute-bound, not memory-bound.

V2.7's "MemBusy 83 %" at tile16 may have been the tail of the
memory-bound regime. At tile32 we hit the next envelope (compute-limited
DP4A issue rate) without further HBM savings to trade.

## Opt-in artifact

Kept as `FLAMBEAU_VARIANT=q8_tile32` for future research (if a
different shape family or a different compiler release moves the
crossover).

## Implication for 27B prefill

The 87 % of 27B prefill wall on one kernel is a structural ceiling
on gfx906 at this shape. To move further we'd need a different
kernel family — most promising candidate: port `mmq_q4_1_4warp_lds`
LDS-tiled structure to Q8_0 (MMQ_Y=128, MMQ_X=64, 4-warp
co-operative load per tile). Significant effort (~200 LOC new
kernel). Filed as potential V2.31.h follow-up.
