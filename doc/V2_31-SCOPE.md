# V2.31 — MMQ_X ≥ 16 restructure for indexed-MoE MMQ (scope doc)

## Context

V2.27 head-to-head measured **Qwen3.6-35B-A3B-UD-Q4_K_S Mesh<4> prefill L=512 at 683 tok/s vs llama.cpp 873 = 78 %**. This is the residual gap on the Q4_K MoE path after V2.6.b tile8 + V2.5 r-family + V2.4 sort/pad landed. The gap's upper bound is the Q4_K MoE MMQ kernel's per-output-row throughput; V2.24.b NULL ruled out Y-LDS-staging at MMQ_X=8 as the lever. The only remaining structural lever is **widening MMQ_X**.

V2.14.d left an unexplored scope:

> **Paths forward** (not pursued in V2.14):
> 1. Widen V2.6.a sort to pad-to-16 and re-run turbo at MMQ_X=16. Doubles padding overhead (15 slots/expert × 256 experts = 3840 padding slots at L=512, vs 7×256=1792 today), unclear if it pays.

V2.31 re-opens Path 1 but with TWO candidate structures, not just the turbo port.

## Three candidate structures at MMQ_X=16

| Variant | Weight home | Activation home | Threads/block | LDS budget | Sync count | Precedent |
|---|---|---|---|---:|---:|---|
| **A. Turbo @ MMQ_X=16** | LDS tile | LDS tile | 256 (4 warps) | ~45 KiB | 4/super | V2.14 (null @ MMQ_X=8) |
| **B. Register-resident tile16** | Registers | L1-broadcast | 64 (1 wave) | 0 | 0 | V2.6.b pattern extended |
| **C. Split-warp tile16** | Registers | L1-broadcast | 128 (2 waves) | 0 | 1/super | No precedent |

Variant B is the cleanest extension of V2.6.b tile8 — same "weights in registers, Y from L1, 0 LDS, 0 syncs" design that already beat turbo at MMQ_X=8. The only knob changed is doubling the tile's N-dim to 16 slot-cols.

## Sort/pad overhead math

Current V2.6.a pad-to-8 at Qwen3.6-35B-A3B Mesh<4> (pp=512, n_experts=256, top_k=8):

- total_pairs = L × top_k = 512 × 8 = **4096**
- worst-case padded_total = total_pairs + n_experts × 7 = 4096 + 1792 = **5888**
- padded grid.y = 5888 / 8 = **736 blocks** of 8 slot-cols each
- padding ratio = 1792 / 5888 = **30.4 %** of issued grid.y work is redundant duplicate-writes

Pad-to-16 at same shape:

- worst-case padded_total = 4096 + 256 × 15 = 4096 + 3840 = **7936**
- grid.y = 7936 / 16 = **496 blocks** of 16 slot-cols each
- padding ratio = 3840 / 7936 = **48.4 %** of issued grid.y work is redundant

**Block-count drops 33 % (736 → 496)** but **redundant-fraction rises from 30 → 48 %**. Net per-block work goes 2× (tile16 vs tile8). Total GPU work multiplier:

- tile8:  736 × 8  = 5888 "slot-col work units" @ 30 % redundant → 4096 real + 1792 redundant
- tile16: 496 × 16 = 7936 "slot-col work units" @ 48 % redundant → 4096 real + 3840 redundant

Total GPU slot-col work: **+35 % at tile16** vs tile8 just from the padding shift. For tile16 to beat tile8 on wall-clock, the per-slot-col work must get **≥ 35 % cheaper**.

## Expected per-slot-col speedup from MMQ_X=16

**Variant B (register-resident tile16)**: weight per thread stays register-resident. Each row's 32 F16 decoded weights are shared across 16 activation cols instead of 8 — 2× the reuse. But activation is L1-hot for both (all 64 threads in a wave read the same Y line), so that's already ~free. The only saving is:

- **Weight HBM read amortisation**: each thread reads its row's 32 F16 weights once per super-block iter; used 16× instead of 8×. Halves weight-side HBM bandwidth per output.
- **Launch overhead**: 736 → 496 blocks = 33 % fewer blocks. At HIP's ~0.2 µs/block scheduling overhead and the typical 40-layer × 4-rank × 3 MoE matmuls = 480 launches / prefill, saving ~3.2k blocks × 0.2 µs = 0.6 ms / prefill. Fraction of 735 ms prefill = 0.08 %. Negligible.
- **Register pressure**: `sums_gate[16] + sums_up[16]` = 32 F32 accumulators vs tile8's 16. +16 VGPR per thread. Still well under 128/wave gfx906 limit → no occupancy drop.

If the kernel is currently HBM-bound on the weight side, halving weight HBM gets close to 2× speedup. Minus the 35 % padding-overhead multiplier, net is ~2× / 1.35 = **+48 % per-kernel**, which would deliver ~780 tok/s on the 683 tok/s baseline (90 % of llama.cpp).

BUT: V2.6.b's MemBusy on this kernel was measured around 65 % at tile8 (rocprofv3 from V2.9.a notes). 35 % headroom exists, so doubling reuse might translate to ~1.5× rather than 2×. Net could be: ~1.5 / 1.35 = **+11 %**. That's 760 tok/s = 87 % of llama.cpp. Still positive but small.

**Variant A (turbo-style LDS @ MMQ_X=16)**: V2.14.d measured turbo with MMQ_X=8 at −17 %. Going to MMQ_X=16 restores turbo's design point ("MMQ_X ≥ 16 where LDS pays"). But pad-to-16 doubles the padding overhead → 35 % baseline cost before any perf math. At best, turbo @ MMQ_X=16 matches tile8 register-resident; structural cost of pad-to-16 likely wipes out any gain. Low expected ROI.

**Variant C (split-warp)**: 128 threads = 2 waves, one wave per 8 cols of the 16-wide tile. Adds 1 `__syncthreads()` per super-block. Unexplored on gfx906; the existing Q4_K wave64 (V2.3.b) shows single-wave register-resident is already near-peak. Likely no improvement over B, and sync cost is a regression risk.

## Recommendation: Variant B, one prototype, clear kill criterion

Go/no-go for V2.31.b prototype:

- **Go** if projected Variant B win ≥ 10 % end-to-end on 35B-A3B-UD-Q4_K_S prefill.
- **No-go** if microbench shows < 5 % per-kernel improvement at MMQ_X=16 vs MMQ_X=8 after pad-to-16 overhead accounted.

Projection band is **+11 % to +48 %** end-to-end; best case closes the gap from 78 % → 90 %, worst case still positive. **Go** — **single prototype** is worth ~1 session.

### Kill criterion for V2.31.b

Author `indexed_moe_mmq_q4_k_gate_up_tile16_dp4a.cu` as a direct clone of V2.6.b's tile8 kernel with `TILE_N=16` and doubled `sums_{gate,up}` arrays. Requires new V2.6.a variant (pad-to-16) — that's V2.31.b's first side-task. Then A/B vs tile8 on 35B-A3B-UD-Q4_K_S pp=512:

- **Per-kernel A/B**: if the tile16 kernel's rocprofv3 total ms is **≥ 10 % lower** than tile8, proceed to wire it as default (V2.31.c).
- **End-to-end A/B**: if pp=512 tok/s is **≥ 750 (+9.8 %)**, ship. Below that, file null with updated scope notes.

### Scope of V2.31.b (one session)

Five sub-steps:

1. `crates/ops/src/hip/moe.rs` + `moe_sort.rs`: add `moe_sort_by_expert_padded_16` (pad-to-16) and a `padded_offsets_16` scratch. Budget scratch +2× bytes vs pad-to-8 (padded_total ceiling doubles).
2. `crates/kernels-hip/src/kernels/indexed_moe_mmq_q4_k_gate_up_tile16_dp4a.cu`: clone tile8 kernel with `TILE_N=16`. VGPR hit: ~16 more FP32 accumulators. Estimated 48 → 64 VGPR → 8 waves/SIMD (gfx906 tier drops from 10). Acceptable unless the memory-latency hiding fails.
3. Dispatch guard in `forward_moe_ffn_prefill`: tile16 only at `n_tokens × top_k ≥ 32` (ensures at least 2 tile16 blocks/expert, else pad-to-16 waste dominates). Below that threshold, fall back to tile8.
4. Cert via new sweep shape (or reuse existing — inner arithmetic is byte-identical).
5. A/B: record Qwen3.6-35B-A3B-UD-Q4_K_S Mesh<4> pp=512 and pp=1024 tok/s, vs V2.30 baseline (697). Decision: ship / null-with-diagnosis.

### V2.31.c (conditional, separate session)

IF V2.31.b green:

1. Port tile16 to `indexed_moe_mmq_q4_k_down_tile8_dp4a.cu` sibling.
2. Port to Q6_K: `indexed_moe_mmq_q6_k_down_tile16_dp4a.cu`.
3. Port to the V2.28.c Q4_0 tile8 family (since V2.30 proved MoE MMQ matters at scale).
4. Dispatch wiring for all three, sort-by-expert-padded_16 shared.
5. End-to-end A/B on all three affected models (35B-UD-Q4_K_S, 35B-A3B-Q4_0, 35B-UD-Q8_K_XL if it fires).
6. Update `certs/perf/v2_27_head_to_head.md`.

## Budget estimates

- LDS: 0 (Variant B is register-resident)
- VGPR: ~64/thread (tier 8 on gfx906, down from 10 at tile8's ~48). Confirmed acceptable for register-resident kernels (V2.3.b wave64 Q4_K at 64 VGPR ships).
- Sort scratch: padded_total upper bound doubles (`total_pairs + n_experts × 15`). At worst case L=1024 × top_k=8 × n_experts=256 = 8192 + 3840 = 12032 i32s = 48 KiB. Flambeau's `ShardedForwardPrefillScratch` already budgets 16 KiB for padded sort; bump to 64 KiB.
- Kernel author time: ~90 min (tile8 → tile16 is a mechanical edit + padded-sort variant is another ~60 min).
- A/B time: ~30 min (build, run, measure, parity-regress).

Total V2.31.b session: **~3 hours active**. V2.31.c (if green): another session, similar budget.

## Risk register

1. **Register spill at VGPR > 64** (tier-5/8 boundary on gfx906). If the compiler spills the `sums[16] + sums_up[16]` arrays to scratch, performance cliffs. V2.9.b saw the inverse pattern: `__launch_bounds__(X, 2)` forced spill on the gate_up kernel's 48 FP32 accumulators. Mitigation: ship with explicit `__launch_bounds__(WARP_SIZE, 1)` and measure VGPR post-compile.
2. **Padding overhead exceeds projection**. If expert-selection entropy is higher than assumed (uniform 1/256 across 4096 pairs), padding-per-expert bunches up and the 48 % redundant-fraction is optimistic. Mitigation: A/B measures real overhead not projection.
3. **MoE entropy makes MMQ_X=16 worse than tile8 even at register-resident parity**. V2.14.d's lesson is "per-shape kernel selection matters, whole-family rewrites rarely win uniformly." V2.31.b might need shape-aware dispatch (tile8 at small total_pairs, tile16 at ≥ 32).

## What V2.31 is NOT

- Not a full turbo port. Turbo's LDS-double-buffer design doesn't fit the MoE indexed model's expert-alignment constraints even at MMQ_X=16. V2.14.d established this.
- Not a Y-LDS redesign. V2.24.b NULL established that Y is L1-hot in our MoE tile kernels; LDS-staging Y is a strict loss.
- Not a dispatch-table rewrite. V2.6.a's padded-sort infrastructure stays; V2.31 adds a pad-to-16 variant that feeds tile16, keeps pad-to-8 for tile8.

## Decision: V2.31.b ships as Variant B prototype next session

Entry gates:
- Variant B prototype kernel exists + compiles.
- pad-to-16 sort + offsets kernel exist + compile.
- Parity smoke on synthetic shape matches tile8 within F32 noise.

Ship gate:
- per-kernel rocprofv3 ms ≥ 10 % lower than tile8.
- end-to-end 35B-UD-Q4_K_S Mesh<4> pp=512 ≥ 750 tok/s (vs V2.30 697).

Kill gate:
- end-to-end < 5 % improvement OR parity-bit-exact regression.
