# V2.31 — levers batch from V2.30.b profile tour

Six levers were enqueued based on V2.30.b's kernel-level attribution.
Session summary:

| ID      | Model     | Scope                                             | Result                                  |
|---------|-----------|---------------------------------------------------|-----------------------------------------|
| V2.31.a | Coder-30B | `indexed_moe_mmq_q5_k_down_tile8_dp4a`            | **+28–50 % prefill (shipped)**          |
| V2.31.b | 27B       | `mmvq_q8_0_r4_dp4a` (decode, 4 rows/block)        | **PARKED — kernel correctness bug**     |
| V2.31.c | 35B       | MoE Q4_K tile16                                   | **SKIP — VGPR spill risk per V2.9.b**   |
| V2.31.d | 27B       | Q8_0 MMQ TILE_N=32                                | **SKIP — VGPR risk, same precedent**    |
| V2.31.e | Coder-30B | `mmvq_q4_k_r4` (decode, 4 rows/block)             | **PARKED — same kernel bug class as b** |
| V2.31.f | 9B        | Q4_1 MMQ wave64 / tile16 re-A/B at 100 W          | **NULL (confirmed, `4warp_lds` wins)**  |

## V2.31.a — shipped

Separate cert: `certs/perf/v2_31_a_q5_k_moe_tile8.md`. Coder-30B
Mesh<4> async ub=128: peak prefill 706 tok/s @ L=2048 (up from 523
pre-V2.31.a).

## V2.31.b — kernel correctness bug

`mmvq_q8_0_r4_dp4a.cu`: 64 threads, 4 rows per block, 16 lanes/row,
VDR=2. Static analysis says math matches `mmvq_q8_0_dp4a_vdr2` —
same `xi0·yi0 + xi1·yi1` DP4A pair per lane per block, same per-lane
scale-and-add, same quarter-warp reduce. But 27B Mesh<4> decode gives
different last_id (1074 → 71093) AND slight perf regression (18.75 →
17.58 tok/s) when `FLAMBEAU_VARIANT=q8_r4` is set.

Kept opt-in via env var (`FLAMBEAU_VARIANT=q8_r4`) — mainline safe.

Hypotheses for the bug (unresolved):
1. Subtle lane-to-block-to-int32-pair mapping error I can't see by
   inspection — needs a unit test that compares r4 output to vdr2 for
   a single row.
2. Quarter-warp DPP reduce interaction with 16-lane data that isn't
   identity-distributed.
3. Interaction with the `rows_per_block=4` grid dim vs the kernel's
   write guard.

## V2.31.c / V2.31.d — skipped

V2.9.b found that doubling per-thread accumulators on the MoE tile8
gate_up kernel (via `__launch_bounds__(X, 2)`) pushed scratch
156→684 B and regressed perf +101 %. Doubling the tile width (tile16)
would double the sum-accumulator count to 32 FP32 per thread, likely
spilling on gfx906 (260 VGPR/wave).

Same argument applies to Q8_0 MMQ TILE_N=32: V2.7 measured VGPR=112 at
TILE_N=16 (2 waves/SIMD); TILE_N=32 almost certainly pushes past the
wave-threshold.

Both are worth revisiting only with explicit PMC headroom verification
via `rocprofv3 --kernel-trace` showing VGPR count before/after.

## V2.31.e — kernel correctness bug (same class as b)

`mmvq_q4_k_r4.cu`: 64 threads, 4 rows/block, 16 lanes/row, 2 bytes/lane
per sub-block. Coder-30B Mesh<4> decode gave last_id 330 → 50467 when
`FLAMBEAU_VARIANT=q4_k_r4` is set. Same debug posture as V2.31.b —
opt-in, mainline safe.

The pattern class (wave64 multi-row r4 with quarter-warp reduction) is
the same one that works in `indexed_moe_mmvq_q4_k_r4_dp4a` (MoE
variant in production). The non-MoE ports for Q8_0 and Q4_K BOTH fail
correctness; parks both until the pattern is debugged.

## V2.31.f — null result confirmed at 100 W

V2.29.e's null verdict at 200 W stands at 100 W. 9B Mesh<1> sync
pp=512:

| variant                     | tok/s | Δ vs baseline |
|-----------------------------|------:|--------------:|
| `mmq_q4_1_4warp_lds` (base) |   631 | —             |
| `mmq_q4_1_wave64`           |   252 | −60 %         |
| `mmq_q4_1_wave64_tile16`    |   377 | −40 %         |

The 4-warp LDS-tiled kernel's 128×64 tile is structurally correct for
Q4_1's 32-element block. Wave64 single-warp emits 16× more
tile-blocks for the same output area and loses on launch overhead.
Same finding as V2.13 / V2.29.e.

Opt-in knobs kept for future testing:
- `FLAMBEAU_VARIANT=q4_1_wave64`
- `FLAMBEAU_VARIANT=q4_1_tile16`

## Net V2.31 delta

- **Coder-30B prefill L=1024 (async ub=128)**: 532 → 682 tok/s
  (**+28 %**, shipped via V2.31.a).
- **Coder-30B prefill L=512**: 426 → 641 tok/s (**+50 %**).
- All four other models unchanged.
- Three opt-in debug variants added (`q8_r4`, `q4_k_r4`,
  `q4_1_{wave64,tile16}`) — zero default-path risk.
