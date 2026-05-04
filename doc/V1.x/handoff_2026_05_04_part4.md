# Handoff — 2026-05-04 (part 4): #288-v2 wave64 routing

Continuation of `handoff_2026_05_04_part3.md`. The previous session
shipped #288 v1 (per-output-row batched-MMVQ, partial win at small
shapes / regression at large shapes due to activation-HBM bound). The
3× cert gate stays open with #288-v2 (row-tiled batched-MMVQ) projected
as the highest-leverage lever.

## What this session shipped

| commit  | task | result |
|---------|------|--------|
| 264f438 | **#288-v2** | route Q4_1 batched-MMVQ via existing `mmq_q4_1_wave64` kernel; shape-aware (n_rows ≥ 8192 → wave64, else fall through). Wins **1.27× over per-row baseline at GDN ssm_out shape (14336×4096/N=8)**, 2.24× over qmatmul(m=N) baseline. |
| 7fbc9e4 | #288-v2 finding | wave64 breaks bit-identical-within-batch at LSB scale due to compiler FMA-contraction asymmetry between c=0 and c=1 in the per-col unrolled loop. New test documents it. |

## Key insight (saves next session a kernel rewrite)

The handoff_2026_05_04_part3 plan was to write a NEW row-tiled kernel
(MMQ-tile pattern adapted to decode). **It turned out the existing
`mmq_q4_1_wave64` PREFILL kernel already has that exact shape**
(gridDim=(n_rows/64, n_cols/8), MMQ_Y=64, TILE_N=8) and works correctly
when invoked at small N (decode batch dim).

Microbench at Qwen3.6-27B GDN matmul shapes:
- 3584 × 4096 (qkv): wave64 0.54× (regression — too few thread-blocks
  to fill GPU), v1 1.06×, per-row baseline best.
- 14336 × 4096 (ssm_out): **wave64 1.27× over per-row, 2.24× over
  baseline qmatmul(m=N) MMVQ-loop**.

So the dispatch is now **shape-aware**:
- `n_rows ≥ 8192` → wave64 (large GDN matmuls)
- `n_rows < 8192` → fall through to per-row (small projections)

Override knobs added: `FLAMBEAU_BATCHED_MMVQ=v1` / `=wave64` to force
either path for A/B.

## Live test result — wave64 changes output (LSB drift)

On Qwen3.6-27B / pp2tp2 / 2 concurrent chat at temp=0:
- Baseline: both slots md5 59f23a62 ("thunderous sound..." = N=1).
- Wave64 path: slot 0 md5 59f23a62 (= N=1), slot 1 md5 c2547432
  ("rocky shore..."). Both coherent poems.

Per-slot identical-activation test (new
`mmvq_q4_1_wave64_identical_slots.rs`): even with bit-identical
activation rows for slot 0 and slot 1, wave64 produces 78% of outputs
differing at f32 LSB scale (max 3.3e-6). Compiler FMA-contraction is
asymmetric across the unrolled `for c in 0..TILE_N` loop in
`mmq_q4_1_wave64.cu` — hipcc emits slightly different FP code per c.

The kernel is IEEE-correct (within tolerance vs single-row reference);
it just isn't bit-identical across slots. LSB drift compounds over 64
layers × multiple matmuls per layer and flips greedy argmax at some
point. Output stays coherent.

The shape-aware dispatch stays **gated opt-in**
(`FLAMBEAU_BATCHED_MMVQ=1`). Users who depend on
bit-identical-within-batch output (existing batched-decode invariant)
should keep it OFF. Production default is unchanged.

## Where the 3× cert gate stands

| lever                               | win                  | status                                |
|-------------------------------------|---------------------:|---------------------------------------|
| #266c batched-attn                  | 1.05×                | landed                                |
| #287 batched-GDN wired              | 1.03×                | landed                                |
| #290 PP=2 pipelining                | ≤1.6× ceiling at N=4 | shipped, blocked at N≥4 by 27B Q4_1 KV OOM |
| **#288-v2 wave64 routing**          | ~1.15× combined      | **opt-in, breaks bit-id within batch** |

Combined ceiling: 1.05 × 1.6 × 1.15 ≈ 1.93× — improves the post-#287
1.03× cert toward 2× but still under the 3× gate.

To clear 3× cleanly, the remaining levers are:

### Lever A: pure-PP=4 pipelining (TP=1) — most promising

Pipelining ceiling at PP=4/N=4 is 2.3×, at PP=4/N=8 is 2.9×. Combined
with #266c (1.05×) + #288-v2 (1.15×): **~3.5×** — clears 3× cleanly.

Memory check: 27B Q4_1 (17.2 GB) over PP=4 / TP=1 = 17.2/4 ≈ 4.3 GB
weights per rank. KV cache scales with INFLIGHT_SLOTS × layers/4.
Should fit on 4×16GB MI50 with reasonable context.

This is mostly a server config / scratch-allocation change, not new
kernel work. The pipelined function `forward_decode_pipelined_hybrid`
already supports any PP value — but the implementation was specialised
to PP=2 (`if n_stages != 2 { bail!(...) }`). Need to remove that
restriction and add proper PP>2 chained-bridge logic.

### Lever B: v3 wave64 with explicit per-slot symmetric processing

Write `mmq_q4_1_wave64_decode.cu` that:
- Forces FMA contraction off (`#pragma clang fp contract(off)`) OR
- Manually unrolls the per-col loop with explicit identical FMA
  sequences for c=0 and c=1 (no compiler asymmetry).

Goal: preserve the 1.27× win AND maintain bit-identical-within-batch.

### Lever C: validate at INFLIGHT_SLOTS=8 with smaller-quant model

27B Q4_1 OOMs at SLOTS=4. Try Qwen3.6-27B-Q4_0 (smaller weights, more
KV headroom) or 27B Q4_1 with reduced ctx. Validates the larger-N
pipelining ceilings (1.78× at PP=2/N=8, 2.91× at PP=4/N=8).

## Open tasks queue (refreshed)

- **#293** [completed]: #288-v2 wave64 routing — shipped opt-in.
- **NEW pending** (not yet a numbered task): pure-PP=4 pipelining.
- **NEW pending**: v3 wave64-decode kernel (FMA-contraction-symmetric).
- **#288** [completed]: v1 + v2 both shipped; further iteration is v3.

## Diagnostic toggles updated

- `FLAMBEAU_BATCHED_MMVQ=1` — engages shape-aware Q4_1 batched-MMVQ
  (wave64 at n_rows ≥ 8192). Default OFF; production decode keeps
  bit-identical-within-batch.
- `FLAMBEAU_BATCHED_MMVQ=v1` — force v1 batched-MMVQ kernel (per-row
  with slot-loop inside K-iter). For A/B comparison.
- `FLAMBEAU_BATCHED_MMVQ=wave64` — force wave64 unconditionally.

(Plus all toggles inherited from prior handoffs.)

## Memory notes status

The session 3 memory note `feedback_mmvq_batched_activation_hbm.md` is
still accurate; v2 doesn't invalidate it (v1's per-row design is what
that note describes). A new memory note for the wave64 FMA-contraction
asymmetry would be useful next session — `feedback_wave64_decode_n_lsb_asymmetry.md`.
