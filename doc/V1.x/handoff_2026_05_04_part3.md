# Handoff — 2026-05-04 (part 3): batched-MMVQ Q4_1 v1

Continuation of `handoff_2026_05_04_part2.md`. The previous session
shipped `forward_decode_pipelined_hybrid` (correct but perf-negative at
PP=2/N=2; gated to N≥4 where validation is blocked by 27B Q4_1 KV
memory). The 3× cert gate stays open with #288 (batched-MMVQ) as the
remaining highest-leverage lever.

## What this session shipped

| commit  | task | result |
|---------|------|--------|
| TODO    | **#288 v1** | Q4_1 batched-MMVQ kernel + parity (11/11 cases under 1e-5) + microbench. Partial win at small shapes (1.29× at 3584×4096 / N=8); regression at large shapes (0.84× at 14336×4096 / N=8). Gated opt-in via `FLAMBEAU_BATCHED_MMVQ=1`. |

The kernel `mmvq_q4_1_q8_1_batched` (256 threads/block, gridDim=(n_rows,))
loops N slots inside the per-K-block math, reusing the loaded weight bytes
across slots. Correctness validated (max abs-err 1.9e-6, f32 LSB scale —
the surrounding slot-loop changes hipcc's FMA-contraction choices vs the
single-row kernel; same scale of drift batched-GDN already accepts).

## Why it doesn't deliver the projected 1.5–2× win

**Root cause**: kernel is activation-HBM-bound, not weight-HBM-bound.

Per-thread per-K-block at N=8: 4 bytes weight + 64 bytes activation. At
n_rows=14336 / k=4096 / N=8: ~528 MB activation reads vs 35 MB weight
reads. Each output-row block re-fetches all N slots' activation rows
from HBM (L2=4MB too small to hold across 14336 rows). The amortization
design assumed weight HBM dominated.

**The projected 1.5–2× would require row tiling**: each block handles
R output rows × N slots, weight tile read once per (R, N), activation
read once per N×R outputs. That's the MMQ tile pattern adapted to
decode-friendly shapes — separate kernel-design effort.

See `certs/perf/mmvq_q4_1_batched_v1_2026_05_04.md` for the full
diagnosis + microbench table.

## What this means for the 3× cert gate

Path to ≥3.0× still hangs together but needs a **different mix**:

| lever                     | win   | status           |
|---------------------------|-------|------------------|
| #266c batched-attn        | 1.05× | landed           |
| #287 batched-GDN wired    | 1.03× | landed           |
| #290 PP-pipelining (PP=2) | ≤1.6× | code shipped, blocked at N≥4 by 27B Q4_1 KV-budget OOM |
| #288 batched-MMVQ v1      | 0.84× to 1.29× shape-dependent | partial; gated opt-in |
| **#288-v2 row-tiled MMVQ**| ~1.5× projected | **next session**, see below |

Combined ceiling with #288-v2: 1.05 × 1.6 × 1.5 ≈ 2.5× — still under 3×
gate. To clear the gate cleanly, also need either:
- Pure-PP=4 (instead of PP=2/TP=2) which raises pipelining ceiling to
  2.3× at N=4 / 2.9× at N=8. Combined: 1.05 × 2.3 × 1.5 = 3.6× ✓.
- A larger-N validation (INFLIGHT_SLOTS=8) which needs smaller ctx /
  smaller quant on this rig (27B Q4_1 OOMs at SLOTS=4 already).

## Open path forward (priority order)

### #288-v2: row-tiled batched-MMVQ Q4_1 (HIGHEST LEVERAGE, NEXT SESSION)

Rewrite the kernel to:
- gridDim = (n_rows / R, ), R ∈ {4, 8, 16}.
- Each block processes R output rows × N slots.
- LDS-tile the activation rows: load N × 36-byte Q8_1 blocks per K-iter
  into LDS, share across the R rows.
- Load weight tiles once per (R-row, K-iter) into registers; reuse
  across N slots.

Reference: `crates/kernels-hip/src/kernels/mmq_q4_1_4warp_lds.cu` for
the LDS-tile pattern. The MMQ pattern targets prefill (large m, R=8 or
16); we want the same shape with m=N (decode batch dim) instead.

Estimated kernel size: ~250-400 lines. Parity test reuses the existing
`mmvq_q4_1_batched_parity.rs` test plus shape sweep. Microbench reuses
the existing `mmvq_q4_1_batched_perf.rs` plus larger shapes (Q8_K-XL
weights at hidden=14336).

### #288-Q4_0: replicate the same approach for Q4_0

Qwen3.6-35B-A3B uses Q4_0 weights. Need the same kernel for that model.

### Pure-PP=4 pipelining

Switch the pipelined-decode design from PP=2/TP=2 to pure PP=4 (TP=1).
Higher pipelining ceiling (2.3× at N=4). Memory: 27B-Q4_1 over PP=4
TP=1 = each rank holds 1/4 of layers with full weights — 17.2/4 ≈ 4.3
GB per rank. KV cache ~3 GB for SLOTS=4. Should fit on 4×16GB MI50.

This is mostly a server config / scratch-allocation change, not new
kernel work.

### Smaller validation rig

For closing the gate without all of the above, run on a smaller model
that fits more inflight slots:
- Qwen3.6-27B-Q4_0 (15.7 GB instead of 17.2 GB) — might fit 4 slots
  with reduced ctx.
- Qwen3.5-9B-Q4_1 (no GDN, no MoE) — wouldn't exercise the GDN path
  but validates infrastructure.

## Diagnostic toggles added

- `FLAMBEAU_BATCHED_MMVQ=1` — engages the v1 batched-MMVQ short-circuit
  at Q4_1 / m ∈ [2, 8]. Default OFF; only safe to enable on shapes
  where the microbench shows a win (small n_rows). Production decode
  on Qwen3.6-27B should KEEP this off until v2 lands.

(Plus all toggles inherited from prior handoffs:
`FLAMBEAU_BATCHED_DECODE`, `FLAMBEAU_INFLIGHT_SLOTS`,
`FLAMBEAU_DECODE_PIPELINE`, `FLAMBEAU_PIPELINE_BLOCKING_COPY`,
`FLAMBEAU_GDN_NO_BATCHED`, `FLAMBEAU_TRACE_BATCH`, etc.)

## Memory note saved

A new memory entry should be added by the next session:
`feedback_mmvq_batched_activation_hbm.md` — at decode shapes (small N,
large n_rows), batched-MMVQ kernels are activation-HBM-bound, not
weight-HBM-bound. The textbook "weight-amortization" win pattern
DOESN'T apply at the qkv/ssm_out widths used by GDN. Row tiling is
required.
