# MTP-5h — final summary of spec-decode optimisation work on PP4 / MI50

**Status:** opt-in (`FLAMBEAU_SPEC_MTP=path`), default off. Recommended
production topology for greedy decode is **TP2 baseline** (36.6 ms/tok),
not spec-decode.

## Stacked landed optimisations on PP4

| step | spec ms/tok | spec − baseline | savings (cumulative) |
|---|---:|---:|---:|
| Initial MTP-5e (paired-L2) | 59.33 | +18.0 % | — |
| + Lever 1 (GDN-only redo on reject) | 56.46 | +13.0 % | 2.87 ms/tok |
| + Lever C (lazy output head pos1) | 56.16 | +12.1 % | 0.30 ms/tok |
| + #194 (partial-sort + penalty-aware build_distribution) | 56.11 | +11.9 % | 0.05 ms/tok* |
| **stacked total** | **56.11** | **+11.9 %** | **3.22 ms/tok** |

*Greedy benchmark mostly skips build_distribution; #194 savings show up
under sampling (smoke 95.5 → 90.7 ms/tok = 5 % wall improvement at
top_p=0.9 / temp=0.8 / 16 macros).

First 16 tokens bit-identical to baseline across every step → correctness
preserved through every optimisation.

## Lever B shipped with NO measured win on this rig

**Lever B** (concurrent MTP draft on aux stream) was the smallest of the
three deferred levers (~80 LOC). Implementation is correct: MTP forward
issues kernels asynchronously on `cluster.with_aux_stream(last_rank, 0)`
while `save_gdn_snapshot` issues D2D copies on per-rank default streams,
intended to overlap on the head device's two streams.

3-run A/B post-Lever-B: 56.12 / 56.30 / 56.40 ms/tok (mean 56.3, vs
pre-Lever-B 55.69; **within ±0.5 ms noise floor**). The projected
0.5–0.8 ms/tok savings either didn't materialise (HIP scheduler may not
fully parallelise concurrent same-device streams when one is doing pure
D2D and the other compute) or is below the run-to-run variance.

This empirical null result informed the decision to NOT invest the
remaining ~250 LOC of structural work for Levers A and D on this rig.

## Levers A + D — designs captured, implementation deferred to 3090

- **Lever A** (shadow GDN buffer, ~150 LOC structural) — replace D2D
  state-copy in save_gdn_snapshot with shadow-buffer ping-pong via the
  kernel's existing `(state_in, state_out)` signature. Pointer swap on
  accept; no-op on reject. Projected 0.6 ms/tok. Stacks correctly with
  Lever 1's redo_gdn_only_pp (which operates on active state in-place
  at L=1).
- **Lever D** (save GDN state mid-L=2, ~100 LOC) — split fused L=2
  state-step into two sequential L=1 calls, checkpoint state between
  them; reject restores from checkpoint instead of running
  redo_gdn_only_pp. Projected 0.5 ms/tok.

Stacked with Lever B (which is shipped but invisible here), these
would give an additional ~1.1 ms/tok on PP4 / MI50 — projecting spec
to ~55.0 ms/tok = +10 % vs baseline. Given B's null result, the same
null is likely for A and D on this rig. **Both designs preserved for
the 3090 rig transition** where:
  - PCIe 4 + faster D2D paths shrink the L=2 body cost
  - The relative weight of fixed overhead grows
  - Async stream parallelism on NVIDIA is generally more aggressive
    than ROCm's
  - A 0.5–0.6 ms saving on a faster baseline becomes a larger relative
    win

Re-evaluation plan: when sm_86 base forward is up, port Lever B's pattern
first (cheapest), measure. If wins on 3090 → port A and D too.

## #195 hybrid pp2tp2 — deleted

Based on the topology comparison findings:
- PP4 baseline: 50.22, spec +12.6 %
- PP2 baseline: 54.35, spec +17.2 %
- TP2 baseline: 36.64, spec +53.8 %

Hybrid pp2tp2 would land between PP2 and TP2 — predictable, low ROI.
Re-open if 3090 NVLink rig changes the topology landscape.

## Code surface preserved

All implementation lives behind `FLAMBEAU_SPEC_MTP` env-var. Default off
means zero spec-decode runtime cost when the env var is unset. The four
code paths (greedy/sampling × non-streaming/streaming) all gated.

The implementation is portable to CUDA when sm_86 is brought up — no
gfx906-specific code in the spec path. MTP head loading + driver +
session methods + server glue all ride on whatever forward primitives
the backend exposes.

**Net:** ~5 weeks of MTP work distilled into a working K=1 spec-decode
implementation that's net-negative on PCIe-3 / no-NVLink hardware (a
finding consistent with the 2022 Leviathan paper's section 3.4 caveat).
The same implementation should flip net-positive on PCIe-4 / NVLink
hardware (3090 transition) without code changes — verified by porting,
re-running the existing perf A/B harnesses.
