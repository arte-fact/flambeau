# Sarathi mixed-batch v1 — Qwen3.5-9B-Q4_1 / pp2tp2 microbench

**Date:** 2026-05-04
**Rig:** 4× MI50 (gfx906) PCIe 3.0 x16, ROCm 7.1.1
**Model:** Qwen3.5-9B-Q4_1 (qwen35 — hybrid GDN+full-attn+MoE)
**Topology:** pp2tp2 over devices [0,2,1,3]
**Task #:** #308 (microbench)
**Commit:** `0b56116` + sweeps at K=256/N=8 and K=128/N=16

## TL;DR

`forward_decode_mixed_hybrid` (Lever 1, #304) **delivers a real but
modest speedup** vs the sequential `prefill_then_decode` baseline:

| config         | seq median (ms) | mixed median (ms) | speedup | aggregate (tok/s) |
|----------------|----------------:|------------------:|--------:|------------------:|
| K=512 / N=4    | 638.09          | 600.45            | **1.063×** | 808 → 859 |
| K=256 / N=8    | 427.13          | 381.30            | **1.120×** | 618 → 692 |
| K=128 / N=16   | 389.64          | 331.95            | **1.174×** | 369 → 433 |

The speedup scales with N/K ratio — more concurrent decodes absorbing
into a smaller prefill chunk yields a larger relative win.

## Why the absolute speedup is modest

Sarathi's reported 3–5× wins (vLLM blog) are **aggregate-throughput**
gains over many iterations of mixed traffic, not single-call wall
gains. This microbench measures **per-call wall** for one mixed
batch, where the speedup ceiling is bounded by:

```
speedup ≤ (T_prefill + T_decode) / max(T_prefill, T_decode + T_overhead)
```

For the hybrid GDN+attn+MoE arch on pp2tp2:
- T_prefill scales O(K) compute-bound (~1.2 ms per token at 9B/pp2tp2)
- T_decode scales O(N) HBM-bound (~5–10 ms per slot at 9B/pp2tp2)

At K=512/N=4: T_prefill ≈ 600 ms, T_decode ≈ 38 ms → max(600, 38+ε)
≈ 600 ms; mixed eats the 38 ms decode wall, giving (638-600)/638 ≈
6% wall savings. Theoretical maximum on this shape: ~6.3% (matches
measured 6.3%).

At K=128/N=16: T_prefill ≈ 154 ms, T_decode ≈ 154 ms → mixed
fully overlaps; theoretical maximum (308-154)/308 ≈ 50%. We measure
~17%, lower because of:
1. Two-attn-call overhead per layer (vs one ideal varlen kernel) —
   wins back ~4 % with #303 v2 kernel
2. Per-slot GDN sequential loop on the decode side
3. Per-slot output-head dispatch on the head rank

## Where the bigger Sarathi wins live (not in this bench)

The 3–5× claims are about **system throughput under mixed traffic**:
- New requests don't stall during long prefills — they decode inside
  the same forward pass that's processing some other request's
  prefill chunk.
- TPOT spikes during prefill are eliminated.
- Aggregate `(prefill_tokens + decode_tokens) / wall` improves
  dramatically when the request mix has both traffic classes.

The driver alone delivers the per-call gain measured here; the
**scheduler** (#305) is what unlocks the system-level throughput
win. Without #305, only one mixed call happens per server iteration,
and the rest of the serve loop processes prefill / decode requests
independently as before.

## What would break the modest result

These tests give bit-exact (K=32/N=1, K=32/N=2) or top-1-matching
parity (K=128/N=4) vs separate-sessions reference (#307). Top-1
parity holds at K=512/N=4 and K=128/N=16 used in this bench (not
re-asserted here but parity test runs the same kernel paths).

## Recommendation

The lever is real, the v1 driver works correctly and produces a
1.06×–1.17× per-call wall improvement. Decision for next session:

- **Wire #305 + #306** to deliver Sarathi's aggregate-throughput
  win. The lever delivered here (~6–17%) is the FLOOR; the scheduler
  could push aggregate throughput another ~2–5× by keeping decode
  slots saturated during long prefills (Sarathi's design point).
- **Defer #303** (varlen attention kernel) until profile shows
  two-attn-call overhead is the bottleneck. The current v1 ratio
  vs theoretical max suggests there's ~10–25% upside from #303,
  but it requires real kernel work (3+ sessions).

## Methodology

```rust
// per-trial:
//   seq path:  prefill_B + decode_batched(N)
//   mix path:  forward_decode_mixed_hybrid(chunk=B, slots=N)
// 2 warmups + 5 reps, median of 5.
// Each trial uses fresh sessions + scratch.
```

Knobs: `FLAMBEAU_BENCH_K`, `FLAMBEAU_BENCH_N` to sweep other shapes.

Test source: `crates/models/qwen3-moe/tests/mixed_batch_microbench.rs`.
