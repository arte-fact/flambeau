# V2.28.d — Q8 KV cache: kernel exists, perf argument evaporates

Investigated the Q8 KV roadmap item. The kernel `attention_decode_q8_kv`
already ships with cert (certs/hip/gfx906/attention_decode_q8_kv_gfx906.json)
and an A/B test (crates/bench/tests/attention_decode_ab.rs).
Re-ran the A/B on Qwen3.6-shape (head_dim=256, n_heads_q=16,
n_heads_kv=2) to validate before committing to the end-to-end wiring.

## A/B result — Q8 KV is SLOWER than F16, and split-K F16 beats both

| n_tokens | F16 decode | Q8 decode | F16 split-K | Q8/F16 | F16/split-K |
|---:|---:|---:|---:|---:|---:|
| 128  |  194 µs |  182 µs |  177 µs | 1.07× | 1.10× |
| 512  |  677 µs |  705 µs | **179 µs** | 0.96× | **3.79×** |
| 1024 | 1416 µs | 1476 µs | **181 µs** | 0.96× | **7.83×** |
| 2048 | 2685 µs | 2873 µs | **344 µs** | 0.93× | **7.81×** |
| 4096 | 5292 µs | 5661 µs | **667 µs** | 0.93× | **7.93×** |

## Why Q8 loses despite "2× bandwidth saving"

CLAUDE.md rationale was: "decode is HBM-bound on MoE and Q8 is ~2×
bandwidth saving". Two invalidating findings:

1. **Q8 decode kernel is occupancy-starved, not HBM-bound.**
   At 16 heads × 1 block = 27 % of 60 CUs (same starvation as F16
   non-split-K). Both kernels sit compute-stalled waiting for the
   CU scheduler, not HBM. Halving HBM bandwidth consumption doesn't
   speed up a kernel that's not reading HBM at peak.

2. **Split-K already solved the occupancy problem for F16 at
   n_tokens ≥ 512.** It partitions context across grid.y, lifting
   block count 16 → 64-256 and saturating all 60 CUs. Result: 7-8×
   speedup over single-pass F16. **We have no Q8 split-K kernel**.
   If we shipped one, it'd match F16 split-K at best (no bandwidth
   gap to exploit since we're already compute-bound post-split-K).

3. **Memory is not the gate.** 4× MI50 × 16 GiB = 64 GiB VRAM.
   9B Q4_1 Mesh<4> F16 KV at context=32K uses ~2.4 GiB/rank
   (9 full-attn layers × 32768 × 4·128·2·2 = 2.4 GiB). Q8 halves
   it to ~1.2 GiB/rank — still sub-1/10 of per-rank budget.
   Even 35B MoE at 32K context has headroom on these cards.

## Recommendation — skip the end-to-end Q8 KV wire

The V1 roadmap (CLAUDE.md) gates Q8 on a quality cert (delta-ppl
≤ 0.5 %). **Not worth running the quality cert** since the perf
hypothesis is dead:

- Perf loses 3-7 % vs F16 for the same context.
- vs split-K F16 (current decode default at n_tokens_kv > 256),
  Q8 loses 7-8×.
- Memory savings (2× context capacity) don't unlock a concrete
  workload we can't already serve.

Keeping the Q8 kernel (`attention_decode_q8_kv`) in-tree as-is for
future use if:
- A Q8 split-K variant is written and benchmarks ≥ F16 split-K.
- We get memory-constrained on a different rig (e.g. 8 GiB cards).

Both conditions require new work + new evidence. Parking this
roadmap item.

## Running state table — what the V1 roadmap items look like

| item | status |
|---|---|
| F16 KV (baseline) | **shipped, default** |
| Q8 KV | **kernel exists, A/B-null, end-to-end wiring skipped** |
| Turbo-quant Q4/Q5 KV | V2+, not started |

## Gate

- No code change from this iteration other than this cert.
- Existing A/B test (`attention_decode_ab.rs`) re-runs clean:
  test passes, split-K parity within tolerance.
- `attention_decode_q8_kv` kernel + cert stay in-tree; no deletion.

## Regeneration

```
cargo test --release -p flambeau-bench --features hip \
  --test attention_decode_ab -- --nocapture
```
