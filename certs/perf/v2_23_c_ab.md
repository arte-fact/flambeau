# V2.23.c — Lever-3: per-rank work balance AB report

## Diagnosis

Per-rank kernel-time breakdown from the V2.23.b post-commit profile
(35B-A3B-Q4_0 Mesh<4> decode tg=64):

| rank | layers | kernel_ms | share | ms delta vs avg |
|---|---:|---:|---:|---:|
| 0 | 10 | 361.4 | 23.0 % | −32 |
| 1 | 10 | 376.1 | 23.9 % | −17 |
| 2 | 10 | 379.7 | 24.1 % | −13 |
| 3 | 10 + LM head | 456.2 | 29.0 % | **+63** |
| Σ | 40 | 1573.4 | 100 % | — |

**max/min = 1.26 → +26 % imbalance.** Rank 3 tail-bottlenecked by
`output_norm + dense_gemv_f32_f16(vocab × hidden) + topk` on top of its
10 transformer layers.

Since decode-PP is serial across ranks (token N's layers are fully
consumed before token N+1 starts that rank), wall-time is
`N × sum_i(rank_i_time) / N_ranks` in the ideal case but is actually
bounded by `N × max_i(rank_i_time)` when imbalance exists. Cutting the
tail rank by one layer's worth of work is the lever.

## Attempts

### C.1 — 11/10/10/9 layer split (shift 1 layer rank 3 → rank 0)

New `LayerAssignment::from_counts(&[u32])` constructor + bench env
`FLAMBEAU_PP_LAYERS="11,10,10,9"`.

### C.2 — 12/10/10/8 (more aggressive shift)

Same mechanism, heavier rebalance.

## Results (decode tok/s, median of 3 unprofiled runs)

| | layers | tok/s | max/min | per-rank ms (agent_1..4) |
|---|---|---:|---:|---|
| B.2 baseline | 10/10/10/10 | 49.30 | 1.26 | 361 / 376 / 380 / **456** |
| **C.1** | **11/10/10/9** | **49.48** | **1.13** | 406 / 360 / 365 / 408 |
| C.2 | 12/10/10/8 | 49.76 | 1.24 | **452** / 366 / 380 / 376 |

C.1 is the clear winner by balance metric (max/min 1.13 — close to noise
floor). C.2's tok/s is a touch higher but noisy (49.04-49.79 range vs
C.1's 48.97-50.19), and structurally it's a worse balance (rank 0 now
dominates). C.2 filed as diagnostic — keep C.1 as the ship config for
Qwen3.6-35B-A3B on 4×MI50.

## Gate

- UD-Q4_K_S 8-token parity vs llama.cpp bit-exact (seed 9419) — unaffected
  by layer-assignment change
- cert-check hip/gfx906: 48 rows, 0 failures

## Takeaway

PP tail-bottleneck is small but real. Per-rank max/min > 1.15 should
trigger rebalance. For Qwen3.6-35B-A3B (40 layers, hidden=2048,
vocab~152k, one LM head pass per decode token) the LM head costs
roughly one transformer layer → shift 1 layer off the last rank.
Generalise: if the model has tied `output` + `token_embd` on the last
rank, reserve ~1 layer's worth of work.

## Next (future V2.23.c.x)

Auto-detect in `LayerAssignment`: when `num_ranks > 1`, default to
`{num_layers - num_ranks + 1, ..., num_layers - num_ranks + 1, num_layers - num_ranks}`
or a measured-cost allocation from a warmup pass. Out of scope for this
AB session — env override is sufficient for bench reproducibility and
ships as-is.
