# #242 — v2 moe_ffn prefill via forward_prefill_tp_f32 (2026-05-18)

## Result

Qwen3.6-35B-A3B-Q4_0 / PP2 / hip:0,2 / K=24:

|              | legacy | post-#241 | **post-#242** | v2 ÷ legacy |
|--------------|-------:|----------:|--------------:|------------:|
| N=1 t/s      | 42.95  | 22.52     | **30.83**     | **0.72×**   |
| N=2 agg t/s  | 42.83  | 22.22     | 30.71         | 0.72×       |
| N=4 agg t/s  | 43.84  | 23.19     | **35.31**     | **0.81×**   |

**+37% at N=1, +52% at N=4 aggregate.** Cumulative v2 improvement
since the start of this session: `6.48 → 30.83 t/s = 4.76×`. The
v2-vs-legacy gap is now `1.39×` (down from `6.6×`).

Dense 9B-Q4_1: unchanged within noise (no MoE path).

## What changed

When `moe_ffn` is called with `n_tokens > 1` (prefill chunk OR
multi-slot batched-decode), the composite now:

1. rmsnorm_f16 at L tokens → norm scratch.
2. quantise [L, hidden] → block prefill `x_q8_1`.
3. router qmatmul at L tokens → router_logits `[L, n_experts]`.
4. One DtoH of all router logits + per-row host topk + one HtoD of
   `expert_ids` `[L, top_k]` + `expert_weights` `[L, top_k]`.
5. `flambeau_blocks::MoeExperts::forward_prefill_tp_f32` (new) →
   writes F32 partial `[L, hidden]` to `pool.down_f32`. Internally
   dispatches the tile8 MMQ kernels when `n_pairs ≥ 8`.
6. `hooks.ar_sum_f32` on the F32 partial (no-op SD/PP, real on TP).
7. cast F32 → F16 → delta.
8. Shared expert (when present): looped per token after the routed
   path. SharedExpert block has no prefill variant; small per-layer
   add but the routed path is now fully batched.

`forward_prefill_tp_f32` is a thin sibling of `forward_prefill_tp`
that swaps `moe_combine_no_residual_f16` for
`moe_combine_no_residual_f32`, mirroring the existing
`forward_decode_tp_f32` shape. Lets the v2 composite stay on the
F32 ar-sum path uniformly.

## Profile delta

Qwen3.6-35B-A3B-Q4_0 / K=4 + ~24-token prefill:

|                                                | post-#241 | post-#242 |
|------------------------------------------------|----------:|----------:|
| Total GPU kernel time                          |  568 ms   |  457 ms   |
| Total launches                                 | 57 411    | 34 065    |
| `indexed_moe_mmq_q4_0_gate_up_tile8_dp4a_q8_1` |     0     | 38 ms / 80 |
| `indexed_moe_mmq_q4_0_down_tile8_dp4a_q8_1`    |     0     | 14 ms / 70 |
| `indexed_moe_mmvq_q4_0_q8_1` (decode path)     | 105 / 1890| 12 / 210  |

Comparison to legacy trace (same K=4 capture):

| kernel                                         | legacy    | v2 post-#242 |
|------------------------------------------------|----------:|-------------:|
| `indexed_moe_mmq_q4_0_gate_up_tile8_dp4a_q8_1` | 40 ms / 80| 38 ms / 80   |
| `indexed_moe_mmq_q4_0_down_tile8_dp4a_q8_1`    | 14 ms / 70| 14 ms / 70   |
| `indexed_moe_mmvq_q4_0_q8_1`                   | 12 ms / 210| 12 ms / 210 |

v2 matches the legacy MoE kernel pattern **call-for-call** on
qwen3.6-35B-A3B-Q4_0 now.

## Cumulative v2 evolution (Qwen3.6-35B-A3B-Q4_0 / PP2 / N=1)

```
                  N=1 t/s   v2 ÷ legacy
pre-#239           6.48     0.151×
post-#239 (MoE indexed decode)      19.80     0.461×    [+205%]
post-#240 (batched-attn N>1)        19.83     0.462×    [ +0% ]
post-#241 (GDN forward_prefill)     22.52     0.524×    [+13.6%]
post-#242 (MoE forward_prefill)     30.83     0.718×    [+37%]
legacy                              42.95     1.000×
```

72% of the original 85% gap closed. v2 internal cumulative: **4.76×**.

## What's left

Remaining ~28% to legacy parity, ranked by trace contribution
(post-#242 vs legacy):

| candidate | v2 ms (×calls) | legacy ms (×calls) | observation |
|-----------|---------------:|-------------------:|-------------|
| `mmvq_q6_k_dp4a_q8_1` | 55 (×278) | 14 (×248) | v2 per-call **3.6× slower** — possibly a row-tile / waves_per_eu dispatch difference at Q6_K; embed/lm_head route. |
| `mmvq_q5_0_q8_1`      | 50 (×3240)| 27 (×3240)| v2 per-call 1.8× slower at same call count — same Q5_0 dispatch row mismatch. |
| `mmvq_q4_0_q8_1` (decode) | 91 (×4554) | 109 (×4524) | v2 is actually FASTER per call; nothing to gain here. |
| Shared expert per-token loop | ~25 / 32 layers × N | — | A `SharedExpert::forward_prefill` could batch this. |

The remaining gap looks like **dispatch-table mismatches at Q6_K /
Q5_0**, not architectural. A focused review of which row v2's
qmatmul dispatch picks for Q6_K / Q5_0 vs legacy could close most of
this.

## Files

- `crates/blocks/src/moe_experts.rs`: + `forward_prefill_tp_f32`.
- `crates/forward/src/core/scratch.rs`: + `moe_prefill_scratch`
  (`OwnedMoeExpertsPrefillScratch`) allocated when
  `max_experts > 0 && max_prefill_tokens > 1`.
- `crates/forward/src/core/composites/moe_ffn.rs`: full rewrite of
  `moe_ffn_loop` (now the batched prefill path, not a per-token loop).
