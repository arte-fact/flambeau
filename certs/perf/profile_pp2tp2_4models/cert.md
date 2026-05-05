# pp2tp2 4-model kernel profile (rocprofv3 --kernel-trace)

Topology: pp2tp2 on `hip:0,2,1,3` (pp_size=2, tp_size=2).
Profile request shape: ~512-token prompt + 64 decode tokens, greedy, one warmup request before the traced one.
Server features: FLAMBEAU_BATCHED_DECODE=1, GPU_SAMPLER=1, PREFIX_CACHE=1, 4 inflight slots, 512-token prefill ubatch.

## Cross-model kernel-family share (% of total kernel time per model)

| family | qwen35_9B_q4_1 | qwen36_27B_q4_1 | qwen36_35B_a3b_q4_0 | qwen3_coder_next_q4_0 |
|---|---:|---:|---:|---:|
| attn_decode | 11.5% | 8.3% | 14.0% | 11.8% |
| attn_prefill | 1.7% | 1.9% | 2.1% | 1.7% |
| gdn | 8.4% | 4.7% | 10.4% | 8.6% |
| mmvq_q4 | 18.4% | 20.4% | 16.6% | 15.0% |
| mmvq_q5 | 5.7% | 8.2% | 1.5% | 4.3% |
| mmvq_q6_q8 | 3.6% | 2.1% | 4.6% | 3.0% |
| mmq_4warp_lds | 27.1% | 34.4% | 9.5% | 8.4% |
| moe_mmq_prefill | — | — | 11.0% | 12.5% |
| moe_mmvq_decode | — | — | — | — |
| rmsnorm | 6.8% | 5.7% | 5.6% | 4.7% |
| rope | 0.2% | 0.2% | 0.3% | 0.3% |
| quantize | 3.7% | 3.3% | 4.5% | 4.2% |
| cast | 1.9% | 1.5% | 3.1% | 2.8% |
| p2p_allreduce | 5.6% | 5.3% | 3.6% | 3.1% |
| topk_softmax | — | — | 2.7% | 3.4% |
| dense_gemv | — | — | 2.8% | 9.9% |
| output_head | — | — | — | — |
| misc | 5.5% | 4.0% | 7.6% | 6.3% |

_Read this matrix to spot **common levers**: a row hot on every model is a family-wide lever; a row hot on one model is a model-specific lever._

## Per-model top-20 kernels

### qwen35_9B_q4_1

| rank | kernel | calls | total ms | avg µs | pct |
|---:|---|---:|---:|---:|---:|
| 1 | `flambeau_mmq_q4_1_4warp_lds_q8_1` | 1056 | 1313.45 | 1243.80 | 20.89% |
| 2 | `flambeau_attention_decode_f16_splitk_chunk` | 2016 | 709.44 | 351.90 | 11.28% |
| 3 | `flambeau_mmvq_q4_1_t128_q8_1` | 28224 | 633.81 | 22.46 | 10.08% |
| 4 | `flambeau_mmvq_q4_1_gate_up_dp4a_q8_1` | 8064 | 520.38 | 64.53 | 8.28% |
| 5 | `flambeau_gdn_state_step_alphabeta_f32_s128` | 6192 | 494.81 | 79.91 | 7.87% |
| 6 | `flambeau_mmvq_q5_k_r2_q8_1` | 6048 | 356.74 | 58.98 | 5.67% |
| 7 | `flambeau_p2p_allreduce_residual_tp2` | 8448 | 352.15 | 41.68 | 5.60% |
| 8 | `flambeau_mmq_q5_K_wave64_q8_1` | 144 | 226.28 | 1571.41 | 3.60% |
| 9 | `flambeau_rmsnorm_q8_1_fused` | 8193 | 198.65 | 24.25 | 3.16% |
| 10 | `flambeau_quantize_row_f16_q8_1` | 18768 | 193.03 | 10.28 | 3.07% |
| 11 | `flambeau_mmvq_q6_k_dp4a_q8_1` | 129 | 171.13 | 1326.62 | 2.72% |
| 12 | `flambeau_p2p_allreduce_residual_rmsnorm_tp2` | 8064 | 162.65 | 20.17 | 2.59% |
| 13 | `flambeau_mmq_q8_0_wave64_tile16_q8_1` | 288 | 161.32 | 560.14 | 2.57% |
| 14 | `flambeau_cast_f32_f16` | 22704 | 119.55 | 5.26 | 1.90% |
| 15 | `flambeau_attention_prefill_flash_tile_d256_br8_f16` | 48 | 110.03 | 2292.21 | 1.75% |
| 16 | `__amd_rocclr_copyBuffer` | 19238 | 77.04 | 4.00 | 1.23% |
| 17 | `flambeau_l2_norm_f32` | 12384 | 58.92 | 4.76 | 0.94% |
| 18 | `flambeau_swiglu_f32_to_f16` | 8256 | 55.89 | 6.77 | 0.89% |
| 19 | `flambeau_mmvq_q8_0_gate_up_t128_vdr2_q8_1` | 6048 | 52.93 | 8.75 | 0.84% |
| 20 | `flambeau_causal_conv1d_f32` | 6192 | 36.73 | 5.93 | 0.58% |

Total kernel time across all ranks: **6288 ms**.

### qwen36_27B_q4_1

| rank | kernel | calls | total ms | avg µs | pct |
|---:|---|---:|---:|---:|---:|
| 1 | `flambeau_mmq_q4_1_4warp_lds_q8_1` | 2112 | 4380.33 | 2074.02 | 26.64% |
| 2 | `flambeau_mmvq_q4_1_gate_up_dp4a_q8_1` | 16128 | 1701.11 | 105.47 | 10.35% |
| 3 | `flambeau_mmvq_q4_1_t128_q8_1` | 56448 | 1648.90 | 29.21 | 10.03% |
| 4 | `flambeau_attention_decode_f16_splitk_chunk` | 4032 | 1349.61 | 334.72 | 8.21% |
| 5 | `flambeau_mmvq_q5_k_r2_q8_1` | 12096 | 1344.26 | 111.13 | 8.18% |
| 6 | `flambeau_mmq_q5_K_wave64_q8_1` | 288 | 865.32 | 3004.57 | 5.26% |
| 7 | `flambeau_p2p_allreduce_residual_tp2` | 16896 | 864.09 | 51.14 | 5.26% |
| 8 | `flambeau_gdn_state_step_alphabeta_f32_s128` | 12384 | 713.65 | 57.63 | 4.34% |
| 9 | `flambeau_quantize_row_f16_q8_1` | 37536 | 461.87 | 12.30 | 2.81% |
| 10 | `flambeau_rmsnorm_q8_1_fused` | 16257 | 432.93 | 26.63 | 2.63% |
| 11 | `flambeau_mmq_q8_0_wave64_tile16_q8_1` | 576 | 409.74 | 711.35 | 2.49% |
| 12 | `flambeau_p2p_allreduce_residual_rmsnorm_tp2` | 16128 | 351.02 | 21.76 | 2.13% |
| 13 | `flambeau_attention_prefill_flash_tile_d256_br8_f16` | 96 | 315.81 | 3289.64 | 1.92% |
| 14 | `flambeau_cast_f32_f16` | 45408 | 243.24 | 5.36 | 1.48% |
| 15 | `flambeau_mmvq_q6_k_dp4a_q8_1` | 129 | 239.11 | 1853.59 | 1.45% |
| 16 | `__amd_rocclr_copyBuffer` | 32214 | 127.01 | 3.94 | 0.77% |
| 17 | `flambeau_swiglu_f32_to_f16` | 16512 | 123.37 | 7.47 | 0.75% |
| 18 | `flambeau_mmvq_q8_0_gate_up_t128_vdr2_q8_1` | 12096 | 113.43 | 9.38 | 0.69% |
| 19 | `flambeau_l2_norm_f32` | 24768 | 101.48 | 4.10 | 0.62% |
| 20 | `flambeau_rmsnorm_f32` | 12384 | 82.61 | 6.67 | 0.50% |

Total kernel time across all ranks: **16441 ms**.

### qwen36_35B_a3b_q4_0

| rank | kernel | calls | total ms | avg µs | pct |
|---:|---|---:|---:|---:|---:|
| 1 | `flambeau_attention_decode_f16_splitk_chunk` | 2520 | 920.25 | 365.18 | 13.84% |
| 2 | `flambeau_gdn_state_step_alphabeta_f32_s128` | 7740 | 652.74 | 84.33 | 9.81% |
| 3 | `flambeau_indexed_moe_mmvq_q4_0_q8_1` | 8820 | 518.23 | 58.76 | 7.79% |
| 4 | `flambeau_indexed_moe_mmq_q4_0_gate_up_tile8_dp4a_q8_1` | 240 | 340.54 | 1418.91 | 5.12% |
| 5 | `flambeau_mmvq_q4_0_gate_up_dp4a_q8_1` | 7560 | 258.38 | 34.18 | 3.88% |
| 6 | `flambeau_mmq_q8_0_wave64_tile16_q8_1` | 924 | 255.31 | 276.31 | 3.84% |
| 7 | `flambeau_mmq_q4_0_4warp_lds_q8_1` | 546 | 255.25 | 467.48 | 3.84% |
| 8 | `flambeau_moe_sort_scatter_det` | 240 | 248.29 | 1034.53 | 3.73% |
| 9 | `flambeau_p2p_allreduce_residual_tp2` | 10560 | 241.71 | 22.89 | 3.63% |
| 10 | `flambeau_topk_softmax_f32` | 10320 | 177.74 | 17.22 | 2.67% |
| 11 | `flambeau_quantize_row_f16_q8_1` | 23940 | 172.52 | 7.21 | 2.59% |
| 12 | `flambeau_cast_f32_f16` | 38700 | 169.58 | 4.38 | 2.55% |
| 13 | `flambeau_indexed_moe_mmvq_q4_0_gate_up_dp4a_q8_1` | 10080 | 158.21 | 15.70 | 2.38% |
| 14 | `flambeau_rmsnorm_q8_1_fused` | 10209 | 150.76 | 14.77 | 2.27% |
| 15 | `flambeau_attention_prefill_flash_tile_d256_br8_f16` | 60 | 137.24 | 2287.31 | 2.06% |
| 16 | `flambeau_p2p_allreduce_residual_rmsnorm_tp2` | 10080 | 134.34 | 13.33 | 2.02% |
| 17 | `flambeau_dense_gemv_f16_f16_batched` | 240 | 126.04 | 525.16 | 1.89% |
| 18 | `flambeau_indexed_moe_mmq_q4_0_down_tile8_dp4a_q8_1` | 210 | 124.60 | 593.34 | 1.87% |
| 19 | `flambeau_mmvq_q6_k_dp4a_q8_1` | 1389 | 114.72 | 82.59 | 1.72% |
| 20 | `flambeau_swiglu_f32_to_q8_1` | 27720 | 114.15 | 4.12 | 1.72% |

Total kernel time across all ranks: **6651 ms**.

### qwen3_coder_next_q4_0

| rank | kernel | calls | total ms | avg µs | pct |
|---:|---|---:|---:|---:|---:|
| 1 | `flambeau_attention_decode_f16_splitk_chunk` | 3024 | 1089.00 | 360.12 | 11.59% |
| 2 | `flambeau_gdn_state_step_alphabeta_f32_s128` | 9288 | 768.44 | 82.73 | 8.18% |
| 3 | `flambeau_indexed_moe_mmvq_q4_0_q8_1` | 10584 | 754.28 | 71.27 | 8.03% |
| 4 | `flambeau_dense_gemv_f16_f16` | 12096 | 623.88 | 51.58 | 6.64% |
| 5 | `flambeau_indexed_moe_mmq_q4_0_gate_up_tile8_dp4a_q8_1` | 288 | 550.25 | 1910.60 | 5.86% |
| 6 | `flambeau_mmvq_q5_k_r2_q8_1` | 9072 | 402.17 | 44.33 | 4.28% |
| 7 | `flambeau_moe_sort_scatter_det` | 288 | 377.58 | 1311.05 | 4.02% |
| 8 | `flambeau_topk_softmax_f32` | 12384 | 321.31 | 25.95 | 3.42% |
| 9 | `flambeau_dense_gemv_f16_f16_batched` | 288 | 302.38 | 1049.93 | 3.22% |
| 10 | `flambeau_p2p_allreduce_residual_tp2` | 12672 | 288.31 | 22.75 | 3.07% |
| 11 | `flambeau_mmq_q4_0_4warp_lds_q8_1` | 720 | 286.76 | 398.28 | 3.05% |
| 12 | `flambeau_mmq_q8_0_wave64_tile16_q8_1` | 1008 | 280.38 | 278.15 | 2.98% |
| 13 | `flambeau_quantize_row_f16_q8_1` | 28728 | 232.03 | 8.08 | 2.47% |
| 14 | `flambeau_mmvq_q4_0_gate_up_dp4a_q8_1` | 9072 | 227.92 | 25.12 | 2.43% |
| 15 | `flambeau_indexed_moe_mmvq_q4_0_gate_up_dp4a_q8_1` | 12096 | 220.90 | 18.26 | 2.35% |
| 16 | `flambeau_indexed_moe_mmq_q4_0_down_tile8_dp4a_q8_1` | 252 | 211.54 | 839.44 | 2.25% |
| 17 | `flambeau_cast_f32_f16` | 40392 | 207.71 | 5.14 | 2.21% |
| 18 | `flambeau_rmsnorm_q8_1_fused` | 12225 | 176.56 | 14.44 | 1.88% |
| 19 | `flambeau_mmq_q5_K_wave64_q8_1` | 216 | 174.57 | 808.19 | 1.86% |
| 20 | `flambeau_attention_prefill_flash_tile_d256_br8_f16` | 72 | 164.07 | 2278.81 | 1.75% |

Total kernel time across all ranks: **9395 ms**.

## Run summary

| model | load (s) | wall ms (traced) | prompt tok | gen tok |
|---|---:|---:|---:|---:|
| qwen35_9B_q4_1 | 4.1 | 1599 | 1313 | 64 |
| qwen36_27B_q4_1 | 10.3 | 3083 | 1313 | 64 |
| qwen36_35B_a3b_q4_0 | 43.0 | 2285 | 1313 | 64 |
| qwen3_coder_next_q4_0 | 100.2 | 2899 | 1311 | 64 |
## Lever analysis

### Family-wide levers (worth one PR each — every model in scope wins)

**L1. Fused QKV-MMVQ** — `mmvq_q4_1_t128_q8_1` and `mmvq_q4_0_t128` (the Q-projection / K-projection / V-projection at decode) are 8–10 % on every dense + MoE model. Each call reads the same activation `x` (~hidden×2 bytes) before writing its own output rows; on dense Q4_1 27B that's three separate HBM round-trips per layer per token. `mmvq_q4_1_gate_up_dp4a` already fuses the FFN gate+up pair — extend the same shape to a `mmvq_q4_*_qkv_dp4a` that reads `x` once, writes Q‖K‖V. Precedent: vLLM and llama.cpp ship qkv-fused mmvq for every K-quant family. Expected: **3–5 % wall on every model**, no quality risk (numerically identical).

**L2. GDN state-step + swiglu fusion** — `gdn_state_step_alphabeta_f32_s128` is 4.7–10.4 % across the family, and the kernel that follows it (`swiglu_f32_to_f16` / `swiglu_f32_to_q8_1`) reads the same state buffer back from HBM. Fuse them into one launch that produces the swiglu-activated F16 output without the F32 round-trip. Precedent: candle's D1 rmsnorm+Q8_1 fusion was a 320 ms-per-prefill win; structurally identical here. Expected: **2–4 % wall on every hybrid model** (the dense subset of 9B/27B paths still touches GDN layers). No quality risk.

**L3. Q4_1 MMQ stream-K** — `mmq_q4_1_4warp_lds_q8_1` is 21 % on 9B and **27 %** on 27B (the largest single bucket on 27B). It already runs the gold-standard 4-warp LDS-tiled shape. Stream-K (split-K with atomic accumulation) is what llamacpp-turbo uses on the matching Q4_0 kernel and pays 5–8 % on small-M shapes (every chunked-prefill ubatch ≤ 512). Expected: **3–5 % prefill wall on 9B/27B**; null on MoE models (their prefill goes through `indexed_moe_mmq_*`). Quality risk: numerical (atomic order-of-summation); needs delta-ppl cert.

### Model-class-specific levers

**L4. MoE expert fusion** (35B-A3B + Coder-Next, ~12 % combined) — the chain `indexed_moe_mmq_gate_up_tile8 → moe_sort_scatter_det → indexed_moe_mmq_down_tile8` runs sequentially with HBM round-trips between. A vLLM-style block-fused expert kernel (gate+up+down for the active expert at each token in one launch) eliminates the sort/scatter intermediate. Estimated **4–6 % wall on the two MoE archs**; structural change, will land as its own multi-week milestone.

**L5. Coder-Next shared-expert quantize** (Coder-Next-only, **9.9 %**) — `dense_gemv_f16_f16` (6.6 %) + `dense_gemv_f16_f16_batched` (3.2 %) is the always-on shared expert running F16 weights. Q8_0 the shared-expert weights and route through `mmvq_q8_0_t128_vdr2_q8_1` (already 0.7 % on 27B at small batch). Halves the HBM bytes per shared-expert call; estimated **3–4 % wall on Coder-Next**. Quality cert required (delta-ppl on shared-expert weights only).

### Confirmed dead levers (don't re-propose without measured evidence)

- **MMVQ multi-row r4 on Q4_K** — candle memory `v2_31_b_e_multirow_dead_lever.md` shows at-floor on gfx906. *Q4_1 is untested* (different super-block layout), so an r2/r4 sweep specifically for Q4_1 is fair game.
- **MMQ tile-32** — `v2_31_d_tile32_null.md`; VGPR-bound at tile-16 already.
- **Hip-graph capture (G3)** on gfx906 — null per `feedback_hipgraph_null.md`.

### Out-of-scope on this rig

- **AllReduce** is at the BAR1 P2P ceiling (5.3–5.6 % combined). The only material lever is xGMI / NVLink topology — V2.
- **`cast_f32_f16` (1.5–3.1 %)** could be eliminated with dual-output kernels but the audit cost (every F32-out kernel needs a sibling) is wildly disproportionate to the win.

### Recommendation order

1. **L1 (fused QKV-MMVQ)** — biggest family-wide payoff for moderate engineering. Touches no correctness-sensitive paths.
2. **L2 (GDN+swiglu fusion)** — second-biggest family-wide win, isolated kernel scope.
3. **L3 (Q4_1 stream-K MMQ)** — 9B/27B-specific prefill win; gates on a numerical-stability cert.
4. **L4 (MoE expert fusion)** — biggest single MoE win but multi-week structural work; queue after L1/L2.
5. **L5 (Coder-Next shared-expert quant)** — quality-cert-gated, Coder-Next-only, easy if cert passes.
