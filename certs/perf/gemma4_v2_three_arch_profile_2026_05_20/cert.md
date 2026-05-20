# gemma4-v2 three-arch profile — E4B vs 31B vs 26B-A4B

Date: 2026-05-20
Goal: identify kernels hot across ≥2 of the three gemma4 variants to
prioritise levers that compound across the family.

## Bench setup

Real-task prompt (~100w technical request) → 128 decode tokens.
rocprofv3 --kernel-trace wrapping `flambeau serve`. Warmup +
measurement; CSV at `certs/perf/gemma4_v2_three_arch_profile_2026_05_20/`.

| arch | dtype | topology | devices | ctx | pp_tok | tg_tok | wall_ms | k-time |
|---|---|---|---|---:|---:|---:|---:|---:|
| E4B (#256 path) | Q4_0 | SD | hip:0 | 4096 | 291 | 128 | 3430 | 3351ms |
| 31B (dense)     | Q4_0 | PP2 | hip:1,2 | 2048 | 300 | 128 | 9866 | 11187ms |
| 26B-A4B (MoE)   | Q8_0 | PP4 | hip:0,1,2,3 | 2048 | 300 | 128 | 3761 | 3503ms |

Caveats:
- Topology forced by VRAM constraints (GPU0 had 6.6 GB leaked from
  earlier debug session; 31B at TP2 OOM'd, forced to PP2 hip:1,2).
- PP kernel-time sums over ranks; wall is serial across stages.
- 26B-A4B PP4 across all 4 GPUs — used the {2,3} link in PP-only
  mode (no AR, safe per the warning in memory).

## Kernel-time pareto, top 12 per arch

| rank | E4B (Q4_0/SD) | 31B (Q4_0/PP2) | 26B-A4B (Q8_0/PP4) |
|---:|---|---|---|
| 1 | mmvq_q4_0_q8_1 **23.2%** | mmvq_q4_0_q8_1 **46.1%** | attn_decode_splitk_chunk **13.3%** |
| 2 | attn_decode_splitk_chunk **16.5%** | mmq_q4_0_4warp_lds_q8_1 11.1% | mmvq_q8_0_t128_vdr2 9.6% |
| 3 | mmvq_q4_k_r2_q8_1 13.4% | mmvq_q4_k_r2_q8_1 9.1% | indexed_moe_mmvq_q8_0_dp4a 9.4% |
| 4 | mmq_q4_0_4warp_lds_q8_1 5.5% | mmvq_q4_0_q8_1_f16 8.5% | indexed_moe_mmvq_q8_0_gate_up 8.1% |
| 5 | dense_gemv_f32_f16_batched 5.0% | attn_decode_splitk_chunk **8.5%** | mmvq_q8_0_t128_vdr2_f16 7.0% |
| 6 | rmsnorm_q8_1_fused 5.0% | rmsnorm_q8_1_fused 4.5% | mmq_q8_0_oracle 6.3% |
| 7 | **copyBuffer 4.0%** | **copyBuffer 2.3%** | **rmsnorm_f16 5.3%** |
| 8 | mmvq_q4_0_q8_1_f16 3.8% | quantize_row_f16_q8_1 1.5% | mmq_q8_0_wave64_tile16 4.7% |
| 9 | **rmsnorm_f16 3.8%** | rmsnorm_f32_to_f16 1.5% | **copyBuffer 3.7%** |
| 10 | dense_gemv_f32_f16 3.1% | mmvq_q4_1_t128_q8_1 1.3% | mmvq_q8_0_gate_up_t128_vdr2 3.5% |
| 11 | add_f16 2.4% | **rmsnorm_f16 1.2%** | quantize_row_f16_q8_1 3.5% |
| 12 | rmsnorm_f32_to_f16 2.3% | rope_neox_partial_f16 0.8% | rmsnorm_q8_1_fused 2.1% |

## Common kernels (hot across ≥2 archs)

### 3-of-3 — the universal levers

| kernel | E4B | 31B | 26B | notes |
|---|---:|---:|---:|---|
| `attention_decode_f16_splitk_chunk` | 16.5% | 8.5% | 13.3% | top-5 in all three |
| `rmsnorm_f16` | 3.8% | 1.2% | 5.3% | per-layer post-norm |
| `__amd_rocclr_copyBuffer` | 4.0% | 2.3% | 3.7% | KV-append (DtoD) + position upload (HtoD) + PLE table for E4B |
| `quantize_row_f16_q8_1` | 2.2% | 1.5% | 3.5% | activation quant for dp4a MMVQ |
| `add_f16` / residual | 2.4% | 0.8% | 1.2% | per-layer fan-in |
| `rope_neox_partial_f16` | 1.5% | 0.8% | 1.2% | per-layer Q/K rope |

### 2-of-3 — dense Q4_0 lane (E4B + 31B)

| kernel | E4B | 31B | notes |
|---|---:|---:|---|
| `mmvq_q4_0_q8_1` | 23.2% | **46.1%** | decode-dominant; the single biggest decode cost |
| `mmq_q4_0_4warp_lds_q8_1` | 5.5% | 11.1% | prefill MMQ |
| `mmvq_q4_0_q8_1_f16` | 3.8% | 8.5% | F16-output variant (lever-1) |
| `mmvq_q4_k_r2_q8_1` | 13.4% | 9.1% | likely output_head (Q4_K vocab matmul) |

### E4B-only (lever-2 paths)

| kernel | E4B | notes |
|---|---:|---|
| `dense_gemv_f32_f16_batched` | 5.0% | PLE apply (gate / proj) at n_tokens>1 |
| `dense_gemv_f32_f16` | 3.1% | PLE apply at decode (n=1) |
| `rmsnorm_f32_to_f16` | 2.3% | PLE post-norm path |

### 26B-A4B-only (MoE lane)

| kernel | 26B | notes |
|---|---:|---|
| `indexed_moe_mmvq_q8_0_dp4a_q8_1` | 9.4% | per-expert decode MMVQ (down) |
| `indexed_moe_mmvq_q8_0_gate_up_dp4a_q8_1` | 8.1% | per-expert decode MMVQ (gate+up fused) |
| `indexed_moe_mmq_q8_0_gate_up_tile8_dp4a_q8_1` | 5.8% | per-expert prefill MMQ |
| `mmvq_q8_0_t128_vdr2_q8_1` (+f16 + gate_up) | 9.6+7.0+3.5 = 20% | dense decode mmvq (Q8_0 attn / shared-MLP) |
| `topk_softmax_f32` | 1.8% | router |

## Recommended lever priority

### Tier 1 — compound across all three archs

**L1: Attention-decode kernel (8–16% everywhere).**
`attention_decode_f16_splitk_chunk` is in the top-5 of every arch.
Already gained from tile-2 (commit 43cf099). Next steps:
- Tile-N (process N K-rows per outer iter for further LDS reuse).
- Better split-K dispatch: currently dispatches splitk when
  `n_tokens_kv > 256` AND `n_chunks > 1`. For short contexts (< 1024)
  the chunk-overhead is sizeable; revisit the threshold.
- For 26B-A4B (full-attention 1/6th layers + GDN-less dense path) the
  ratio is 13% — same kernel, same lever applies.

**L2: Promote rmsnorm_q8_1_fused everywhere quant-follows
(1–5% per arch + 1.5–3.5% on the quantize half).** Today's profile
shows both `rmsnorm_f16` AND `quantize_row_f16_q8_1` separately
hot — the fused variant collapses them. The fused version exists
(`rmsnorm_q8_1_fused`, also hot at 2–5%) but isn't used at every
quantize-follows-norm site. Audit the call sites.

**L3: copyBuffer audit (2–4% across all).** Hot use sites:
- Per-layer position-buffer HtoD at prefill (RoPE input).
- KV-append DtoD (one per layer per token at decode).
- PLE build DtoH + HtoD (E4B only — already host-bound; lever-1
  moved the matmul to GPU but the build-finishing DtoH+HtoD remains).
For PLE specifically, finishing the build on-GPU (rmsnorm + add
fused) would save ~22 MB DtoH + 22 MB HtoD per E4B prefill chunk.

### Tier 2 — dense Q4_0 only (E4B + 31B)

**L4: mmvq_q4_0_q8_1 occupancy hunt (23–46%).** The single biggest
decode cost on dense Q4_0. Already has dp4a. Next investigation:
- VGPR pressure vs occupancy.
- Wider row tile to amortise weight HBM across more outputs.
- The MMVQ-DP4A `r2` half-warp pattern is in `mmvq_q4_k_r2_q8_1`
  (Q4_K) but not in Q4_0 — port it.

**L5: mmvq_q4_k_r2_q8_1 — output_head at Q4_K (9–13%).** Likely the
LM-head matmul (vocab × hidden). One launch with `n_rows = vocab`
is heavy. Either:
- Verify the `r2` half-warp reduce is already the best;
- Cache cumulative head activations and skip recompute at decode
  (only the last hidden state matters).

### Tier 3 — arch-specific

**26B-A4B (MoE):** dense Q8_0 mmvq family (`mmvq_q8_0_t128_vdr2*`)
sums to ~20% and the indexed-MoE family sums to ~23%. Both are
already DP4A-fused. Next would be a bigger LDS tile for the MoE
mmvq.

**E4B (PLE):** the apply chain `dense_gemv_f32_f16_batched` +
`dense_gemv_f32_f16` sums to 8.1%. Fusing the apply chain
(gate→gelu→mul→proj→norm→add) into one kernel per layer would
crush this — already on the lever roadmap.

## Ratio framing

E4B prefill 558 t/s vs llama.cpp 1046 = 0.53× (post-lever-2).
E4B decode 41 t/s vs llama.cpp 71 = 0.58×.
**L1 + L2 + L3 together would move both numbers by ~10–20%** on E4B,
since they hit exactly the kernels that dominate the per-layer loop.
Same shape on 31B and 26B.

## CSVs

- `e4b/threadreaper/14803_kernel_trace.csv`
- `31b_q4_pp2/threadreaper/16092_kernel_trace.csv`
- `26b_a4b_pp4/threadreaper/16361_kernel_trace.csv`
