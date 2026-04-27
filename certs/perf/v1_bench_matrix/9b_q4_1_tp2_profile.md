# V1-BENCH-#114 — 9B-Q4_1 tp2 prefill L=512 profile

GGUF: `/artefact/models/Qwen3.5-9B-Q4_1.gguf`
Rig: 4× MI50, 100W cap, ROCm 7.1.1.
Topology: tp2 (devices 0,1), `FLAMBEAU_TP_BATCHED=1` (AUTO-6 default).
Wall: 923.96 tok/s (554 ms / 512 tokens) — bench cert says 951.7; rocprofv3 overhead.

## Method

Wrote `crates/models/qwen3-moe/tests/profile_point_tp.rs` (TP sister of
the existing `profile_point.rs`). Wrapped with rocprofv3 7.1.1 in
`--kernel-trace --stats` mode. **`--kernel-trace` does NOT trip the
multi-rank SIGABRT documented in V2.30 memory** — that crash was specific
to PMC counter finalize, not kernel-dispatch trace finalize. Both ranks
produced clean stats.

Captured 1 warmup prefill + 1 measured prefill = 2 forward passes.
Counts in the table below are 2× a single-prefill count.

## Top kernels by wall time

| Rank | Kernel | Calls | Total ms | % of wall |
|-----:|-------|------:|--------:|---------:|
| 1 | `flambeau_mmq_q4_1_4warp_lds_q8_1`    | 352 | 489.3 | **50.56 %** |
| 2 | `flambeau_p2p_allreduce_residual_tp2` | 192 | 100.1 | 10.34 % |
| 3 | `flambeau_mmq_q5_K_wave64_q8_1`       |  48 |  88.1 |  9.10 % |
| 4 | `flambeau_gdn_state_step_alphabeta_f32_s128` | 96 | 78.2 | 8.08 % |
| 5 | `flambeau_mmq_q8_0_wave64_tile16_q8_1`|  96 |  54.0 |  5.59 % |
| 6 | `flambeau_quantize_row_f16_q8_1`      | 352 |  46.4 |  4.80 % |
| 7 | `flambeau_attention_prefill_flash_tile_d256_br8_f16` | 16 | 17.5 | 1.81 % |
| 8 | `flambeau_cast_f32_f16`               | 352 |  14.2 |  1.47 % |
| 9 | `flambeau_quantize_f16_q8_1_mmq`      | 208 |  11.9 |  1.23 % |
| 10 | `__amd_rocclr_copyBuffer`            | 2646 | 10.8 |  1.12 % |

Full CSV: `9b_q4_1_tp2_l512_kernel_stats.csv`.

## Findings

1. **mmq_q4_1_4warp_lds dominates at 50.56 % of wall** — this is the
   same kernel that V2.12 measured at 33.4 % on Mesh<1>. TP-2 sharded
   the work but the kernel still owns half the wall. **Single biggest
   lever.**
2. **AR (`p2p_allreduce_residual_tp2`) is 10.34 %** — non-trivial TP
   tax. A fused AR + residual-add already exists (TP-perf path), so
   reducing this needs either lower-frequency AR (one per layer instead
   of one per attn + one per ffn = two per layer) or a faster AR
   primitive (V2.x AR-on-aux-stream, etc.).
3. **mmq_q5_K_wave64 + mmq_q8_0_wave64_tile16 = 14.69 % combined** —
   Qwen3.5-9B-Q4_1 has Q5_K (`ssm_out`?) + Q8_0 sub-tensors. Same
   wave64 family that V2.31.d showed is at the local optimum on gfx906.
4. **gdn_state_step at 8.08 %** — 9B has GDN layers despite the
   "dense" naming. Already serialised post-V2.30.a; further fusion is
   the V2.x lever.
5. Quantise + cast helpers (rmsnorm_q8_1_fused excluded; that's row 23
   at 0.16 %) total ~7 %. Already within "small ops" budget.

## Path forward for #115 (the actual fix)

The 50.56 % kernel is **`mmq_q4_1_4warp_lds`** — and per memory, every
prior attempt has measured null vs the current 4warp_lds:

- V2.13.a wave64: null
- V2.29.e wave64_tile16: null at 100 W
- V2.31.f re-test: null
- V2.31.d / V2.9.b confirmed: doubling FP32 accumulators per thread is
  an anti-pattern on gfx906.

Options for #115 that have NOT been tried:
- (a) Multi-row MMVQ at large M (P29-style r2/r4 over rows of the
  weight matrix). MMVQ usually wins at M=1; multi-row variants amortise
  weight loads at moderate M. Untested for Q4_1 at M=128+.
- (b) Restructure to call the kernel in fewer launches (block-batched).
  Currently 352 launches over 2 prefills = 176 per prefill = ~16 layers
  × 11 calls/layer. Per-layer fusion of attn + ffn proj could amortise.
- (c) gfx906-specific: investigate `__builtin_amdgcn_mfma_f32_*` (does
  not exist on gfx906), or DPP-permute-based reduction shapes that
  weren't tried in 4warp_lds.

These are all multi-session structural experiments.

## Closes

- #114 #106a — profile captured + top hot-spot identified.
- #114a #121 — TP profile_point harness in-tree (also reusable for
  any TP topology / model).

## Does NOT close

- #115 #106b — actual kernel fix; needs a new structural angle (above).
