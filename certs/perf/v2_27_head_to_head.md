# V2.27 — flambeau vs llama.cpp head-to-head, all 6 supported Qwen3 GGUFs

**Rig**: 4×MI50 PCIe-only (no xGMI / no NVLink peer), ROCm 7.1.1.
**llama.cpp**: build `2799d933b` (8880), `-fa 1 -ngl 99 -sm layer`.
**flambeau**: HEAD at V2.26 (commit `d43b2f7`), per-layer Mesh<N> PP.
**Config**: pp=512, tg=64, single rep each side.

Discovered during this run: **V2.12's reported "llama.cpp crashes on Qwen3MoE" was a misconfigured rocBLAS Tensile path**. Setting `ROCBLAS_TENSILE_LIBPATH=/opt/rocm-7.1.1/core-7.13/lib/rocblas/library` (the core-7.13 subdir, which has 269 gfx906 files vs 255 in the top-level `lib/rocblas/library`) unblocks every MoE model. That correction is why this bench goes wider than V2.12's head-to-head.

## Results

| Model | Mesh | Size | flambeau pp | llama.cpp pp | flambeau tg | llama.cpp tg |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B-Q4_1 | 1 | 5.43 GiB | 817.9 | **988.6** | 65.6 | **72.5** |
| Qwen3.6-27B-Q8_0 | 4 | 26.62 GiB | **135.2** | 135.2 | 19.3 | **19.5** |
| Qwen3.6-27B-UD-Q8_K_XL | 4 | 32.89 GiB | 56.2 | **141.9** | **16.9** | 16.8 |
| Qwen3.6-35B-A3B-UD-Q4_K_S | 4 | 19.45 GiB | 683.1 | **872.6** | 52.4 | **58.3** |
| Qwen3.6-35B-A3B-UD-Q8_K_XL | 4 | 35.80 GiB | **240.9** | **OOM** | **48.8** | **OOM** |
| Qwen3.6-35B-A3B-Q4_0 | 4 | 18.56 GiB | 132.1 | **1118.4** | 47.2 | **62.3** |

Ratios flambeau / llama.cpp (higher = flambeau leads):

| Model | pp ratio | tg ratio |
|---|---:|---:|
| 9B-Q4_1 Mesh<1> | 0.83 | 0.91 |
| 27B-Q8_0 Mesh<4> | **1.00** | 0.99 |
| 27B-UD-Q8_K_XL Mesh<4> | 0.40 | 1.01 |
| 35B-A3B-UD-Q4_K_S Mesh<4> | 0.78 | 0.90 |
| 35B-A3B-UD-Q8_K_XL Mesh<4> | — (flambeau-only) | — |
| 35B-A3B-Q4_0 Mesh<4> | **0.12** | 0.76 |

## Update — V2.28 Q4_0 MMQ lands

Post-V2.27, the V2.28 chain shipped Q4_0 MMQ kernels (dense + indexed-MoE tile8). Updated numbers for the primary target:

| Model | Mesh | flambeau pp (V2.28) | llama.cpp pp | Δ vs V2.27 | flambeau tg (V2.28) | llama.cpp tg |
|---|---:|---:|---:|---|---:|---:|
| Qwen3.6-35B-A3B-Q4_0 | 4 | **275.1** | 1118.4 | +108 % (132 → 275) | 46.9 | 62.3 |

Prefill ratio climbed from **0.12 → 0.25** of llama.cpp. Decode unchanged at 0.75 (V2.28.d r2 port was NULL — see note below). All other models unchanged by V2.28 (Q4_0 path only).

Contributions:
- **V2.28.b** (dense Q4_0 MMQ): 132 → 163 tok/s = +24 % (closes attn_qkv / attn_gate / attn_output / ssm_out on the Q4_0 weight side)
- **V2.28.c** (indexed-MoE Q4_0 MMQ tile8, gate+up fused + down): 163 → **275 tok/s = +68 % on top of .b, +108 % cumulative** (closes the 40 / 40 layers of ffn_*_exps Q4_0 in MoE prefill)
- **V2.28.d NULL** (Q4_0 r2 decode MMVQ): −21 % decode + argmax drift. Candle P29 r2 pattern (scalar F32 per lane + half-warp reduce) strictly loses on Q4_0 because it throws away the DP4A advantage that Q4_0's flat-block structure enables. Kernel moved to `_unverified/` with diagnosis. Decode path stays on single-row DP4A.

## Remaining gap after V2.28

Still 4× behind llama.cpp on 35B-A3B-Q4_0 prefill (275 vs 1118). Breakdown:

- ~5 layers of 40 (V2.23.a Q4_1→Q8_0 for ffn_down_exps) still use MMVQ fallback for the down step — V2.22.b deferred Q8_0 indexed-MoE MMQ tile8 would close this.
- Residual sort/pad overhead at every MoE layer entry — common with Q4_K path, V2.31 MMQ_X≥16 restructure is the lever.
- Q4_0 decode still on single-row MMVQ — no clean r-family win exists per V2.28.d finding. The real decode-side lever for 35B-A3B-Q4_0 is either: (a) launch-overhead reduction via CUDA-Graph-style batching (V2.x multi-session), or (b) MoE-indexed fused decode (no candle precedent).

## Findings

### Prefill: the MMQ-coverage gap is the story

Three clusters of result, and they all line up with **"does flambeau have an MMQ tile kernel for this dtype?"**:

- **Parity where we have MMQ tile**: 27B-Q8_0 Mesh<4> is *exactly* tied (135.2 vs 135.2 prefill). Our Q8_0 wave64 tile16 MMQ (V2.7) matches llama.cpp's MMQ per-call.
- **78% of llama.cpp where MoE indexed-MMQ tile8 lives**: 35B-A3B-UD-Q4_K_S (683 vs 873). Our V2.6.b tile8 + V2.4-2.6 r-family + sort-by-expert combo is competitive. Remaining 22% gap matches the candle analysis pointer at Y-LDS (now NULL, V2.24.b) — so the residual is structural.
- **Massive gap where we only have MMVQ (row-by-row prefill)**: 35B-A3B-Q4_0 at 12% of llama.cpp (132 vs 1118). V2.23 shipped Q4_0 MMVQ only; `qmatmul()` falls back to row-by-row at L>1. Same story on 27B-UD-Q8_K_XL (56 vs 142 = 40%): V2.21/V2.25 F16 MMVQ multi-row is still one-output-per-block.

### Decode: consistently 10-15% behind, closer on UD-Q8_K_XL

- 9B-Q4_1 Mesh<1>: 91% of llama.cpp (65.6 vs 72.5). Our V2.2.d Q4_1 MMQ turbo port + V2.19.b split-K + DP4A-VDR=2 stack. Gap consistent with the V2.12 posture.
- 35B-A3B-UD-Q4_K_S Mesh<4>: 90% (52.4 vs 58.3). Split-K kicks in here but kernel-launch overhead on 40 layers × 4 ranks plus the PP hand-off keep us just short.
- 27B-Q8_0 / 27B-UD-Q8_K_XL: essentially tied (within 1%). Decode on these is dominated by Q8_0 MMVQ DP4A-VDR=2 where we're kernel-for-kernel equal.
- 35B-A3B-Q4_0 Mesh<4>: 76% (47.2 vs 62.3). Our Q4_0 decode MMVQ is single-row not r-family (V2.23 only shipped r1); llama.cpp tile kernels still win.

### Flambeau-only model: 35B-A3B-UD-Q8_K_XL

llama.cpp OOMs on Mesh<4> at 35B-A3B-UD-Q8_K_XL (35.8 GiB model) — tries to allocate 10 GB on device 0 despite `-sm layer -ts 1/1/1/1`. Root cause: llama.cpp's layer-split puts token_embd + output + output_norm all on device 0, pushing rank 0's budget past the 16 GB card. Flambeau's V1.7.5 per-layer `LayerAssignment` distributes globals more carefully (and also auto-converts the BF16 stragglers to Q8_0 at load per V2.22), so we fit 35.80 GiB into 4×16 GiB cards with 240.9 / 48.8 tok/s.

## Prioritised gaps to close

Based on the ratios, the ordered next-cycle lever list is:

1. **Q4_0 MMQ tile kernel** (8.5× prefill gap on 35B-A3B-Q4_0). Candle has this; turbo has this. Port structure mirrors V2.3.b Q4_K wave64 tile with Q4_0's simpler (no-min) bias-correction. Expected: close the gap on 35B-A3B-Q4_0 prefill from 12% to ~80% of llama.cpp. **High ROI, single session.**
2. **F16 MMQ tile kernel** (2.5× prefill gap on 27B-UD-Q8_K_XL). Same shape as V2.25.a but with LDS-shared F16 weight per tile. Expected: close 27B-UD-Q8_K_XL prefill from 40% → ~70% of llama.cpp. **Moderate ROI, one session.**
3. **Q5_0 MMQ tile** (V2.26.b deferred) — same analysis; shexp is small fraction of prefill on 35B-A3B-Q4_0, but Q4_0 MMQ fix above dominates the gap anyway. Low priority.
4. **Q4_K MMQ Y-LDS or MMQ_X≥16 restructure** (22% gap on 35B-A3B-UD-Q4_K_S). V2.24.b NULL showed naive Y-LDS regresses; the real lever is restructuring to wider tile (MMQ_X≥16) which has room for real LDS amortisation. Multi-session.

## Where flambeau is already ahead, to preserve

- **35B-A3B-UD-Q8_K_XL**: the only model in our lineup that llama.cpp cannot run on this rig. Keep the PP loader + BF16 auto-quant.
- **27B-Q8_0 prefill parity**: our V2.7 Q8_0 wave64 tile16 MMQ is matching llama.cpp kernel-for-kernel. Don't regress.
- **27B-UD-Q8_K_XL decode ≥ llama.cpp** (16.9 vs 16.8): F16 MMVQ DP4A-VDR=2 path is at parity. Our row-by-row wasn't penalising decode (single row = tile, same thing).
- **Split-K flash-decoding** (V2.19.b): still unique to flambeau; kicks in at long context ≥ 256 tokens and gives us the only hope of catching/beating llama.cpp on long-context decode.

## Methodology notes

- Single rep per side, not 3-rep median (V2.12 did medians; this run traded statistical rigour for covering 12 rows of the matrix in one session).
- Same `-sm layer` on llama.cpp as our PP topology — both split by layers, not tensors.
- `-fa 1` (flash attention) on llama.cpp; our flash-tile prefill is the default.
- No `-ub` tuning on llama.cpp; default ubatch=512. Our prefill processes L tokens at a time (L passed to scratch).
- flambeau run: `cargo test --release --features hip -p flambeau-qwen3-moe --test perf_baseline_qwen*` with default prefill grid + tg=64.
