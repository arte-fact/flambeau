# V2.22.b — Q8_0 indexed-MoE tile8 MMQ

**Source of truth for this work:** `doc/V2_22B-SCOPE.md`. Delete after V2.22.b.6 lands.

## Why

V2.30 profile (2026-04-23) — see `certs/perf/profiles/v2_30_*_kernel_stats.csv`:

- **35B-A3B-UD-Q8_K_XL prefill L=512**: `indexed_moe_mmvq_q8_0_dp4a_q8_1` is **77.0 %** of GPU time (1.62 s / 2.14 s). Flambeau 239 tok/s vs llama.cpp 415 → **0.58×**.
- **35B-A3B-Q4_0 prefill L=512**: 5 mixed-quant layers (Q4_1→Q8_0 at load) fall through to Q8_0 MMVQ for **14.0 %** of prefill, and the Q4_0 half of those same layers lands on `indexed_moe_mmvq_q4_0_q8_1` for another **9.3 %**. Flambeau 603 tok/s vs llama.cpp 1113 → **0.54×**.

Both are a single missing kernel: V2.6.b's Q4_K tile8 structure ported to Q8_0. The fallback is explicit in `crates/models/qwen3-moe/src/forward/moe.rs:1009-1015`:
> *"Mixed (Q4_0 gate / Q8_0 down …) stays on the MMVQ fast path until a Q8_0 indexed-MoE down tile8 kernel lands (V2.22.b future work)."*

Applying V2.6.b's ~3× per-kernel ratio:
- UD-Q8_K_XL prefill 239 → ≈ 480 tok/s (ratio 1.15×).
- Q4_0 prefill 603 → ≈ 720 tok/s (ratio ≈ 0.65×; residual is attn Q4_0 MMQ + router).

## What

Port V2.6.b's `indexed_moe_mmq_q4_k_{gate_up,down}_tile8_dp4a.cu` structure to Q8_0. Q8_0 is structurally simpler than Q4_K:
- block = 32 int8 + 1 fp16 scale (no sub-block scales, no `dmin`, no super-block decode)
- `n_sb_per_row = hidden / 32` (one Q8_0 block per k-stride of 32, matches `QK8_1` exactly)
- Inner loop is flat: `for ib in 0..n_sb_per_row` — no `sub` loop, no `half/il` nibble decode

Pad-to-8 sort already ships (`moe_sort_by_expert_padded` + `flambeau_moe_sort_pad_copy`). No new sort infra needed.

## Decomposition (session-sized)

### Session A (this issue's first session)

- **V2.22.b.1** — Author `kernels-hip/src/kernels/indexed_moe_mmq_q8_0_gate_up_tile8_dp4a.cu`.
  - `MMQ_Y=64`, `TILE_N=8`, `__launch_bounds__(WARP_SIZE, 1)` (match Q4_K precedent — attempt `(_, 2)` is V3).
  - Weights: load `gbx->d, ubx->d` (fp16) + 8 packed int32s per block per thread.
  - Y: same Q8_1 layout as Q4_K gate_up (`by->d`, 8×int32 per block).
  - Inner `c` loop: 8 DP4As for gate, 8 for up, 8 for `sumi_y` (constant 0x01010101). Same `d8 * (sumi_w * d_w)` with Q8_0 there's no dmin so the formula collapses to `sums += d8 * d_w * sumi_w`.
- **V2.22.b.2** — Author `indexed_moe_mmq_q8_0_down_tile8_dp4a.cu` (clone of V2.6.b down kernel minus Q4_K sub-block decode).
- **V2.22.b.3** — Register stems in `crates/ops/src/hip/mod.rs::KERNEL_STEMS`:
  - `indexed_moe_mmq_q8_0_gate_up_tile8_dp4a`
  - `indexed_moe_mmq_q8_0_down_tile8_dp4a`
  - Build-side registration (`build.rs` is recursive over `src/kernels/*.cu` so the files pick up automatically).
- **V2.22.b.4** — Ops wrappers in `crates/ops/src/hip/moe.rs`: mirror `indexed_moe_mmq_q4_k_gate_up_tile8` / `indexed_moe_mmq_q4_k_down_tile8`. Same `MoeShape` signature; only the weight pointer type changes and `n_sb_per_row` is now `hidden/32` / `inter/32`.
- **V2.22.b.5** — KernelDescriptors in `crates/backend-hip/src/impls.rs` + dispatch row in `dispatch/hip/gfx906.toml`.

### Session B

- **V2.22.b.6** — Cert sweep: extend `bench sweep --op indexed_moe_mmq` to emit `certs/hip/indexed_moe_mmq_q8_0_{gate_up,down}_tile8_dp4a.json`. Shapes: min 3 sizes × 3 n_tokens (cover 32, 128, 512). Gate greens before routing.
- **V2.22.b.7** — Wire through `crates/models/qwen3-moe/src/forward/moe.rs`:
  - Add `gate_dt_pre == Q8_0` branch parallel to the Q4_0 tile8 branch at line 1120–1143 (needs `q8_0_use_tile8 = gate_dt == Q8_0 && down_dt == Q8_0 && n_tokens >= 32`).
  - Add `gate_dt_pre == Q4_0 && down_dt_pre == Q8_0` path: Q4_0 gate+up via existing `indexed_moe_mmq_q4_0_gate_up_tile8`, new Q8_0 down via tile8.
  - Keep MMVQ fallback for `n_tokens < 32`.
- **V2.22.b.8** — Parity: bit-exact on Qwen3.6-35B-A3B-UD-Q8_K_XL seed 9419 (8 tokens). Cert path: `certs/parity/qwen3_6_35b_a3b_ud_q8_k_xl_greedy_mesh4.json`.
- **V2.22.b.9** — Bench: refresh `certs/perf/qwen3_6_35b_a3b_ud_q8_k_xl_mesh{2,4}.json` and `qwen3_6_35b_a3b_q4_0_mesh{2,4}.json`; update head-to-head. Expected: UD-Q8_K_XL Mesh<4> pp=512 ≥ 450 tok/s; Q4_0 Mesh<4> pp=1024 ≥ 700 tok/s.

## Invariants to preserve

- **Padding idempotency.** V2.6.b uses repeat-last padding so tail slots redundantly compute the same output as the last real slot and their stores either overwrite-equal or are themselves overwritten. Same guarantee for Q8_0 — don't add per-slot validity checks in the hot path.
- **Layout.** gate/up weight tensors are `[n_experts, n_rows, n_sb_per_row]` of `flambeau_block_q8_0` = `{ __half d, int8_t qs[32] }`. Y is `[n_tokens, n_sb_per_row]` of `flambeau_block_q8_1` (already produced by `quantize_row_f16_q8_1` upstream, same path that feeds Q4_K tile8).
- **VGPR budget.** Q4_K tile8 is 156 B scratch, 128 VGPR, 1 wave/SIMD. Q8_0 eliminates the 8×uint8 sub-scale arrays + the gbx/ubx super-block pointer caching → fewer live regs. Expect `__launch_bounds__(WARP_SIZE, 1)` to leave headroom; don't over-tune occupancy before microbench PMC confirms it's the bottleneck (see `feedback_pmc_ratio_not_wallclock`).
- **Parity gate before dispatch.** New kernel ships behind `#[cfg(unverified)]` until V2.22.b.6's cert is green. Don't switch the dispatch row over in V2.22.b.5 beyond a stub; the actual runtime path flip is V2.22.b.7.

## Rules of thumb for the port

- Skipping `dmin`/sub-block machinery vs Q4_K tile8 removes ~30 LOC from the inner loop and ~8 VGPR. Do NOT re-introduce a "general Q4_K-shaped" skeleton that nests a trivial sub-loop — the Q8_0 inner `c` loop is 3 DP4As (gate/up/sumi_y), not 24.
- Do NOT author a tile16 variant in Session A. V2.31.b proved tile16 regressed Q4_K tile8 by −33 % on 35B-A3B because tile8 is already compute-bound (`memory/project_v2_31_tile16_null.md` lessons). Q8_0's compute-to-memory ratio may be different, but that's a V2.32+ question — finish the tile8 port first.
- rocprofv3 on multi-agent (4-rank) traces requires **rocm-6.3.4**, NOT rocm-7.1.1 (which SIGABRTs on `Information of the agent with handle: 0 is not present`). Use `/opt/rocm-6.3.4/bin/rocprofv3 --kernel-trace --stats` for V2.22.b.9 profiling.
- Use `FLAMBEAU_PREFILL_ONLY=1 FLAMBEAU_PREFILL_L=512` (not `FLAMBEAU_PREFILL_SINGLE_L`) to scope bench to a single prefill length.
