# Kernel Implementation Inventory — Phase 0c

**Status:** read-only inventory for `code_path_cleanup_plan.md` Phase 0.
Cross-references `crates/kernels-hip/src/kernels/*.cu`,
`crates/ops/src/hip/*.rs` (launchers), `crates/backend-hip/src/impls.rs`
(KernelDescriptor / DirectCallKernel registration),
`dispatch/hip/gfx906.toml` (production dispatch table), and
`certs/hip/gfx906/*.json` (correctness certs).

## Sources

- **146** kernel `.cu` files in `crates/kernels-hip/src/kernels/`
  (production); **3** under `_unverified/`.
- **1** shared header: `crates/kernels-shared/include/block_quant.cuh`
  (no kernel functions, just block-quant struct definitions).
- **2** arch headers: `crates/kernels-hip/src/arch_primitives/{gfx906,mmq_prefetch}.cuh`
  (no kernels — gfx906 wave64 primitives + MMQ tile prefetch).
- **64 unique `impl_id` rows** in `dispatch/hip/gfx906.toml`
  (production dispatch).
- **67 registered KernelDescriptors** + DirectCallKernels in
  `crates/backend-hip/src/impls.rs`.
- **92 cert JSONs** under `certs/hip/gfx906/`.
- **137 stems** in `crates/ops/src/hip/mod.rs::KERNEL_STEMS` (the
  authoritative production module-load list — every stem here is
  loaded into `OpsRegistry` at boot).
- **`#[cfg(unverified)]` policy** (CLAUDE.md rule #10): no Rust marker
  found; kept-but-disabled kernels live as `.cu` files under
  `crates/kernels-hip/src/kernels/_unverified/` with a leading
  comment explaining the null result.

## How a kernel reaches production (4 paths)

1. **Shape-dispatched `KernelDescriptor`** — `dispatch/hip/gfx906.toml`
   row → `impls.rs::QMATMUL_GFX906` (etc.) match → `Recipe::from_impl_id`
   in `ops/src/hip/qmatmul.rs` rewrites to a kernel stem → `OpsRegistry::expect_module(stem)`.
2. **`DirectCallKernel`** — `impls.rs::DIRECT_CALL_KERNELS_GFX906` →
   one stem per `(op, dtype)`, called via `reg.expect_module("stem")`
   from `ops/src/hip/*` and `models/qwen3-moe/forward/*`.
3. **`BENCH_REFERENCE_KERNELS_GFX906`** — kept in-tree as A/B
   baselines; **not** invoked from forward path. Loaded by
   `crates/bench/*` and CLI PMC-refresh.
4. **`bar_p2p.rs` / direct `HipModule::load`** — collective kernels
   (`p2p_allreduce_residual*`) loaded at cluster init, not via
   `OpsRegistry`. Bench / test direct loads also land here.

A kernel is **orphan** if and only if it's in (4) without a forward-path
caller AND not in BENCH_REFERENCE.

## Section A — Per-kernel `.cu` file table

(146 production + 3 unverified, abbreviated to one row per file. `name`
omits the `flambeau_` prefix and `_q8_1` suffix when redundant. Every
file lives at `crates/kernels-hip/src/kernels/<file>`.)

| stem (file) | global fn(s) | loaded? (KERNEL_STEMS) | dispatch impl_id | cert | status |
|---|---|---|---|---|---|
| **MMVQ — Q4_0 family** ||||||
| `mmvq_q4_0` | `mmvq_q4_0_q8_1` | ✅ | `mmvq_q4_0_gfx906` (DirectCall) | `mmvq_q4_0_gfx906.json` | live |
| `mmvq_q4_0_t128` | `mmvq_q4_0_t128_q8_1` | ✅ | (Recipe-only intercept) | — | live (DP4A intercept) |
| `mmvq_q4_0_warpcoop64` | `mmvq_q4_0_warpcoop64_q8_1` | ✅ | (DP4A path) | `mmvq_q4_0_warpcoop64_gfx906.json` | live |
| `mmvq_q4_0_gate_up_dp4a` | `mmvq_q4_0_gate_up_dp4a_q8_1` | ✅ | (fused gate+up) | — | live |
| `mmvq_q4_0_gate_up_t128_dp4a` | `..._t128_dp4a_q8_1` | ✅ | (fused gate+up T128) | — | live |
| `mmvq_q4_0_gate_up_warpcoop64` | `..._warpcoop64_q8_1` | ✅ | (fused gate+up wc64) | — | live |
| `mmvq_q4_0_kv_f16dst_dp4a` | `..._kv_f16dst_dp4a_q8_1` | ✅ | (KV-fused decode) | — | live |
| **MMVQ — Q4_1 family** ||||||
| `mmvq_q4_1` | `mmvq_q4_1_q8_1` | ✅ | (Recipe intercept) | — | live |
| `mmvq_q4_1_t128` | `mmvq_q4_1_t128_q8_1` | ✅ | `qmatmul_q4_1_mmvq_t128_gfx906` | `qmatmul_q4_1_mmvq_t128_gfx906.json` | live (default Q4_1 mmvq) |
| `mmvq_q4_1_r2` | `mmvq_q4_1_r2_q8_1` | ✅ | (Recipe row, alt) | — | live |
| `mmvq_q4_1_r2_dp4a` | `..._r2_dp4a_q8_1` | ✅ | (Recipe row, alt) | — | live |
| `mmvq_q4_1_batched` | `mmvq_q4_1_q8_1_batched` | ✅ | (FLAMBEAU_BATCHED_MMVQ gate, opt-in) | — | live (gated) |
| `mmvq_q4_1_gate_up_dp4a` | `mmvq_q4_1_gate_up_dp4a_q8_1` | ✅ | (fused gate+up) | — | live |
| **MMVQ — Q4_K family (multi-row DPP)** ||||||
| `mmvq_q4_k` | `mmvq_q4_k_q8_1` | ✅ | (Recipe row) | — | live |
| `mmvq_q4_k_r2` | `mmvq_q4_k_r2_q8_1` | ✅ | `qmatmul_q4_K_mmvq_nw1_r2_gfx906` | `qmatmul_q4_K_mmvq_nw1_r2_gfx906.json` | live (default Q4_K mmvq) |
| `mmvq_q4_k_r4` | `mmvq_q4_k_r4_q8_1` | ✅ | (Recipe row, alt) | — | live |
| **MMVQ — Q5_0 / Q5_1 / Q5_K** ||||||
| `mmvq_q5_0` | `mmvq_q5_0_q8_1` | ✅ | `mmvq_q5_0_gfx906` (DirectCall) | `mmvq_q5_0_gfx906.json` | live |
| `mmvq_q5_1` | `mmvq_q5_1_q8_1` | ✅ | `mmvq_q5_1_gfx906` (DirectCall) | `mmvq_q5_1_gfx906.json` | live |
| `mmvq_q5_k` | `mmvq_q5_k_q8_1` | ✅ | (Recipe row) | — | live |
| `mmvq_q5_k_r2` | `mmvq_q5_k_r2_q8_1` | ✅ | `qmatmul_q5_K_mmvq_nw1_r2_gfx906` | `qmatmul_q5_K_mmvq_nw1_r2_gfx906.json` | live (default Q5_K mmvq) |
| `mmvq_q5_k_r2_f16dst` | `..._r2_f16dst_q8_1` | ✅ | (KV-fused) | — | live |
| **MMVQ — Q6_K family** ||||||
| `mmvq_q6_k` | `mmvq_q6_k_q8_1` | ✅ | (Recipe row) | — | live |
| `mmvq_q6_k_r4` | `mmvq_q6_k_r4_q8_1` | ✅ | `qmatmul_q6_K_mmvq_nw1_r4_gfx906` | `qmatmul_q6_K_mmvq_nw1_r4_gfx906.json` | bench-ref only |
| `mmvq_q6_k_dp4a` | `mmvq_q6_k_dp4a_q8_1` | ✅ | `qmatmul_q6_K_mmvq_dp4a_gfx906` | `qmatmul_q6_K_mmvq_dp4a_gfx906.json` | live (default Q6_K mmvq) |
| **MMVQ — Q8_0 family (8 variants — perf bake-off)** ||||||
| `mmvq_q8_0` | `mmvq_q8_0_q8_1` | ✅ | (Recipe baseline) | — | live |
| `mmvq_q8_0_dp4a` | `..._dp4a_q8_1` | ✅ | (Recipe row) | — | live |
| `mmvq_q8_0_dp4a_vdr2` | `..._dp4a_vdr2_q8_1` | ✅ | `qmatmul_q8_0_mmvq_single_row_gfx906` (intercept) | `qmatmul_q8_0_mmvq_single_row_gfx906.json` | live (default Q8_0 mmvq, +19% vs scalar) |
| `mmvq_q8_0_r4_dp4a` | `..._r4_dp4a_q8_1` | ✅ | (Recipe row, alt) | — | live |
| `mmvq_q8_0_t128` | `..._t128_q8_1` | ✅ | (Recipe row, alt) | `qmatmul_q8_0_mmvq_t128_gfx906.json` | live (FLAMBEAU_Q8_0_MMVQ_T128 gate) |
| `mmvq_q8_0_t128_vdr2` | `..._t128_vdr2_q8_1` | ✅ | (Recipe row, alt) | `qmatmul_q8_0_mmvq_t128_vdr2_gfx906.json` | live (FLAMBEAU_Q8_0_MMVQ_T128_VDR2) |
| `mmvq_q8_0_gate_up_dp4a` | `..._gate_up_dp4a_q8_1` | ✅ | (fused gate+up) | — | live |
| `mmvq_q8_0_gate_up_t128_vdr2` | `..._gate_up_t128_vdr2_q8_1` | ✅ | (FLAMBEAU_Q8_0_GU_T128_VDR2) | — | live (gated) |
| `mmvq_q8_0_llamacpp_style` | `..._llamacpp_style_q8_1` | ✅ | (Recipe row, alt) | — | live (A/B reference) |
| **MMVQ — F16 / BF16** ||||||
| `mmvq_f16_q8_1` | `mmvq_f16_q8_1` | ✅ | `mmvq_f16_q8_1_gfx906` (DirectCall) | `mmvq_f16_q8_1_gfx906.json` | live (UD-Q8_K_XL F16 layers) |
| `mmvq_bf16_bf16` | `mmvq_bf16_bf16` | ✅ | `mmvq_bf16_bf16_gfx906` (BF16 dense) | `mmvq_bf16_bf16_gfx906.json` | live |
| **MMQ — Q4_0 / Q4_1 prefill** ||||||
| `mmq_q4_0_4warp_lds` | `mmq_q4_0_4warp_lds_q8_1` | ✅ | `qmatmul_q4_0_mmq_4warp_lds_gfx906` | `qmatmul_q4_0_mmq_4warp_lds_gfx906.json` | live (default Q4_0 prefill m≥128) |
| `mmq_q4_0_wave64` | `mmq_q4_0_wave64_q8_1` | ✅ | `qmatmul_q4_0_mmq_wave64_gfx906` | `qmatmul_q4_0_mmq_wave64_gfx906.json` | bench-ref only (V2.29.e regress) |
| `mmq_q4_1_4warp_lds` | `mmq_q4_1_4warp_lds_q8_1` | ✅ | `qmatmul_q4_1_mmq_4warp_lds_gfx906` | `qmatmul_q4_1_mmq_4warp_lds_gfx906.json` | live (default Q4_1 prefill m≥128) |
| `mmq_q4_1_wave64` | `mmq_q4_1_wave64_q8_1` | ✅ | `qmatmul_q4_1_mmq_wave64_gfx906` (DORMANT m_range MAX,MAX) | `qmatmul_q4_1_mmq_wave64_gfx906.json` | dormant (lookup-only) |
| `mmq_q4_1_wave64_tile16` | `..._wave64_tile16_q8_1` | ✅ | (no dispatch row) | `qmatmul_q4_1_mmq_wave64_tile16_gfx906.json` | **orphan in dispatch** (cert exists) |
| **MMQ — Q4_K family (BENCH-ONLY)** ||||||
| `mmq_q4_K_4warp` | `mmq_q4_K_4warp_q8_1` | ❌ NOT in KERNEL_STEMS | `qmatmul_q4_K_mmq_4warp_lds_gfx906` (BENCH_REFERENCE) | `qmatmul_q4_K_mmq_4warp_lds_gfx906.json` | bench-only |
| `mmq_q4_K_turbo` | `mmq_q4_K_turbo_q8_1` | ❌ NOT in KERNEL_STEMS | `qmatmul_q4_K_mmq_turbo_gfx906` (DORMANT m_range MAX,MAX per impls.rs comment) | `qmatmul_q4_K_mmq_turbo_gfx906.json` | bench-only |
| `mmq_q4_K_wave64` | `mmq_q4_K_wave64_q8_1` | ❌ NOT in KERNEL_STEMS | `qmatmul_q4_K_mmq_wave64_gfx906` (default Q4_K prefill via dispatch row) | `qmatmul_q4_K_mmq_wave64_gfx906.json` | **CONTRADICTION**: dispatch references but not in OpsRegistry — confirm |
| **MMQ — Q5 / Q6_K (BENCH-ONLY for K-quants)** ||||||
| `mmq_q5_0_wave64` | `mmq_q5_0_wave64_q8_1` | ✅ | `qmatmul_q5_0_mmq_wave64_gfx906` | `qmatmul_q5_0_mmq_wave64_gfx906.json` | live |
| `mmq_q5_K_wave64` | `mmq_q5_K_wave64_q8_1` | ❌ NOT in KERNEL_STEMS | `qmatmul_q5_K_mmq_wave64_gfx906` (dispatch row) | `qmatmul_q5_K_mmq_wave64_gfx906.json` | **CONTRADICTION** |
| `mmq_q6_K_4warp` | `mmq_q6_K_4warp_q8_1` | ❌ | `qmatmul_q6_K_mmq_4warp_lds_gfx906` (BENCH_REFERENCE) | `qmatmul_q6_K_mmq_4warp_lds_gfx906.json` | bench-only |
| `mmq_q6_K_wave64` | `mmq_q6_K_wave64_q8_1` | ❌ NOT in KERNEL_STEMS | `qmatmul_q6_K_mmq_wave64_gfx906` (default Q6_K prefill) | `qmatmul_q6_K_mmq_wave64_gfx906.json` | **CONTRADICTION** |
| **MMQ — Q8_0 family** ||||||
| `mmq_q8_0_4warp` | `mmq_q8_0_4warp_q8_1` | ✅ | `qmatmul_q8_0_mmq_4warp_lds_gfx906` (BENCH_REFERENCE) | `qmatmul_q8_0_mmq_4warp_lds_gfx906.json` | bench-only |
| `mmq_q8_0_oracle` | `mmq_q8_0_oracle_q8_1` | ✅ | `qmatmul_q8_0_mmq_oracle_gfx906` | `qmatmul_q8_0_mmq_oracle_gfx906.json` | live (debug oracle) |
| `mmq_q8_0_wave64` | `mmq_q8_0_wave64_q8_1` | ✅ | `qmatmul_q8_0_mmq_wave64_gfx906` (BENCH_REFERENCE) | `qmatmul_q8_0_mmq_wave64_gfx906.json` | bench-only |
| `mmq_q8_0_wave64_tile16` | `mmq_q8_0_wave64_tile16_q8_1` | ✅ | `qmatmul_q8_0_mmq_wave64_tile16_gfx906` | `qmatmul_q8_0_mmq_wave64_tile16_gfx906.json` | live (default Q8_0 prefill m≥128) |
| `mmq_q8_0_wave64_tile32` | `mmq_q8_0_wave64_tile32_q8_1` | ✅ | (no dispatch row) | — | **orphan** (no cert, no dispatch) |
| `mmq_f16_q8_1` | `mmq_f16_q8_1` | ✅ | `mmq_f16_q8_1_gfx906` (DirectCall) | `mmq_f16_q8_1_gfx906.json` | live |
| `mmq_f16_tile` | `mmq_f16_tile` | ✅ | `mmq_f16_tile_gfx906` (Recipe) | `mmq_f16_tile_gfx906.json` | live |
| **Indexed-MoE MMVQ (Q4_0, Q4_1, Q4_K family — 12 variants — perf bake-off)** ||||||
| `indexed_moe_mmvq_q4_0` | `..._q4_0_q8_1` | ✅ | `indexed_moe_mmvq_q4_0_gfx906` (DirectCall) | `indexed_moe_mmvq_q4_0_gfx906.json` | live |
| `indexed_moe_mmvq_q4_0_gate_up_dp4a` | `..._q4_0_gate_up_dp4a_q8_1` | ✅ | (Recipe-only / opt-in via env) | — | live |
| `indexed_moe_mmvq_q4_1` | `..._q4_1_q8_1` | ✅ | (DirectCall? — see below) | `indexed_moe_mmvq_q4_1_gfx906.json` | **orphan in impls.rs** |
| `indexed_moe_mmvq_q4_k` | `..._q4_k_q8_1` | ✅ | (BENCH_REFERENCE) | `indexed_moe_mmvq_q4_k_gfx906.json` | bench-only |
| `indexed_moe_mmvq_q4_k_r2` | `..._q4_k_r2_q8_1` | ✅ | `indexed_moe_mmvq_q4_k_r2_gfx906` (Recipe) | `indexed_moe_mmvq_q4_k_r2_gfx906.json` | live (default MoE Q4_K mmvq) |
| `indexed_moe_mmvq_q4_k_r2_dp4a` | `..._q4_k_r2_dp4a_q8_1` | ✅ | (alt; opt-in via FLAMBEAU_MOE_VARIANT) | — | live (gated) |
| `indexed_moe_mmvq_q4_k_r4_dp4a` | `..._q4_k_r4_dp4a_q8_1` | ✅ | (alt; opt-in) | — | live (gated) |
| `indexed_moe_mmvq_q4_k_r4_sorted_dp4a` | `..._r4_sorted_dp4a_q8_1` | ✅ | (alt; opt-in via FLAMBEAU_MOE_SORTED) | — | live (gated) |
| `indexed_moe_mmvq_q4_k_gate_up` | `..._gate_up_q8_1` | ✅ | `indexed_moe_mmvq_q4_k_gate_up_gfx906` (Recipe) | `indexed_moe_mmvq_q4_k_gate_up_gfx906.json` | live (default fused gate+up Q4_K) |
| `indexed_moe_mmvq_q4_k_gate_up_dp4a` | `..._gate_up_dp4a_q8_1` | ✅ | (alt; opt-in) | — | live (gated) |
| `indexed_moe_mmvq_q4_k_gate_up_mbatch` | `..._gate_up_mbatch_q8_1` | ✅ | (FLAMBEAU_MBATCH gate, opt-in) | — | live (gated) |
| `indexed_moe_mmvq_q4_k_gate_up_r2_dp4a` | `..._r2_dp4a_q8_1` | ✅ | (alt; opt-in) | — | live (gated) |
| `indexed_moe_mmvq_q4_k_gate_up_r4_dp4a` | `..._r4_dp4a_q8_1` | ✅ | (alt; opt-in) | — | live (gated) |
| `indexed_moe_mmvq_q4_k_gate_up_r4_sorted_dp4a` | `..._r4_sorted_dp4a_q8_1` | ✅ | (alt; opt-in via FLAMBEAU_MOE_SORTED) | — | live (gated) |
| `indexed_moe_mmvq_q4_k_gate_up_r8_dp4a` | `..._r8_dp4a_q8_1` | ✅ | (alt; opt-in) | — | live (gated) |
| `indexed_moe_mmvq_q5_k` | `..._q5_k_q8_1` | ✅ | `indexed_moe_mmvq_q5_k_gfx906` (DirectCall) | `indexed_moe_mmvq_q5_k_gfx906.json` | live |
| `indexed_moe_mmvq_q6_k` | `..._q6_k_q8_1` | ✅ | `indexed_moe_mmvq_q6_k_gfx906` (DirectCall) | `indexed_moe_mmvq_q6_k_gfx906.json` | live |
| `indexed_moe_mmvq_q8_0` | `..._q8_0_dp4a_q8_1` | ✅ | `indexed_moe_mmvq_q8_0_gfx906` (DirectCall) | `indexed_moe_mmvq_q8_0_gfx906.json` | live |
| **Indexed-MoE MMQ (prefill)** ||||||
| `indexed_moe_mmq_q4_k` | `..._q4_k_q8_1` | ✅ | `indexed_moe_mmq_q4_k_gfx906` (Recipe) | `indexed_moe_mmq_q4_k_gfx906.json` | live |
| `indexed_moe_mmq_q4_0_down_tile8_dp4a` | `..._q4_0_down_tile8_dp4a_q8_1` | ✅ | (Recipe path) | — | live |
| `indexed_moe_mmq_q4_0_gate_up_tile8_dp4a` | `..._q4_0_gate_up_tile8_dp4a_q8_1` | ✅ | (Recipe path) | — | live |
| `indexed_moe_mmq_q4_1_down_tile8_dp4a` | `..._q4_1_down_tile8_dp4a_q8_1` | ✅ | (Recipe path) | — | live |
| `indexed_moe_mmq_q4_k_down_tile8_dp4a` | `..._q4_k_down_tile8_dp4a_q8_1` | ✅ | (Recipe) | — | live |
| `indexed_moe_mmq_q4_k_down_turbo` | `..._q4_k_down_turbo_q8_1` | ✅ | (alt turbo path) | — | live (gated) |
| `indexed_moe_mmq_q4_k_gate_up_tile8_dp4a` | `..._q4_k_gate_up_tile8_dp4a_q8_1` | ✅ | (Recipe — default) | — | live |
| `indexed_moe_mmq_q4_k_gate_up_tile16_dp4a` | (in `_unverified/`) | ✅ in KERNEL_STEMS but file is `_unverified/` | (no dispatch) | — | **anomaly: stem listed but file in _unverified/** |
| `indexed_moe_mmq_q4_k_gate_up_turbo` | `..._gate_up_turbo_q8_1` | ✅ | (alt turbo path) | — | live (gated) |
| `indexed_moe_mmq_q5_k_down_tile8_dp4a` | `..._q5_k_down_tile8_dp4a_q8_1` | ✅ | (Recipe) | — | live |
| `indexed_moe_mmq_q6_k_down_tile8_dp4a` | `..._q6_k_down_tile8_dp4a_q8_1` | ✅ | (Recipe) | — | live |
| `indexed_moe_mmq_q8_0_down_tile8_dp4a` | `..._q8_0_down_tile8_dp4a_q8_1` | ✅ | (Recipe) | `indexed_moe_mmq_q8_0_down_tile8_gfx906.json` | live |
| `indexed_moe_mmq_q8_0_gate_up_tile8_dp4a` | `..._q8_0_gate_up_tile8_dp4a_q8_1` | ✅ | (Recipe) | `indexed_moe_mmq_q8_0_gate_up_tile8_gfx906.json` | live |
| **Attention** ||||||
| `attention_decode_f16` | `attention_decode_f16` | ✅ | `attention_decode_f16_gfx906` | `attention_decode_f16_gfx906.json` | live (default decode) |
| `attention_decode_f16_batched` | `attention_decode_f16_batched` | ✅ | (batched-decode path; #266) | — | live (P2.9b-i2-E) |
| `attention_decode_f16_splitk` | `attention_decode_f16_splitk_chunk`, `_combine` | ✅ | `attention_decode_f16_splitk_gfx906` (DirectCall) | `attention_decode_f16_splitk_gfx906.json` | live (long-context, n_kv > 256) |
| `attention_decode_q8_kv` | `attention_decode_q8_kv` | ✅ | `attention_decode_q8_kv_gfx906` (DirectCall) | `attention_decode_q8_kv_gfx906.json` | live (Q8 KV cache) |
| `attention_decode_bf16` | `attention_decode_bf16` | ✅ | `attention_decode_bf16_gfx906` | `attention_decode_bf16_gfx906.json` | live |
| `attention_prefill_f16` | `attention_prefill_f16` | ✅ | `attention_prefill_f16_gfx906` | `attention_prefill_f16_gfx906.json` | live |
| `attention_prefill_flash_tile_f16` | `attention_prefill_flash_tile_f16` | ✅ | (BENCH_REFERENCE_KERNELS_GFX906) | `attention_prefill_flash_tile_f16_gfx906.json` | bench-only |
| **Norms / pointwise** ||||||
| `rmsnorm_f16` | `rmsnorm_f16` | ✅ | `rmsnorm_f16_gfx906` | `rmsnorm_f16_gfx906.json` | live |
| `rmsnorm_bf16` | `rmsnorm_bf16` | ✅ | `rmsnorm_bf16_gfx906` | `rmsnorm_bf16_gfx906.json` | live |
| `rmsnorm_f32` | `rmsnorm_f32` | ✅ | `rmsnorm_f32_gfx906` | `rmsnorm_f32_gfx906.json` | live |
| `rmsnorm_q8_1_fused` | `rmsnorm_q8_1_fused` | ✅ | `rmsnorm_q8_1_fused_gfx906` | `rmsnorm_q8_1_fused_gfx906.json` | live (D1 fused) |
| `rmsnorm_f16_add_residual` | `rmsnorm_f16_add_residual` | ✅ | (direct call) | — | live |
| `l2_norm_f32` | `l2_norm_f32` | ✅ | `l2_norm_f32_gfx906` | `l2_norm_f32_gfx906.json` | live |
| `silu_f32`, `swiglu_f16`, `swiglu_f32`, `swiglu_f32_to_{f16,bf16,q8_1}`, `sigmoid_mul_{f16,bf16}`, `scale_f32`, `add_{f16,f32}`, `split_q_gate_{f16,bf16}`, `shared_expert_scale_f32` | (1 fn each) | ✅ | various direct rows | mostly certed | live |
| **RoPE** ||||||
| `rope_f16` | `rope_f16` | ✅ | `rope_f16_gfx906` | `rope_f16_gfx906.json` | live |
| `rope_neox_partial_f16` / `_bf16` | (1 fn each) | ✅ | `rope_neox_partial_{f16,bf16}_gfx906` | each has cert | live |
| **Cast / quantize / topk / softmax** ||||||
| `cast_*` (6 dtype pairs) | `cast_*` (1 fn each) | ✅ | each has dispatch row + cert | — | live |
| `quantize_f16_q8_0` | `quantize_row_f16_q8_0` | ✅ | (direct call) | — | live |
| `quantize_f16_q8_1` | `quantize_row_f16_q8_1` | ✅ | `quantize_f16_q8_1_gfx906` | `quantize_f16_q8_1_gfx906.json` | live |
| `quantize_f16_q8_1_mmq` | `quantize_f16_q8_1_mmq` | ✅ | (direct call, MMQ activation) | — | live |
| `quantize_q8_1` | `quantize_row_q8_1` | ✅ | (direct call) | — | live |
| `quantize_q8_1_mmq` | `quantize_q8_1_mmq` | ✅ | `quantize_q8_1_mmq_gfx906` (BENCH_REFERENCE) | `quantize_q8_1_mmq_gfx906.json` | bench-only |
| `topk_f32` | `topk_softmax_f32` | ✅ | `topk_f32_gfx906` | `topk_f32_gfx906.json` | live (router top-K) |
| `softmax_masked_f16` | `softmax_masked_f16` | ✅ | `softmax_masked_f16_gfx906` | `softmax_masked_f16_gfx906.json` | live (F5) |
| **GDN (recurrent / hybrid)** ||||||
| `gdn_alpha_beta_f32` | `gdn_alpha_beta_f32` | ✅ | `gdn_alpha_beta_f32_gfx906` | `gdn_alpha_beta_f32_gfx906.json` | live |
| `gdn_state_step_f32` | (`__launch_bounds__`-tagged kernel; no extern "C" name in stem grep due to header-line wrap) | ✅ | `gdn_state_step_f32_s128_gfx906` | `gdn_state_step_f32_s128_gfx906.json` | live |
| `gdn_state_step_alphabeta_f32` | (same shape) | ✅ | `gdn_state_step_alphabeta_f32_s128_gfx906` | `gdn_state_step_alphabeta_f32_s128_gfx906.json` | live |
| `gdn_split_qkv_f32` | `gdn_split_qkv_f32` | ✅ | (direct call from forward/gdn.rs) | — | live |
| `gdn_assemble_conv_input_f32` | `gdn_assemble_conv_input_f32` | ✅ | (direct call) | — | live |
| `causal_conv1d_f32` | `causal_conv1d_f32` | ✅ | `causal_conv1d_f32_gfx906` | `causal_conv1d_f32_gfx906.json` | live |
| **Sampler (D-track)** ||||||
| `sampler_topk_softmax_f32` | `sampler_topk_softmax_f32` | ✅ | (direct call from gpu_sampler.rs) | — | live (FLAMBEAU_GPU_SAMPLER) |
| `sampler_apply_penalties_f32` | `sampler_apply_penalties_f32` | ✅ | (direct call) | — | live (D4) |
| **MoE combine / sort** ||||||
| `moe_combine_f16` | `moe_combine_f16` | ✅ | `moe_combine_f16_gfx906` | `moe_combine_f16_gfx906.json` | live |
| `moe_combine_no_residual_f16` | (1 fn) | ✅ | (direct call) | — | live |
| `moe_combine_two_residuals_f16` | (1 fn) | ✅ | (direct call) | — | live |
| `moe_sort_by_expert` | 7 funcs (`zero_counts`, `count`, `scan_offsets`, `scatter`, `scatter_det`, `scan_padded_offsets[_16]`, `pad_copy`) | ✅ | (direct call from forward/moe.rs) | — | live |
| **Dense GEMV / batched** ||||||
| `dense_gemv_f32_f16` | `dense_gemv_f32_f16` | ✅ | `dense_gemv_f32_f16_gfx906` | `dense_gemv_f32_f16_gfx906.json` | live |
| `dense_gemv_f32_f16_batched` | (1 fn) | ✅ | (direct call) | — | live |
| `dense_gemv_f16_f16` | (1 fn) | ✅ | (direct call) | — | live |
| `dense_gemv_f16_f16_batched` | (1 fn) | ✅ | (direct call) | — | live |
| **P2P collectives (separate loader path)** ||||||
| `p2p_allreduce_residual` | `flambeau_p2p_allreduce_{residual,sum}_{tp2,tp4}` (4 fns) | ❌ NOT in KERNEL_STEMS — loaded via `bar_p2p.rs::HipModule::load` | (cert: `p2p_allreduce_residual_tp4.json`) | partial cert | live (collective path) |
| `p2p_allreduce_residual_rmsnorm` | `..._rmsnorm_{tp2,tp4}` (2 fns) | ❌ NOT in KERNEL_STEMS — `bar_p2p.rs` | — | uncertified | live |
| `p2p_allreduce_residual_rmsnorm_q8_1` | `..._rmsnorm_q8_1_{tp2,tp4}` (2 fns) | ❌ NOT in KERNEL_STEMS — `bar_p2p.rs` | — | uncertified | live (FLAMBEAU_AR_FUSE_Q8_1) |
| **Unverified (cfg-equivalent)** ||||||
| `_unverified/indexed_moe_mmq_q4_k_gate_up_tile8_ylds` | (—) | not loaded | (—) | (—) | unverified — V2.24.b NULL |
| `_unverified/indexed_moe_mmq_q4_k_gate_up_tile16_dp4a` | (—) | listed in KERNEL_STEMS (anomaly — see notes) | (—) | (—) | unverified — V2.31.b NULL |
| `_unverified/mmvq_q4_0_r2` | (—) | not loaded | (—) | (—) | unverified — V2.28.d NULL |

> **Notes:**
> - `indexed_moe_mmq_q4_k_gate_up_tile16_dp4a` appears in both
>   `KERNEL_STEMS` (as a loaded module) AND `_unverified/`. **Check
>   if `build.rs` skips `_unverified/` paths or compiles them anyway**.
> - "CONTRADICTION" rows: dispatch table references the impl but the
>   stem is not in `KERNEL_STEMS`. Either dispatch never picks them at
>   runtime (for K-quant MMQ wave64 — the certs imply they were measured
>   via `bench/sweep_mmq.rs` which does its own `HipModule::load`), or
>   there's a missing entry in `KERNEL_STEMS`. **Action**: in Phase 1,
>   verify whether dispatch_qmatmul actually returns these `KernelDescriptor`s
>   for any (model, dtype, m) the anchor models exercise.

## Section B — Per-op Rust launcher table

(`crates/ops/src/hip/*.rs` — each `pub fn` op uses one or more kernel
stems via `reg.expect_module(...)`.)

| op file | exposed fns | kernel stems used |
|---|---|---|
| `attention.rs` | `attention_decode_*`, `attention_prefill_*`, `attention_decode_batched`, `attention_decode_q8_kv`, `attention_decode_splitk` | `attention_decode_{f16,bf16,batched,splitk,q8_kv}`, `attention_prefill_f16` |
| `cast.rs` | 6 cast fns | `cast_*` (6 stems) |
| `conv.rs` | `causal_conv1d_f32` | `causal_conv1d_f32` |
| `mlp.rs` | `silu`, `swiglu_*`, `scale_f32`, `add_{f16,f32}`, `swiglu_f16`, `sigmoid_mul_{f16,bf16}` | matching stems |
| `moe.rs` | indexed-MoE MMVQ + MMQ + sort + combine launchers | `indexed_moe_mm{vq,q}_*`, `moe_combine_*`, `moe_sort_by_expert`, `topk_f32` |
| `norm.rs` | rmsnorm fns + quantize fns + l2_norm | `rmsnorm_*`, `quantize_*`, `l2_norm_f32` |
| `pe.rs` | rope fns | `rope_f16`, `rope_neox_partial_{f16,bf16}` |
| `qmatmul.rs` | qmatmul (Recipe-driven) + dense f16/bf16 launchers | every MMVQ + MMQ stem (Recipe), `mmvq_{f16_q8_1,bf16_bf16}`, `mmq_{f16_q8_1,f16_tile}`, `mmvq_q4_1_batched`, `mmvq_q5_k_r2_f16dst`, `mmvq_q4_0_kv_f16dst_dp4a` |
| `recurrent.rs` | GDN fns | `gdn_{alpha_beta,state_step,state_step_alphabeta,split_qkv,assemble_conv_input}_f32` |
| `router.rs` | dense GEMV (router) | `dense_gemv_{f32_f16,f16_f16}{,_batched}` |
| `sampling.rs` | GPU sampler | `sampler_topk_softmax_f32`, `sampler_apply_penalties_f32` |
| `softmax.rs` | masked softmax | `softmax_masked_f16` |
| `mod.rs` | `OpsRegistry` + `KERNEL_STEMS` (137 entries) | (all of the above) |

## Section C — `impls.rs` registration

- **`QMATMUL_GFX906`** — 18 KernelDescriptor rows; first-match by `(dtype_w, dtype_a, m)`.
- **`RMSNORM_GFX906`** — 3 rows.
- **`SWIGLU_GFX906`** — 1 row.
- **`ROPE_GFX906`** — 1 row.
- **`SOFTMAX_GFX906`** — 1 row.
- **`ATTENTION_DECODE_GFX906`** — N rows.
- **`INDEXED_MOE_MMQ_GFX906`** — 1 row (Q4_K).
- **`DIRECT_CALL_KERNELS_GFX906`** — 11 entries (no `m_range` choice).
- **`BENCH_REFERENCE_KERNELS_GFX906`** — 11 entries (A/B baselines, not
  forward-path).
- **Dormant** (m_range MAX,MAX, lookup-only): `qmatmul_q4_1_mmq_wave64_gfx906`.
- The header comment also names `qmatmul_q4_K_mmq_turbo_gfx906` as a
  V2.14.b dormant row pending A/B promotion — but the live KernelDescriptor
  for that impl_id has m_range `(128, usize::MAX)`. **Resolve**: is the
  comment stale, or is there a duplicate row?

## Summary lists — Phase 2 slice S2 deletion candidates

### Orphan kernels (in repo, not in any dispatch row, not BENCH_REFERENCE, not collective)

These three are the cleanest deletion candidates:

- **`mmq_q8_0_wave64_tile32`** — in `KERNEL_STEMS` (loaded), no
  dispatch row, no cert. Last referenced as `Dtype::Q8_0Wave64Tile32`
  in `sweep_mmq.rs` Recipe; never selected by anchors. **Delete with
  Recipe row**.
- **`mmq_q4_1_wave64_tile16`** — in `KERNEL_STEMS`, has cert
  (`qmatmul_q4_1_mmq_wave64_tile16_gfx906.json`), but no dispatch
  row. A/B reference for sweep_mmq. **Move to BENCH_REFERENCE** or
  **delete + cert + Recipe row**.
- **`indexed_moe_mmvq_q4_1`** — in KERNEL_STEMS, cert exists, no
  KernelDescriptor / DirectCall registration in `impls.rs`. Likely
  unused (Q4_1 MoE is rare). **Confirm at Phase 1** (rg launchers).

### Bench-only kernels (kept for sweep_mmq / PMC-refresh A/B)

These are intentional per `BENCH_REFERENCE_KERNELS_GFX906`. **Keep** as
A/B references unless the user wants to drop the bench infrastructure
too:

- `mmq_q4_K_4warp`, `mmq_q4_K_turbo`, `mmq_q6_K_4warp` — all bench-only
  (not in KERNEL_STEMS, only sweep_mmq loads them).
- `mmq_q8_0_4warp`, `mmq_q8_0_wave64`, `quantize_q8_1_mmq`,
  `attention_prefill_flash_tile_f16` — in KERNEL_STEMS + bench-only.
- `qmatmul_q{4,5,6}_K_mmvq_single_row_gfx906`, `qmatmul_q6_K_mmvq_nw1_r4_gfx906`
  — bench-ref impl_ids without dispatch rows.
- `indexed_moe_mmvq_q4_k_gfx906` — bench-ref single-row baseline.
- `peer_copy_via_host_gfx906` — bandwidth probe only.

### Uncertified kernels (in dispatch but cert missing)

`comm -23 dispatch_impls certs` returns **0** rows — every dispatched
impl has a cert. Phase 1 will reverse-check via `cert-check` tool.

### `cfg(unverified)` (kept-but-disabled)

3 files under `_unverified/`:
- `indexed_moe_mmq_q4_k_gate_up_tile8_ylds.cu` (V2.24.b NULL).
- `indexed_moe_mmq_q4_k_gate_up_tile16_dp4a.cu` (V2.31.b NULL).
  **Anomaly**: stem is also listed in `KERNEL_STEMS`. Confirm whether
  build excludes `_unverified/` paths, or this is a bug.
- `mmvq_q4_0_r2.cu` (V2.28.d NULL).

Per CLAUDE.md rule #10, these are kept for A/B reference. **Can be
deleted if Phase 1 confirms no caller exists** (most likely yes; they
have explanatory comments saying "kept for reference").

### Anchor-model coverage (preview for Phase 1)

The four anchor models all hit:
- **Qwen3.5-9B-Q4_1** dense — `mmq_q4_1_4warp_lds`, `mmvq_q4_1_t128`, dense FFN swiglu.
- **Qwen3.6-27B-Q4_0** + **Qwen3.6-35B-A3B-Q4_0** — `indexed_moe_mmvq_q4_0`, `mmvq_q4_0` shared expert, GDN, MoE-MMVQ Q4_0, hybrid path.
- **Qwen3-Coder-Next-80B-Q4_0** — same hybrid forward as qwen36moe (qwen3next arch shares `crates/models/qwen3-moe/forward/`).

Anchors do **not** exercise:
- Q4_K / Q5_K / Q6_K MMQ paths (only used for K-quant prefill — anchors are Q4_0/Q4_1).
- `mmq_f16_tile`, `mmvq_bf16_bf16` (BF16 dense — Gemma-4 / Mistral non-anchor).
- `attention_decode_bf16` / `rmsnorm_bf16` / `rope_neox_partial_bf16` /
  `swiglu_f32_to_bf16` / `sigmoid_mul_bf16` / `split_q_gate_bf16` (BF16
  weights — non-anchor).

So the **non-anchor-coverage suspects** for Phase 1 triage:
- All BF16 kernels (~6).
- All K-quant MMQ kernels (~6, mostly bench-only already).
- `mmvq_q4_k`, `mmvq_q5_k`, `mmvq_q6_k` single-row (Recipe alts) — Q4_0/Q4_1 anchors don't hit Q*_K decode.
- `mmvq_q5_0`, `mmvq_q5_1` direct-call — anchors are Q4_0/Q4_1, so these are non-anchor.
- `attention_prefill_flash_tile_f16` (already bench-only).
