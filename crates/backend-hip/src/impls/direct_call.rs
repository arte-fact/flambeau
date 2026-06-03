//! `DirectCallKernel` catalogs for `gfx906`. Split out of `impls.rs` so its
//! ~250 LOC compiles as a separate codegen unit.

use flambeau_core::DirectCallKernel;

/// Catalog of kernels that are invoked directly from call sites (via
/// `reg.expect_module("stem")`) rather than through shape-based dispatch.
/// Each entry must have a matching row in `dispatch/hip/gfx906.toml` and a
/// cert on disk. The `dispatch_toml_roundtrip` test (below) enforces both.
/// Membership rule: a kernel belongs here when there is exactly one
/// implementation per `(op, dtype)` — no `m_range` choice to make. Adding
/// a second implementation for the same dtype means promoting the pair into
/// a `KernelDescriptor` table so shape-dispatch can pick between them.
pub const DIRECT_CALL_KERNELS_GFX906: &[DirectCallKernel] = &[
    // 9.b flash-decoding split-K attention for long-context decode.
    // Forward layer crosses the threshold at `n_tokens_kv > 256` and
    // invokes this kernel directly rather than `attention_decode_f16`.
    DirectCallKernel {
        impl_id: "attention_decode_f16_splitk_gfx906",
        cert_rel_path: "certs/hip/gfx906/attention_decode_f16_splitk_gfx906.json",
    },
    // 3.a — dense Q4_0 / Q5_0 MMVQ (DP4A) for Qwen3.6-35B-A3B-Q4_0
    // attention and shared-expert weights. Single-kernel per dtype.
    DirectCallKernel {
        impl_id: "mmvq_q4_0_gfx906",
        cert_rel_path: "certs/hip/gfx906/mmvq_q4_0_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "mmvq_q5_0_gfx906",
        cert_rel_path: "certs/hip/gfx906/mmvq_q5_0_gfx906.json",
    },
    // 6.a — Q5_1 dense MMVQ (llama.cpp parity).
    DirectCallKernel {
        impl_id: "mmvq_q5_1_gfx906",
        cert_rel_path: "certs/hip/gfx906/mmvq_q5_1_gfx906.json",
    },
    // 1.b / 5.a — F16 × Q8_1 MMVQ + MMQ (Unsloth UD-Q8_K_XL F16 layers).
    DirectCallKernel {
        impl_id: "mmvq_f16_q8_1_gfx906",
        cert_rel_path: "certs/hip/gfx906/mmvq_f16_q8_1_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "mmq_f16_q8_1_gfx906",
        cert_rel_path: "certs/hip/gfx906/mmq_f16_q8_1_gfx906.json",
    },
    // V2.29.a — tile-M F16 × Q8_1 MMQ for m >= 8. Call-site dispatched
    // alongside mmq_f16_q8_1 inside qmatmul.rs::dispatch_qmatmul.
    DirectCallKernel {
        impl_id: "mmq_f16_tile_gfx906",
        cert_rel_path: "certs/hip/gfx906/mmq_f16_tile_gfx906.json",
    },
    // 2.a / 3.a / indexed-MoE MMVQ for Q8_0 / Q4_0 / Q6_K
    // expert weights. Routing is by GGUF tensor dtype, not shape.
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q8_0_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q8_0_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_0_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q4_0_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q5_0_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q5_0_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q5_1_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q5_1_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q2_k_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q2_k_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q3_k_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q3_k_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q6_k_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q6_k_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q5_k_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q5_k_gfx906.json",
    },
    // Indexed-MoE MMQ tile8 (gate_up + down) per dtype. Routed by GGUF
    // tensor dtype, not via dispatch_qmatmul; cert harness in
    // crates/bench/src/sweep_moe.rs (T2.3b — generic tile8 harness).
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q8_0_gate_up_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q8_0_gate_up_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q8_0_down_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q8_0_down_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_0_gate_up_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_0_gate_up_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_0_down_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_0_down_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_1_gate_up_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_1_gate_up_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_1_down_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_1_down_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q5_0_gate_up_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q5_0_gate_up_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q5_0_down_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q5_0_down_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q5_1_gate_up_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q5_1_gate_up_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q5_1_down_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q5_1_down_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_k_gate_up_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_k_gate_up_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_k_down_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_k_down_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q5_k_gate_up_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q5_k_gate_up_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q5_k_down_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q5_k_down_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q6_k_gate_up_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q6_k_gate_up_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q6_k_down_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q6_k_down_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q2_k_gate_up_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q2_k_gate_up_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q2_k_down_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q2_k_down_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q3_k_gate_up_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q3_k_gate_up_tile8_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q3_k_down_tile8_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q3_k_down_tile8_gfx906.json",
    },
    // C10 — fused alpha-beta + state-step for Gated-Delta-Net (Qwen3.6).
    // Single impl for the alphabeta op; routed via `expect_module` at the
    // call site in `ops::hip::recurrent::gdn_state_step_alphabeta_f32_s128`.
    DirectCallKernel {
        impl_id: "gdn_state_step_alphabeta_f32_s128_gfx906",
        cert_rel_path: "certs/hip/gfx906/gdn_state_step_alphabeta_f32_s128_gfx906.json",
    },
];

/// Catalog of kernels that are **not** dispatched at runtime but are kept
/// in-tree as A/B baselines for correctness / perf comparison. Each has a
/// cert under `certs/hip/gfx906/` and a call site in `crates/bench/src/`
/// or `crates/cli/src/main.rs` (PMC-refresh target list).
/// Unlike [`DIRECT_CALL_KERNELS_GFX906`], these are not invoked by any
/// forward-path code — a bench sweep (or PMC refresh) is the only caller.
/// They stay registered here so:
/// 1. The `dispatch_toml_roundtrip` test does not need to special-case
///    TOML rows for bench baselines.
/// 2. Future simplifier passes have a single source of truth for
///    "this kernel is not dead — it's a reference baseline" and don't
///    propose deletion. (Sessions 1 and 2 of the simplification pass both
///    initially flagged these as orphans; this catalog closes that loop.)
///    Matches the "Single-row reference MMVQ kernels for the K-quants are
///    kept in-tree for cert cross-checks" note in `dispatch/hip/gfx906.toml`.
pub const BENCH_REFERENCE_KERNELS_GFX906: &[DirectCallKernel] = &[
    // long-context attention baseline — pre-split-K reference.
    DirectCallKernel {
        impl_id: "attention_prefill_flash_tile_f16_gfx906",
        cert_rel_path: "certs/hip/gfx906/attention_prefill_flash_tile_f16_gfx906.json",
    },
    // single-row indexed-MoE MMVQ baseline. Production ships r2 (and Q6_K
    // dp4a); this stays as the A/B reference it was promoted from.
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_k_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q4_k_gfx906.json",
    },
    // PP hand-off bandwidth cert. Exercised by `bench sweep
    // peer_copy_via_host` on the rig; not a kernel-launch dispatch.
    DirectCallKernel {
        impl_id: "peer_copy_via_host_gfx906",
        cert_rel_path: "certs/hip/gfx906/peer_copy_via_host_gfx906.json",
    },
    // 4-warp LDS-tiled MMQ placeholders. Superseded by wave64 in production
    // (Q4_K wave64, Q6_K wave64, Q8_0 wave64_tile16). Kept as bench A/B
    // baselines — `sweep_mmq` runs them as Q{4,6}K4Warp / Q8_04Warp variants
    // alongside the shipped kernels so the comparison stays live. See CLI
    // PMC-refresh target list at `crates/cli/src/main.rs`.
    DirectCallKernel {
        impl_id: "qmatmul_q4_K_mmq_4warp_lds_gfx906",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_K_mmq_4warp_lds_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q6_K_mmq_4warp_lds_gfx906",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q6_K_mmq_4warp_lds_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmq_4warp_lds_gfx906",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmq_4warp_lds_gfx906.json",
    },
    // superseded Q8_0 MMQ wave64 by wave64_tile16 at m >= 128. Wave64 stays
    // as the bench baseline (PMC-refresh target).
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmq_wave64_gfx906",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmq_wave64_gfx906.json",
    },
    // single-row MMVQ references. Production dispatches r2/r4/dp4a at
    // m < 128 for the K-quants; the single_row kernels stay as cross-check
    // baselines (called out in the TOML header).
    DirectCallKernel {
        impl_id: "qmatmul_q4_K_mmvq_single_row_gfx906",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_K_mmvq_single_row_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q2_K_mmvq_single_row_gfx906",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q2_K_mmvq_single_row_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q3_K_mmvq_single_row_gfx906",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q3_K_mmvq_single_row_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q5_K_mmvq_single_row_gfx906",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q5_K_mmvq_single_row_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q6_K_mmvq_single_row_gfx906",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q6_K_mmvq_single_row_gfx906.json",
    },
    // Q6_K multi-row r4. Superseded by dp4a at runtime; r4 remains
    // a cross-check baseline invoked from the PMC-refresh target list.
    DirectCallKernel {
        impl_id: "qmatmul_q6_K_mmvq_nw1_r4_gfx906",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q6_K_mmvq_nw1_r4_gfx906.json",
    },
    // turbo Q8_1 quantise — bench-only sanity (the forward path uses
    // `quantize_f16_q8_1` via `DIRECT_CALL_KERNELS_GFX906`, which is a
    // distinct kernel).
    DirectCallKernel {
        impl_id: "quantize_q8_1_mmq_gfx906",
        cert_rel_path: "certs/hip/gfx906/quantize_q8_1_mmq_gfx906.json",
    },
];
