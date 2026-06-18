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
    // Flash-decoding split-K attention for long-context decode.
    // Forward layer crosses the threshold at `n_tokens_kv > 256` and
    // invokes this kernel directly rather than `attention_decode_f16`.
    DirectCallKernel {
        impl_id: "attention_decode_f16_splitk_gfx906",
        cert_rel_path: "certs/hip/gfx906/attention_decode_f16_splitk_gfx906.json",
    },
    // Q8_0-KV sibling of the split-K decode (ring-SWA + long-ctx full-attn);
    // forward invokes it directly above n_tokens_kv > 256 with --kv q8.
    DirectCallKernel {
        impl_id: "attention_decode_q8_kv_splitk_gfx906",
        cert_rel_path: "certs/hip/gfx906/attention_decode_q8_kv_splitk_gfx906.json",
    },
    // Dense Q4_0 / Q5_0 MMVQ (DP4A) for Qwen3.6-35B-A3B-Q4_0
    // attention and shared-expert weights. Single-kernel per dtype.
    DirectCallKernel {
        impl_id: "mmvq_q4_0_gfx906",
        cert_rel_path: "certs/hip/gfx906/mmvq_q4_0_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "mmvq_q5_0_gfx906",
        cert_rel_path: "certs/hip/gfx906/mmvq_q5_0_gfx906.json",
    },
    // Q5_1 dense MMVQ (llama.cpp parity).
    DirectCallKernel {
        impl_id: "mmvq_q5_1_gfx906",
        cert_rel_path: "certs/hip/gfx906/mmvq_q5_1_gfx906.json",
    },
    // F16 × Q8_1 MMVQ + MMQ (Unsloth UD-Q8_K_XL F16 layers).
    DirectCallKernel {
        impl_id: "mmvq_f16_q8_1_gfx906",
        cert_rel_path: "certs/hip/gfx906/mmvq_f16_q8_1_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "mmq_f16_q8_1_gfx906",
        cert_rel_path: "certs/hip/gfx906/mmq_f16_q8_1_gfx906.json",
    },
    // Tile-M F16 × Q8_1 MMQ for m >= 8. Call-site dispatched
    // alongside mmq_f16_q8_1 inside qmatmul.rs::dispatch_qmatmul.
    DirectCallKernel {
        impl_id: "mmq_f16_tile_gfx906",
        cert_rel_path: "certs/hip/gfx906/mmq_f16_tile_gfx906.json",
    },
    // Templated cross-backend F16 certs (neutral impl_ids; same kernels as the
    // _gfx906 dispatch rows above, registered as bench-references).
    DirectCallKernel {
        impl_id: "qmatmul_f16_mmvq",
        cert_rel_path: "certs/hip/gfx906/qmatmul_f16_mmvq.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_f16_mmq",
        cert_rel_path: "certs/hip/gfx906/qmatmul_f16_mmq.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_f16_mmq_tile",
        cert_rel_path: "certs/hip/gfx906/qmatmul_f16_mmq_tile.json",
    },
    // Indexed-MoE MMVQ for Q8_0 / Q4_0 / Q6_K
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
    // crates/bench/src/sweep_moe.rs (generic tile8 harness).
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
    // Templated mmvq dp4a reference: HIP production dispatches the t128_vdr2
    // Q8_0 row, but the shared `mmvq_dp4a_row<Q8_0>` instantiation is certed on
    // gfx906 too as the cross-backend bit-match baseline against the sm_86 cert.
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmvq_dp4a",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmvq_dp4a.json",
    },
    // Q8_0 config variants (mmvq_dp4a_row/mmvq_dp4a_vdr_row instantiations) —
    // autotune candidates on both backends; t128_vdr2 is also the production
    // decode kernel via the runtime alias, not a standalone dispatch row.
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmvq_t128",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmvq_t128.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmvq_t128_vdr2",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmvq_t128_vdr2.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmvq_r2",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmvq_r2.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmvq_r4",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmvq_r4.json",
    },
    // Fused gate+up dense FFN decode kernel — launched directly by the ops
    // layer (mmvq_q8_0_gate_up), not via an m-range dispatch row.
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_gate_up_t128_vdr2",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_gate_up_t128_vdr2.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_0_gate_up_dp4a",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_0_gate_up_dp4a.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_1_gate_up_dp4a",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_1_gate_up_dp4a.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q5_k_gate_up_dp4a",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q5_k_gate_up_dp4a.json",
    },
    // Batched (N-slot) MMVQ — launched directly for N>1 concurrent decode, not
    // via an m-range dispatch row.
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmvq_batched",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmvq_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_0_mmvq_batched",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_0_mmvq_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_1_mmvq_batched",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_1_mmvq_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_k_mmvq_batched",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_k_mmvq_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q6_k_mmvq_batched",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q6_k_mmvq_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q5_k_r2_mmvq_batched",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q5_k_r2_mmvq_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmvq_row_tile_batched",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmvq_row_tile_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_0_mmvq_row_tile_batched",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_0_mmvq_row_tile_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q5_k_mmvq_row_tile_batched",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q5_k_mmvq_row_tile_batched.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_0_gate_up_row_tile_mmvq_batched",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_0_gate_up_row_tile_mmvq_batched.json",
    },
    // MMQ correctness oracle (cross-backend pilot) — batched dot, not dispatched.
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmq_oracle",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmq_oracle.json",
    },
    // Q8_0 single-pass tiled MMQ GEMM (cross-backend); HIP dispatch uses the
    // legacy _gfx906 row, this is the neutral cross-backend cert.
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmq_wave64_tile16",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmq_wave64_tile16.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_0_mmq_4warp",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmq_4warp.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_k_mmq_4warp",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_k_mmq_4warp.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q6_k_mmq_4warp",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q6_k_mmq_4warp.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_0_mmq_4warp_lds",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_0_mmq_4warp_lds.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_1_mmq_4warp_lds",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_1_mmq_4warp_lds.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_k_mmq_lds_tiled",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_k_mmq_lds_tiled.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_0_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_0_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_1_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_1_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q5_0_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q5_0_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q5_1_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q5_1_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q6_k_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q6_k_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q4_k_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_k_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q5_k_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q5_k_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q2_k_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q2_k_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q3_k_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q3_k_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q8_k_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_k_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq4_nl_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq4_nl_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq4_xs_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq4_xs_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq2_xxs_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq2_xxs_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq2_xs_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq2_xs_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq2_s_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq2_s_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq3_xxs_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq3_xxs_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq3_s_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq3_s_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq1_s_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq1_s_mmq_wave64.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_iq1_m_mmq_wave64",
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq1_m_mmq_wave64.json",
    },
    // Q3_K / Q2_K templated mmvq dp4a — not dispatched on HIP (decode uses
    // other rows), kept as the cross-backend bit-match baseline for the sm_86
    // certs.
    DirectCallKernel {
        impl_id: "qmatmul_q3_k_mmvq_dp4a",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q3_k_mmvq_dp4a.json",
    },
    DirectCallKernel {
        impl_id: "qmatmul_q2_k_mmvq_dp4a",
        cert_rel_path: "certs/hip/gfx906/qmatmul_q2_k_mmvq_dp4a.json",
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
    // P2d — indexed-MoE MMVQ, neutral cross-backend certs. HIP dispatch uses
    // the legacy _gfx906 rows; these prove the templated kernel matches on
    // both backends (the sm_86 twin lives under certs/cuda/sm_86/).
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q8_0",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q8_0.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_0",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q4_0.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_1",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q4_1.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q5_0",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q5_0.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q5_1",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q5_1.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_k_r2",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q4_k_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_k_r4",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q4_k_r4.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q5_k_r2",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q5_k_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q6_k_r2",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q6_k_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q2_k_r2",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q2_k_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q3_k_r2",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q3_k_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq4_nl_r2",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_iq4_nl_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq4_xs_r2",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_iq4_xs_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq2_xxs_r2",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_iq2_xxs_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq2_xs_r2",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_iq2_xs_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq2_s_r2",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_iq2_s_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq3_xxs_r2",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_iq3_xxs_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq3_s_r2",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_iq3_s_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq1_s_r2",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_iq1_s_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_iq1_m_r2",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_iq1_m_r2.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q8_0_gate_up",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q8_0_gate_up.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_0_gate_up",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q4_0_gate_up.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_k_gate_up",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q4_k_gate_up.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q4_k_gate_up_r4_sorted",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q4_k_gate_up_r4_sorted.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q8_0_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q8_0_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q8_0_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q8_0_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_0_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_0_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_0_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_0_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_1_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_1_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_1_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_1_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q5_0_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q5_0_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q5_0_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q5_0_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q5_1_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q5_1_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q5_1_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q5_1_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_k_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_k_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_k_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_k_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q5_k_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q5_k_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q5_k_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q5_k_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q6_k_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q6_k_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q6_k_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q6_k_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q2_k_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q2_k_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q2_k_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q2_k_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q3_k_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q3_k_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q3_k_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q3_k_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq4_nl_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq4_nl_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq4_nl_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq4_nl_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq4_xs_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq4_xs_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq4_xs_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq4_xs_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq2_xxs_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq2_xxs_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq2_xxs_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq2_xxs_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq2_xs_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq2_xs_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq2_xs_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq2_xs_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq2_s_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq2_s_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq2_s_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq2_s_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq3_xxs_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq3_xxs_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq3_xxs_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq3_xxs_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq3_s_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq3_s_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq3_s_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq3_s_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq1_s_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq1_s_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq1_s_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq1_s_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq1_m_gate_up_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq1_m_gate_up_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_iq1_m_down_tile8",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_iq1_m_down_tile8.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_k_gate_up_lds_tiled",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_k_gate_up_lds_tiled.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmq_q4_k_down_lds_tiled",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_k_down_lds_tiled.json",
    },
    // P3 Stream F — cross-backend (neutral impl_id) certs for the fixed
    // pointwise kernels. Same gfx906 kernels as the `_gfx906` dispatch rows
    // above; these are the cross-backend bit-match baselines against the
    // sm_86 twins under certs/cuda/sm_86/. Sweep harness:
    // crates/bench/src/sweep_pointwise_cb.rs.
    DirectCallKernel {
        impl_id: "add_f16",
        cert_rel_path: "certs/hip/gfx906/add_f16.json",
    },
    DirectCallKernel {
        impl_id: "cast_f16_f32",
        cert_rel_path: "certs/hip/gfx906/cast_f16_f32.json",
    },
    DirectCallKernel {
        impl_id: "cast_f32_f16",
        cert_rel_path: "certs/hip/gfx906/cast_f32_f16.json",
    },
    DirectCallKernel {
        impl_id: "scale_f32",
        cert_rel_path: "certs/hip/gfx906/scale_f32.json",
    },
    DirectCallKernel {
        impl_id: "scale_f16",
        cert_rel_path: "certs/hip/gfx906/scale_f16.json",
    },
    DirectCallKernel {
        impl_id: "silu_f32",
        cert_rel_path: "certs/hip/gfx906/silu_f32.json",
    },
    DirectCallKernel {
        impl_id: "apply_softcap_f32",
        cert_rel_path: "certs/hip/gfx906/apply_softcap_f32.json",
    },
    DirectCallKernel {
        impl_id: "swiglu_f16",
        cert_rel_path: "certs/hip/gfx906/swiglu_f16.json",
    },
    DirectCallKernel {
        impl_id: "sigmoid_mul_f16",
        cert_rel_path: "certs/hip/gfx906/sigmoid_mul_f16.json",
    },
    DirectCallKernel {
        impl_id: "swiglu_f32_to_q8_1",
        cert_rel_path: "certs/hip/gfx906/swiglu_f32_to_q8_1.json",
    },
    DirectCallKernel {
        impl_id: "split_q_gate_f16",
        cert_rel_path: "certs/hip/gfx906/split_q_gate_f16.json",
    },
    // Tier-B block-reduce kernels, neutralized to a shared `.cuh` (the body
    // rides block_reduce_sum over the WARP_SIZE seam). The `_gfx906` rows above
    // remain the dispatch certs; these are the cross-backend baselines.
    DirectCallKernel {
        impl_id: "rmsnorm_f32",
        cert_rel_path: "certs/hip/gfx906/rmsnorm_f32.json",
    },
    DirectCallKernel {
        impl_id: "l2_norm_f32",
        cert_rel_path: "certs/hip/gfx906/l2_norm_f32.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_f16",
        cert_rel_path: "certs/hip/gfx906/rmsnorm_f16.json",
    },
    DirectCallKernel {
        impl_id: "quantize_f16_q8_1",
        cert_rel_path: "certs/hip/gfx906/quantize_f16_q8_1.json",
    },
    DirectCallKernel {
        impl_id: "quantize_f16_q8_0",
        cert_rel_path: "certs/hip/gfx906/quantize_f16_q8_0.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_f32_to_f16",
        cert_rel_path: "certs/hip/gfx906/rmsnorm_f32_to_f16.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_f32_to_f16_add_residual",
        cert_rel_path: "certs/hip/gfx906/rmsnorm_f32_to_f16_add_residual.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_f16_to_f16_add_residual",
        cert_rel_path: "certs/hip/gfx906/rmsnorm_f16_to_f16_add_residual.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_f16_add_residual",
        cert_rel_path: "certs/hip/gfx906/rmsnorm_f16_add_residual.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_q8_1_fused",
        cert_rel_path: "certs/hip/gfx906/rmsnorm_q8_1_fused.json",
    },
    DirectCallKernel {
        impl_id: "rope_f16",
        cert_rel_path: "certs/hip/gfx906/rope_f16.json",
    },
    DirectCallKernel {
        impl_id: "rope_neox_partial_f16",
        cert_rel_path: "certs/hip/gfx906/rope_neox_partial_f16.json",
    },
    DirectCallKernel {
        impl_id: "moe_combine_no_residual_f16",
        cert_rel_path: "certs/hip/gfx906/moe_combine_no_residual_f16.json",
    },
    DirectCallKernel {
        impl_id: "moe_combine_f16",
        cert_rel_path: "certs/hip/gfx906/moe_combine_f16.json",
    },
    DirectCallKernel {
        impl_id: "moe_combine_two_residuals_f16",
        cert_rel_path: "certs/hip/gfx906/moe_combine_two_residuals_f16.json",
    },
    DirectCallKernel {
        impl_id: "apply_per_expert_scale_f32",
        cert_rel_path: "certs/hip/gfx906/apply_per_expert_scale_f32.json",
    },
    DirectCallKernel {
        impl_id: "shared_expert_scale_f32",
        cert_rel_path: "certs/hip/gfx906/shared_expert_scale_f32.json",
    },
    DirectCallKernel {
        impl_id: "topk_f32",
        cert_rel_path: "certs/hip/gfx906/topk_f32.json",
    },
    DirectCallKernel {
        impl_id: "sampler_apply_penalties_f32",
        cert_rel_path: "certs/hip/gfx906/sampler_apply_penalties_f32.json",
    },
    DirectCallKernel {
        impl_id: "softmax_masked_f16",
        cert_rel_path: "certs/hip/gfx906/softmax_masked_f16.json",
    },
    DirectCallKernel {
        impl_id: "sampler_topk_softmax_f32",
        cert_rel_path: "certs/hip/gfx906/sampler_topk_softmax_f32.json",
    },
    DirectCallKernel {
        impl_id: "kv_append_f16_batched_slots",
        cert_rel_path: "certs/hip/gfx906/kv_append_f16_batched_slots.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_f16",
        cert_rel_path: "certs/hip/gfx906/attention_decode_f16.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_f16_splitk",
        cert_rel_path: "certs/hip/gfx906/attention_decode_f16_splitk.json",
    },
    DirectCallKernel {
        impl_id: "moe_sort_by_expert",
        cert_rel_path: "certs/hip/gfx906/moe_sort_by_expert.json",
    },
    // P3 Wave 2 — pointwise activations + F32 MoE combine.
    DirectCallKernel {
        impl_id: "swiglu_f32",
        cert_rel_path: "certs/hip/gfx906/swiglu_f32.json",
    },
    DirectCallKernel {
        impl_id: "swiglu_f32_to_f16",
        cert_rel_path: "certs/hip/gfx906/swiglu_f32_to_f16.json",
    },
    DirectCallKernel {
        impl_id: "gelu_mul_f32",
        cert_rel_path: "certs/hip/gfx906/gelu_mul_f32.json",
    },
    DirectCallKernel {
        impl_id: "gelu_f32_to_f16",
        cert_rel_path: "certs/hip/gfx906/gelu_f32_to_f16.json",
    },
    DirectCallKernel {
        impl_id: "moe_combine_no_residual_f32",
        cert_rel_path: "certs/hip/gfx906/moe_combine_no_residual_f32.json",
    },
    DirectCallKernel {
        impl_id: "dense_gemv_f32_f16",
        cert_rel_path: "certs/hip/gfx906/dense_gemv_f32_f16.json",
    },
    DirectCallKernel {
        impl_id: "dense_gemv_f16_f16",
        cert_rel_path: "certs/hip/gfx906/dense_gemv_f16_f16.json",
    },
    DirectCallKernel {
        impl_id: "dense_gemv_f32_f16_batched",
        cert_rel_path: "certs/hip/gfx906/dense_gemv_f32_f16_batched.json",
    },
    DirectCallKernel {
        impl_id: "dense_gemv_f16_f16_batched",
        cert_rel_path: "certs/hip/gfx906/dense_gemv_f16_f16_batched.json",
    },
    DirectCallKernel {
        impl_id: "v_unit_norm_per_head_f16",
        cert_rel_path: "certs/hip/gfx906/v_unit_norm_per_head_f16.json",
    },
    DirectCallKernel {
        impl_id: "kv_append_v_unit_norm_f16",
        cert_rel_path: "certs/hip/gfx906/kv_append_v_unit_norm_f16.json",
    },
    DirectCallKernel {
        impl_id: "rmsnorm_rope_neox_partial_f16",
        cert_rel_path: "certs/hip/gfx906/rmsnorm_rope_neox_partial_f16.json",
    },
    // P3 Wave 4 — GDN (qwen3-coder-next) F32 kernels.
    DirectCallKernel {
        impl_id: "gdn_alpha_beta_f32",
        cert_rel_path: "certs/hip/gfx906/gdn_alpha_beta_f32.json",
    },
    DirectCallKernel {
        impl_id: "gdn_split_qkv_f32",
        cert_rel_path: "certs/hip/gfx906/gdn_split_qkv_f32.json",
    },
    DirectCallKernel {
        impl_id: "gdn_assemble_conv_input_f32",
        cert_rel_path: "certs/hip/gfx906/gdn_assemble_conv_input_f32.json",
    },
    DirectCallKernel {
        impl_id: "causal_conv1d_f32",
        cert_rel_path: "certs/hip/gfx906/causal_conv1d_f32.json",
    },
    DirectCallKernel {
        impl_id: "gdn_conv_trio_decode_f32_batched_slots",
        cert_rel_path: "certs/hip/gfx906/gdn_conv_trio_decode_f32_batched_slots.json",
    },
    DirectCallKernel {
        impl_id: "gdn_state_step_f32",
        cert_rel_path: "certs/hip/gfx906/gdn_state_step_f32.json",
    },
    DirectCallKernel {
        impl_id: "gdn_state_step_alphabeta_f32",
        cert_rel_path: "certs/hip/gfx906/gdn_state_step_alphabeta_f32.json",
    },
    DirectCallKernel {
        impl_id: "gdn_state_step_alphabeta_f32_batched_slots",
        cert_rel_path: "certs/hip/gfx906/gdn_state_step_alphabeta_f32_batched_slots.json",
    },
    // P3 Wave 3 — paged KV append + attention serving paths.
    DirectCallKernel {
        impl_id: "kv_append_f16_paged_slots",
        cert_rel_path: "certs/hip/gfx906/kv_append_f16_paged_slots.json",
    },
    DirectCallKernel {
        impl_id: "kv_append_f16_paged_prefill",
        cert_rel_path: "certs/hip/gfx906/kv_append_f16_paged_prefill.json",
    },
    DirectCallKernel {
        impl_id: "attention_prefill_f16",
        cert_rel_path: "certs/hip/gfx906/attention_prefill_f16.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_f16_batched",
        cert_rel_path: "certs/hip/gfx906/attention_decode_f16_batched.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_f16_paged",
        cert_rel_path: "certs/hip/gfx906/attention_decode_f16_paged.json",
    },
    DirectCallKernel {
        impl_id: "attention_prefill_f16_paged",
        cert_rel_path: "certs/hip/gfx906/attention_prefill_f16_paged.json",
    },
    DirectCallKernel {
        impl_id: "attention_prefill_flash_tile_f16",
        cert_rel_path: "certs/hip/gfx906/attention_prefill_flash_tile_f16.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_q8_kv",
        cert_rel_path: "certs/hip/gfx906/attention_decode_q8_kv.json",
    },
    DirectCallKernel {
        impl_id: "attention_prefill_q8_kv",
        cert_rel_path: "certs/hip/gfx906/attention_prefill_q8_kv.json",
    },
    DirectCallKernel {
        impl_id: "attention_decode_q8_kv_splitk",
        cert_rel_path: "certs/hip/gfx906/attention_decode_q8_kv_splitk.json",
    },
    DirectCallKernel {
        impl_id: "attention_prefill_flash_tile_q8_kv",
        cert_rel_path: "certs/hip/gfx906/attention_prefill_flash_tile_q8_kv.json",
    },
    DirectCallKernel {
        impl_id: "quantize_f16_q8_1_mmq",
        cert_rel_path: "certs/hip/gfx906/quantize_f16_q8_1_mmq.json",
    },
];
