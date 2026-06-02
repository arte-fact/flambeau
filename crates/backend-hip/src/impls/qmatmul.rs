//! `QMatMul` `KernelDescriptor` table for `gfx906`. Split out of
//! `impls.rs` so its ~490 LOC compiles as a separate codegen unit.

use flambeau_core::{KernelDescriptor, QDtype};

/// Static table of every `QMatMul` impl on `gfx906` this build ships. Order
/// matches `dispatch/hip/gfx906.toml`; the first `matches` win is returned.
pub const QMATMUL_GFX906: &[KernelDescriptor] = &[
    // MMVQ (decode): m ∈ [1, 3] only for Q8_0 — MMQ oracle handles 4..127,
    // MMQ 4-warp handles ≥128. Predicates must not overlap (architectural
    // rule: "two rows matching the same concrete shape = build-time error").
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q8_0_mmvq_single_row_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q8_0,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 3),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmvq_single_row_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q4_K_mmvq_dp4a_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q4_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_K_mmvq_dp4a_gfx906.json",
    },
    // native IQ4 MMVQ. r2 multi-row is the decode default
    // (parity tested 4..16 rows × 256..5120 K). MMQ + MoE indexed
    // variants land in . Wave64-shaped, scalar inner loop —
    // dp4a-with-LUT deferred until a measured bottleneck appears.
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq4_xs_mmvq_dp4a_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ4_XS,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq4_xs_mmvq_dp4a_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq4_xs_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ4_XS,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq4_xs_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq4_nl_mmvq_dp4a_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ4_NL,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq4_nl_mmvq_dp4a_gfx906.json",
    },
    // native IQ3 MMVQ (codebook lookup, 256/512-entry u32 grid).
    // r2 multi-row is the decode default; single-row variant retained for
    // the sweep/cert harness.
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq3_xxs_mmvq_dp4a_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ3_XXS,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq3_xxs_mmvq_dp4a_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq3_s_mmvq_dp4a_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ3_S,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq3_s_mmvq_dp4a_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq3_s_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ3_S,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq3_s_mmq_wave64_gfx906.json",
    },
    // IQ2 + IQ1 family native MMVQ. r2 multi-row default.
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq2_xxs_mmvq_dp4a_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ2_XXS,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq2_xxs_mmvq_dp4a_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq2_xs_mmvq_dp4a_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ2_XS,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq2_xs_mmvq_dp4a_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq2_s_mmvq_dp4a_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ2_S,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq2_s_mmvq_dp4a_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq1_s_mmvq_dp4a_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ1_S,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq1_s_mmvq_dp4a_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq1_m_mmvq_dp4a_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ1_M,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq1_m_mmvq_dp4a_gfx906.json",
    },
    KernelDescriptor {
        // 4.a.3 — 128-thread single-row DP4A Q4_1 MMVQ. Same DP4A
        // math as but half the threads per block, reducing kernel
        // dispatch overhead + increasing CU occupancy. r1 scalar +
        // r2 DP4A attempts NULL'd; thin-block is the last MMVQ lever.
        op_name: "QMatMul",
        impl_id: "qmatmul_q4_1_mmvq_t128_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q4_1,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_1_mmvq_t128_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q5_K_mmvq_dp4a_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q5_K,
        dtype_activation: QDtype::Q8_1,
        // DP4A port (90aa05c). End-to-end measured +14.6% wall t/s on
        // Qwen3.6-27B-Q4_0 PP4 single-stream decode (17.29 → 19.82
        // t/s, 200-token greedy), microbench shows 2.48× per-call vs
        // the prior nw1_r2 at the production shape. m>=128 hits the
        // wave64 MMQ kernel below.
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q5_K_mmvq_dp4a_gfx906.json",
    },
    // Dormant nw1_r2 baseline — preserved for A/B regression checks.
    // m_range=(MAX,MAX) keeps shape-dispatch from picking it; the cert
    // at certs/hip/gfx906/qmatmul_q5_K_mmvq_nw1_r2_gfx906.json stays
    // green for cross-validation runs.
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q5_K_mmvq_nw1_r2_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q5_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (usize::MAX, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q5_K_mmvq_nw1_r2_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q5_K_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q5_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q5_K_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        // the DP4A kernel (3.24× per-call win on LM-head) used
        // to swap in via a runtime intercept in `ops/qmatmul.rs`, which
        // routed under the r4 scalar cert even though the actual kernel
        // was uncertified. Sweep-certed the DP4A variant directly and
        // also fixed a 32-bit unsigned-subtract borrow-chain bug in it
        // (same class as mmq_q6_K_wave64 fix). Now dispatched
        // directly, intercept removed.
        impl_id: "qmatmul_q6_K_mmvq_dp4a_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q6_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q6_K_mmvq_dp4a_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q8_K_mmvq_single_row_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q8_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_K_mmvq_single_row_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q2_K_mmvq_r2_dp4a_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q2_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q2_K_mmvq_r2_dp4a_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q3_K_mmvq_r2_dp4a_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q3_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q3_K_mmvq_r2_dp4a_gfx906.json",
    },
    // MMQ (prefill): m ≥ 128
    KernelDescriptor {
        op_name: "QMatMul",
        // TILE_N=16 wave64 MMQ replaces TILE_N=8 at m ≥ 128. Halves weight
        // HBM fetches (each decoded weight tile reused across 16 activations vs 8).
        // Kernel PMC at m=512 k=2048 n=4096: MemUnitBusy 97.6 % → 83.1 %,
        // VALUBusy 19.1 % → 65.5 %. End-to-end Qwen3.6-35B prefill is a wash
        // (MoE Q4_K dominates); tile16 wins show on Q8_0-heavy workloads.
        // `FLAMBEAU_VARIANT=baseline` reverts to the TILE_N=8 kernel.
        // 1.b: dispatch_qmatmul() overrides tile16 → tile8 when n < 1024
        // (small-N shapes like shexp gate/up k=2048 n=512 lose at tile16).
        impl_id: "qmatmul_q8_0_mmq_wave64_tile16_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q8_0,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmq_wave64_tile16_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        // 3.a: wave64 port of Q4_1 MMQ. Never matched by default dispatch
        // (m_range MAX..MAX); lookup-only. Promoted to default after 3.b
        // A/B confirms uplift. 9.e recycle A/B'd this against 4warp_lds
        // as default — regressed -60 % (see v2_29_e_research_null.md cert).
        impl_id: "qmatmul_q4_1_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q4_1,
        dtype_activation: QDtype::Q8_1,
        m_range: (usize::MAX, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_1_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q4_1_mmq_4warp_lds_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q4_1,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_1_mmq_4warp_lds_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        // 4-warp LDS-tiled Q4_0 MMQ (port of Q4_1 4warp_lds with
        // Q4_0 bias-correction in the dot). Engages at m >= 128 ahead of the
        // wave64 row below; wave64 still owns m=32..127 where the larger tile's
        // grid-fill dominates. Closes part of the dense-prefill gap on 27B-Q4_0.
        impl_id: "qmatmul_q4_0_mmq_4warp_lds_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q4_0,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_0_mmq_4warp_lds_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        // 8.b: wave64 Q4_0 MMQ — owns m=32..127 (the C1 4warp_lds row above
        // takes m >= 128). At m < 32 the qmatmul() wrapper short-circuits to
        // the Q4_0 MMVQ single-row kernel (3).
        impl_id: "qmatmul_q4_0_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q4_0,
        dtype_activation: QDtype::Q8_1,
        m_range: (32, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_0_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        // 0.a: wave64 Q5_0 MMQ — shexp dense prefill for
        // Qwen3.6-35B-A3B-Q4_0 (20/40 layers have Q5_0 shared-expert FFN).
        // Default dispatched at m >= 32; qmatmul() short-circuits m < 32 to
        // the Q5_0 MMVQ single-row kernel (3).
        impl_id: "qmatmul_q5_0_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q5_0,
        dtype_activation: QDtype::Q8_1,
        m_range: (32, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q5_0_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q5_1_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q5_1,
        dtype_activation: QDtype::Q8_1,
        m_range: (32, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q5_1_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        // 4.b kernel, promoted (2026-04-27): llamacpp-turbo
        // 4-warp LDS-tiled Q4_K MMQ port. 256 threads (4 warps × 64), MMQ_Y=128,
        // MMQ_X=16, double-buffered Y LDS per super-block. Owns m >= 128; the
        // wave64 row below covers m = 32..127. Static dispatch is first-match
        // — keep this entry ABOVE the wave64 row (lesson).
        impl_id: "qmatmul_q4_K_mmq_turbo_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q4_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_K_mmq_turbo_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        // candle port `mmq_q4_K_wave64.cu`; structural mirror of the
        // Q5_K wave64 with flat 4-bit nibble decode (no qh merge).
        // Owns m = 32..127 since promoted Q4_K turbo to (128, MAX).
        impl_id: "qmatmul_q4_K_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q4_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (32, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_K_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        // wave64 MMQ replaces the F32-tile placeholder
        // (`qmatmul_q6_K_mmq_4warp_lds_gfx906`) at m >= 128. Kernel:
        // `mmq_q6_K_wave64.cu`, authored fresh — candle has no Q6_K MMQ.
        impl_id: "qmatmul_q6_K_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q6_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q6_K_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q8_K_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q8_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_K_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q2_K_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q2_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q2_K_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q3_K_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q3_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q3_K_mmq_wave64_gfx906.json",
    },
    // MMQ oracle covers the mid-M band (4..128) for Q8_0.
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_q8_0_mmq_oracle_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q8_0,
        dtype_activation: QDtype::Q8_1,
        m_range: (4, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q8_0_mmq_oracle_gfx906.json",
    },
    //— remaining 7 dense MMQ wave64 (IQ4_NL,
    // IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M). Same wave64 tile shape.
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq4_nl_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ4_NL,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq4_nl_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq3_xxs_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ3_XXS,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq3_xxs_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq2_xxs_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ2_XXS,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq2_xxs_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq2_xs_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ2_XS,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq2_xs_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq2_s_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ2_S,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq2_s_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq1_s_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ1_S,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq1_s_mmq_wave64_gfx906.json",
    },
    KernelDescriptor {
        op_name: "QMatMul",
        impl_id: "qmatmul_iq1_m_mmq_wave64_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::IQ1_M,
        dtype_activation: QDtype::Q8_1,
        m_range: (128, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/qmatmul_iq1_m_mmq_wave64_gfx906.json",
    },
];
