//! `KernelImpl` registrations for `QMatMul` on `HipDevice`.
//! Each row corresponds to a `[[qmatmul]]` / `[[qmatmul_mmq]]` entry in
//! `dispatch/hip/gfx906.toml` and a cert under `certs/hip/gfx906/`. This
//! module is the Rust-typed mirror of the TOML — the dispatcher calls
//! [`dispatch_qmatmul`] with a runtime `QMatMulCfg` and gets back the
//! matching `KernelDescriptor` (or `None` if no impl applies).

use flambeau_core::{DirectCallKernel, KernelDescriptor, QDtype, QMatMulCfg};

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
        impl_id: "qmatmul_q4_K_mmvq_nw1_r2_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q4_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q4_K_mmvq_nw1_r2_gfx906.json",
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
        impl_id: "qmatmul_q5_K_mmvq_nw1_r2_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q5_K,
        dtype_activation: QDtype::Q8_1,
        // narrowed from (1, 512) to (1, 127) so prefill paths
        // route to the wave64 MMQ kernel below instead of looping MMVQ per row.
        m_range: (1, 127),
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
        impl_id: "qmatmul_q3_K_mmvq_single_row_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q3_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, 127),
        cert_rel_path: "certs/hip/gfx906/qmatmul_q3_K_mmvq_single_row_gfx906.json",
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
];

/// RMSNorm + SwiGLU + D1 fused. The dispatch key is `(dtype_in,
/// dtype_out)` + the shape m-band, but since all three impls accept any
/// `m`, the m_range is `(1, usize::MAX)` and we route solely on dtype pair.
pub const RMSNORM_GFX906: &[KernelDescriptor] = &[
    KernelDescriptor {
        op_name: "RmsNorm",
        impl_id: "rmsnorm_f16_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F16,   // input x dtype
        dtype_activation: QDtype::F16,   // output y dtype
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/rmsnorm_f16_gfx906.json",
    },
    KernelDescriptor {
        op_name: "RmsNorm",
        impl_id: "rmsnorm_q8_1_fused_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F16,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/rmsnorm_q8_1_fused_gfx906.json",
    },
];

pub const SWIGLU_GFX906: &[KernelDescriptor] = &[
    KernelDescriptor {
        op_name: "SwiGLU",
        impl_id: "swiglu_f16_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F16,
        dtype_activation: QDtype::F16,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/swiglu_f16_gfx906.json",
    },
];

pub const ROPE_GFX906: &[KernelDescriptor] = &[
    KernelDescriptor {
        op_name: "RoPE",
        impl_id: "rope_f16_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F16,
        dtype_activation: QDtype::F16,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/rope_f16_gfx906.json",
    },
];

pub const SOFTMAX_GFX906: &[KernelDescriptor] = &[
    KernelDescriptor {
        op_name: "Softmax",
        impl_id: "softmax_masked_f16_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F16,
        dtype_activation: QDtype::F16,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/softmax_masked_f16_gfx906.json",
    },
];

pub const ATTENTION_DECODE_GFX906: &[KernelDescriptor] = &[
    KernelDescriptor {
        op_name: "AttentionDecode",
        impl_id: "attention_decode_f16_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F16,    // KV dtype
        dtype_activation: QDtype::F16, // Q/out dtype
        m_range: (1, 4),              // decode path (M = seq_len tokens this step)
        cert_rel_path: "certs/hip/gfx906/attention_decode_f16_gfx906.json",
    },
    KernelDescriptor {
        op_name: "AttentionDecode",
        impl_id: "attention_decode_q8_kv_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q8_0,   // KV stored as Q8_0 blocks
        dtype_activation: QDtype::F16,
        m_range: (1, 4),
        cert_rel_path: "certs/hip/gfx906/attention_decode_q8_kv_gfx906.json",
    },
];

pub const ATTENTION_PREFILL_GFX906: &[KernelDescriptor] = &[
    KernelDescriptor {
        op_name: "AttentionPrefill",
        impl_id: "attention_prefill_f16_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F16,
        dtype_activation: QDtype::F16,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/attention_prefill_f16_gfx906.json",
    },
];

// ----- MoE -------------------------------------------------------------

pub const TOPK_GFX906: &[KernelDescriptor] = &[
    KernelDescriptor {
        op_name: "TopK",
        impl_id: "topk_f32_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F32,
        dtype_activation: QDtype::F32,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/topk_f32_gfx906.json",
    },
];

pub const SPLIT_Q_GATE_GFX906: &[KernelDescriptor] = &[
    // split (Q | gate) output of Qwen3.5/3.6 gated attention.
    KernelDescriptor {
        op_name: "SplitQGate",
        impl_id: "split_q_gate_f16_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F16,
        dtype_activation: QDtype::F16,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/split_q_gate_f16_gfx906.json",
    },
];

pub const SHARED_EXPERT_SCALE_GFX906: &[KernelDescriptor] = &[
    // fused per-token sigmoid-gate scaling for shared expert.
    KernelDescriptor {
        op_name: "SharedExpertScale",
        impl_id: "shared_expert_scale_f32_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F32,
        dtype_activation: QDtype::F32,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/shared_expert_scale_f32_gfx906.json",
    },
];

pub const CAUSAL_CONV1D_GFX906: &[KernelDescriptor] = &[
    // depthwise causal conv1d used inside GDN.
    KernelDescriptor {
        op_name: "CausalConv1d",
        impl_id: "causal_conv1d_f32_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F32,
        dtype_activation: QDtype::F32,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/causal_conv1d_f32_gfx906.json",
    },
];

pub const ADD_F16_GFX906: &[KernelDescriptor] = &[
    // e1 — pointwise F16 + F16 → F16. Residual fan-in primitive.
    KernelDescriptor {
        op_name: "AddF16",
        impl_id: "add_f16_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F16,
        dtype_activation: QDtype::F16,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/add_f16_gfx906.json",
    },
];

pub const CAST_F16_F32_GFX906: &[KernelDescriptor] = &[
    // c1 glue — F16 → F32 for the GDN path's internal recurrence.
    KernelDescriptor {
        op_name: "CastF16F32",
        impl_id: "cast_f16_f32_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F16,
        dtype_activation: QDtype::F32,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/cast_f16_f32_gfx906.json",
    },
];

pub const SILU_F32_GFX906: &[KernelDescriptor] = &[
    // c1 — standalone F32 SiLU for GDN's silu(conv_out).
    KernelDescriptor {
        op_name: "SiluF32",
        impl_id: "silu_f32_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F32,
        dtype_activation: QDtype::F32,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/silu_f32_gfx906.json",
    },
];

pub const SWIGLU_F32_GFX906: &[KernelDescriptor] = &[
    // c1 — F32 SwiGLU for GDN's gated output compose.
    KernelDescriptor {
        op_name: "SwigluF32",
        impl_id: "swiglu_f32_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F32,
        dtype_activation: QDtype::F32,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/swiglu_f32_gfx906.json",
    },
];

pub const SCALE_F32_GFX906: &[KernelDescriptor] = &[
    // c1 — pointwise scalar multiply. GDN scales Q by 1/sqrt(head_k_dim).
    KernelDescriptor {
        op_name: "ScaleF32",
        impl_id: "scale_f32_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F32,
        dtype_activation: QDtype::F32,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/scale_f32_gfx906.json",
    },
];

pub const RMSNORM_F32_GFX906: &[KernelDescriptor] = &[
    // c1 — F32 RMSNorm for GDN's ssm_norm (per-head on F32 state out).
    KernelDescriptor {
        op_name: "RmsnormF32",
        impl_id: "rmsnorm_f32_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F32,
        dtype_activation: QDtype::F32,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/rmsnorm_f32_gfx906.json",
    },
];

pub const CAST_F32_F16_GFX906: &[KernelDescriptor] = &[
    // b glue — F32 MMVQ output → F16 for attention / rmsnorm inputs.
    KernelDescriptor {
        op_name: "CastF32F16",
        impl_id: "cast_f32_f16_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F32,
        dtype_activation: QDtype::F16,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/cast_f32_f16_gfx906.json",
    },
];

pub const QUANTIZE_F16_Q8_1_GFX906: &[KernelDescriptor] = &[
    // g — F16 → Q8_1 activation quantise. Replaces the b
    // host-roundtrip placeholder between swiglu and the output MMVQ.
    KernelDescriptor {
        op_name: "QuantizeF16Q8_1",
        impl_id: "quantize_f16_q8_1_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F16,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/quantize_f16_q8_1_gfx906.json",
    },
];

pub const DENSE_GEMV_F32_F16_GFX906: &[KernelDescriptor] = &[
    // d3 — F32 weight × F16 activation → F32 output.
    // The MoE router (ffn_gate_inp × x_norm → expert logits). Single shape.
    KernelDescriptor {
        op_name: "DenseGemvF32F16",
        impl_id: "dense_gemv_f32_f16_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F32,
        dtype_activation: QDtype::F16,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/dense_gemv_f32_f16_gfx906.json",
    },
];

pub const GDN_ALPHA_BETA_GFX906: &[KernelDescriptor] = &[
    // g — fused GDN α/β/gate compute. Replaces the c2 host
    // roundtrip (download α/β/ssm_a/ssm_dt_bias, compute softplus+sigmoid,
    // upload gate/beta) with a single device launch.
    KernelDescriptor {
        op_name: "GdnAlphaBetaF32",
        impl_id: "gdn_alpha_beta_f32_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F32,
        dtype_activation: QDtype::F32,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/gdn_alpha_beta_f32_gfx906.json",
    },
];

pub const GDN_STATE_STEP_GFX906: &[KernelDescriptor] = &[
    // fused Gated-Delta-Net recurrent step at S_v = 128.
    // One launch handles both decode (L = 1) and prefill (L > 1); state
    // stays register-resident across the full L-token recurrence loop.
    KernelDescriptor {
        op_name: "GdnStateStep",
        impl_id: "gdn_state_step_f32_s128_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F32,
        dtype_activation: QDtype::F32,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/gdn_state_step_f32_s128_gfx906.json",
    },
];

pub const L2_NORM_GFX906: &[KernelDescriptor] = &[
    // per-row L2 norm. Used inside GDN.
    KernelDescriptor {
        op_name: "L2Norm",
        impl_id: "l2_norm_f32_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F32,
        dtype_activation: QDtype::F32,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/l2_norm_f32_gfx906.json",
    },
];

pub const ROPE_NEOX_PARTIAL_GFX906: &[KernelDescriptor] = &[
    // NeoX-split partial RoPE for Qwen3.5/3.6 full-attention layers.
    KernelDescriptor {
        op_name: "RoPENeoXPartial",
        impl_id: "rope_neox_partial_f16_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F16,
        dtype_activation: QDtype::F16,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/rope_neox_partial_f16_gfx906.json",
    },
];

pub const INDEXED_MOE_MMVQ_GFX906: &[KernelDescriptor] = &[
    // Active: P29 multi-row r2 (landed ). Halves launches vs single-row,
    // keeps VGPR 24 / waves/SIMD 10.
    KernelDescriptor {
        op_name: "IndexedMoEMMVQ",
        impl_id: "indexed_moe_mmvq_q4_k_r2_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q4_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q4_k_r2_gfx906.json",
    },
];

pub const MOE_COMBINE_GFX906: &[KernelDescriptor] = &[
    KernelDescriptor {
        op_name: "MoECombine",
        impl_id: "moe_combine_f16_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::F16,
        dtype_activation: QDtype::F16,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/moe_combine_f16_gfx906.json",
    },
];

pub const INDEXED_MOE_MMVQ_GATE_UP_GFX906: &[KernelDescriptor] = &[
    KernelDescriptor {
        op_name: "IndexedMoEMMVQGateUp",
        impl_id: "indexed_moe_mmvq_q4_k_gate_up_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q4_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q4_k_gate_up_gfx906.json",
    },
];

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
        impl_id: "indexed_moe_mmvq_q6_k_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q6_k_gfx906.json",
    },
    DirectCallKernel {
        impl_id: "indexed_moe_mmvq_q5_k_gfx906",
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmvq_q5_k_gfx906.json",
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
/// TOML rows for bench baselines.
/// 2. Future simplifier passes have a single source of truth for
/// "this kernel is not dead — it's a reference baseline" and don't
/// propose deletion. (Sessions 1 and 2 of the simplification pass both
/// initially flagged these as orphans; this catalog closes that loop.)
/// Matches the "Single-row reference MMVQ kernels for the K-quants are
/// kept in-tree for cert cross-checks" note in `dispatch/hip/gfx906.toml`.
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
    // 4-warp LDS-tiled MMQ placeholders. Superseded by wave64 in
    // production (Q4_K wave64 , Q6_K wave64 , Q8_0
    // wave64_tile16 ). Kept as bench A/B baselines — `sweep_mmq` runs
    // them as Q{4,6}K4Warp / Q8_04Warp variants alongside the shipped
    // kernels so the comparison stays live. See CLI PMC-refresh target
    // list at `crates/cli/src/main.rs`.
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
    // superseded Q8_0 MMQ wave64 by wave64_tile16 at m >= 128. Wave64
    // stays as the bench baseline (PMC-refresh target).
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

pub const INDEXED_MOE_MMQ_GFX906: &[KernelDescriptor] = &[
    // 4-warp LDS-tiled MoE MMQ. Caller sorts (token, slot) pairs into
    // per-expert buckets so the weight tile amortises across MMQ_X=8 slots.
    KernelDescriptor {
        op_name: "IndexedMoEMMQ",
        impl_id: "indexed_moe_mmq_q4_k_gfx906",
        backend: "hip",
        arch: "gfx906",
        dtype_weight: QDtype::Q4_K,
        dtype_activation: QDtype::Q8_1,
        m_range: (1, usize::MAX),
        cert_rel_path: "certs/hip/gfx906/indexed_moe_mmq_q4_k_gfx906.json",
    },
];

/// Pick the `KernelDescriptor` that matches `cfg`. Predicates are
/// non-overlapping by construction (see architectural rule: "two rows
/// matching the same concrete shape = build-time error"), so first-match
/// wins. Panics in debug if the invariant is violated.
pub fn dispatch_qmatmul(cfg: &QMatMulCfg) -> Option<&'static KernelDescriptor> {
    first_match(QMATMUL_GFX906, cfg.dtype_weight, cfg.dtype_activation, cfg.m)
}
// 1.b was an attempt at shape-aware (n < 1024 → tile8) for Q8_0 MMQ,
// based on microbench PMC showing tile16 MemBusy crashes 75 % → 35 % at
// small-N. End-to-end was a regression: the shexp shapes (k=2048 n=512)
// are already-fast (~0.15 ms/call), and routing them through tile8's
// smaller MMQ_X adds 24 ms of kernel time over keeping them on tile16.
// PMC efficiency > wall-clock only when the per-call work is large
// enough for the efficiency gap to matter. Reverted.

/// Pick the RMSNorm impl for `(dtype_in, dtype_out)` — `rmsnorm_f16_gfx906`
/// when both are F16, `rmsnorm_q8_1_fused_gfx906` when output is Q8_1.
pub fn dispatch_rmsnorm(
    dtype_in: flambeau_core::QDtype,
    dtype_out: flambeau_core::QDtype,
    m: usize,
) -> Option<&'static KernelDescriptor> {
    first_match(RMSNORM_GFX906, dtype_in, dtype_out, m)
}

pub fn dispatch_swiglu(
    dtype: flambeau_core::QDtype,
    m: usize,
) -> Option<&'static KernelDescriptor> {
    first_match(SWIGLU_GFX906, dtype, dtype, m)
}

fn first_match(
    table: &'static [KernelDescriptor],
    dtype_w: flambeau_core::QDtype,
    dtype_a: flambeau_core::QDtype,
    m: usize,
) -> Option<&'static KernelDescriptor> {
    let mut hit: Option<&'static KernelDescriptor> = None;
    for d in table {
        if d.matches(dtype_w, dtype_a, m) {
            if let Some(prev) = hit {
                debug_assert!(
                    false,
                    "overlapping dispatch predicates in table for m={m}: {} and {}",
                    prev.impl_id, d.impl_id
                );
            }
            hit = Some(d);
        }
    }
    hit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(w: QDtype, a: QDtype, m: usize) -> QMatMulCfg {
        QMatMulCfg {
            dtype_weight: w,
            dtype_activation: a,
            m,
            k: 2048,
            n: 2048,
        }
    }

    #[test]
    fn mmvq_decode_selects_single_row_q8_0() {
        let d = dispatch_qmatmul(&cfg(QDtype::Q8_0, QDtype::Q8_1, 1)).unwrap();
        assert_eq!(d.impl_id, "qmatmul_q8_0_mmvq_single_row_gfx906");
    }

    #[test]
    fn mmvq_decode_selects_r2_for_q4_k() {
        let d = dispatch_qmatmul(&cfg(QDtype::Q4_K, QDtype::Q8_1, 1)).unwrap();
        assert_eq!(d.impl_id, "qmatmul_q4_K_mmvq_nw1_r2_gfx906");
    }

    #[test]
    fn mmvq_decode_selects_dp4a_for_q6_k() {
        // Q6_K MMVQ decode dispatch is the DP4A kernel directly
        // (runtime intercept removed after the borrow-chain bug was fixed).
        let d = dispatch_qmatmul(&cfg(QDtype::Q6_K, QDtype::Q8_1, 1)).unwrap();
        assert_eq!(d.impl_id, "qmatmul_q6_K_mmvq_dp4a_gfx906");
    }

    #[test]
    fn mmq_oracle_at_m_8() {
        let d = dispatch_qmatmul(&cfg(QDtype::Q8_0, QDtype::Q8_1, 8)).unwrap();
        assert_eq!(d.impl_id, "qmatmul_q8_0_mmq_oracle_gfx906");
    }

    #[test]
    fn mmq_tile16_at_m_128() {
        let d = dispatch_qmatmul(&cfg(QDtype::Q8_0, QDtype::Q8_1, 128)).unwrap();
        // default large-N Q8_0 MMQ is tile16.
        assert_eq!(d.impl_id, "qmatmul_q8_0_mmq_wave64_tile16_gfx906");
    }

    #[test]
    fn mmq_tile16_at_m_2048() {
        let d = dispatch_qmatmul(&cfg(QDtype::Q8_0, QDtype::Q8_1, 2048)).unwrap();
        assert_eq!(d.impl_id, "qmatmul_q8_0_mmq_wave64_tile16_gfx906");
    }


    #[test]
    fn unknown_dtype_combo_returns_none() {
        let d = dispatch_qmatmul(&cfg(QDtype::F32, QDtype::F32, 1));
        assert!(d.is_none());
    }

    #[test]
    fn every_descriptor_cert_path_exists_on_disk() {
        let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        for table in [
            QMATMUL_GFX906,
            RMSNORM_GFX906,
            SWIGLU_GFX906,
            ROPE_GFX906,
            SOFTMAX_GFX906,
            ATTENTION_DECODE_GFX906,
            ATTENTION_PREFILL_GFX906,
            TOPK_GFX906,
            INDEXED_MOE_MMVQ_GFX906,
            MOE_COMBINE_GFX906,
            INDEXED_MOE_MMVQ_GATE_UP_GFX906,
            INDEXED_MOE_MMQ_GFX906,
            ROPE_NEOX_PARTIAL_GFX906,
            L2_NORM_GFX906,
            ADD_F16_GFX906,
            CAST_F16_F32_GFX906,
            CAST_F32_F16_GFX906,
            CAUSAL_CONV1D_GFX906,
            DENSE_GEMV_F32_F16_GFX906,
            GDN_ALPHA_BETA_GFX906,
            GDN_STATE_STEP_GFX906,
            QUANTIZE_F16_Q8_1_GFX906,
            RMSNORM_F32_GFX906,
            SCALE_F32_GFX906,
            SHARED_EXPERT_SCALE_GFX906,
            SILU_F32_GFX906,
            SPLIT_Q_GATE_GFX906,
            SWIGLU_F32_GFX906,
        ] {
            for d in table {
                let p = repo_root.join(d.cert_rel_path);
                assert!(
                    p.exists(),
                    "cert missing for {}: {}",
                    d.impl_id,
                    p.display()
                );
            }
        }
        for d in DIRECT_CALL_KERNELS_GFX906 {
            let p = repo_root.join(d.cert_rel_path);
            assert!(
                p.exists(),
                "cert missing for direct-call kernel {}: {}",
                d.impl_id,
                p.display()
            );
        }
        for d in BENCH_REFERENCE_KERNELS_GFX906 {
            let p = repo_root.join(d.cert_rel_path);
            assert!(
                p.exists(),
                "cert missing for bench-reference kernel {}: {}",
                d.impl_id,
                p.display()
            );
        }
    }

    /// class regression guard. Every `impl = "..."` row in
    /// `dispatch/hip/gfx906.toml` must be registered in either a
    /// `KernelDescriptor` table or `DIRECT_CALL_KERNELS_GFX906`. Otherwise a
    /// routing swap in the TOML won't actually change runtime behaviour
    /// (tile16 bug).
    #[test]
    fn dispatch_toml_roundtrip() {
        let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let toml_path = repo_root.join("dispatch/hip/gfx906.toml");
        let toml_src = std::fs::read_to_string(&toml_path)
            .unwrap_or_else(|e| panic!("read {}: {}", toml_path.display(), e));

        // Collect every `impl = "..."` value in the TOML (commented lines
        // starting with `#` are ignored).
        let mut toml_impls: Vec<String> = Vec::new();
        for line in toml_src.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with('#') {
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("impl") {
                let rest = rest.trim_start();
                if let Some(rest) = rest.strip_prefix('=') {
                    let rest = rest.trim();
                    if let Some(val) = rest.strip_prefix('"') {
                        if let Some(end) = val.find('"') {
                            toml_impls.push(val[..end].to_string());
                        }
                    }
                }
            }
        }
        assert!(
            !toml_impls.is_empty(),
            "no `impl = \"...\"` rows parsed from {}",
            toml_path.display()
        );

        // Build the registered-impl set from every KernelDescriptor table
        // plus DIRECT_CALL_KERNELS_GFX906.
        let mut registered: std::collections::HashSet<&'static str> =
            std::collections::HashSet::new();
        for table in [
            QMATMUL_GFX906,
            RMSNORM_GFX906,
            SWIGLU_GFX906,
            ROPE_GFX906,
            SOFTMAX_GFX906,
            ATTENTION_DECODE_GFX906,
            ATTENTION_PREFILL_GFX906,
            TOPK_GFX906,
            INDEXED_MOE_MMVQ_GFX906,
            MOE_COMBINE_GFX906,
            INDEXED_MOE_MMVQ_GATE_UP_GFX906,
            INDEXED_MOE_MMQ_GFX906,
            ROPE_NEOX_PARTIAL_GFX906,
            L2_NORM_GFX906,
            ADD_F16_GFX906,
            CAST_F16_F32_GFX906,
            CAST_F32_F16_GFX906,
            CAUSAL_CONV1D_GFX906,
            DENSE_GEMV_F32_F16_GFX906,
            GDN_ALPHA_BETA_GFX906,
            GDN_STATE_STEP_GFX906,
            QUANTIZE_F16_Q8_1_GFX906,
            RMSNORM_F32_GFX906,
            SCALE_F32_GFX906,
            SHARED_EXPERT_SCALE_GFX906,
            SILU_F32_GFX906,
            SPLIT_Q_GATE_GFX906,
            SWIGLU_F32_GFX906,
        ] {
            for d in table {
                registered.insert(d.impl_id);
            }
        }
        for d in DIRECT_CALL_KERNELS_GFX906 {
            registered.insert(d.impl_id);
        }
        for d in BENCH_REFERENCE_KERNELS_GFX906 {
            registered.insert(d.impl_id);
        }

        let unregistered: Vec<&str> = toml_impls
            .iter()
            .filter(|id| !registered.contains(id.as_str()))
            .map(String::as_str)
            .collect();
        assert!(
            unregistered.is_empty(),
            "dispatch/hip/gfx906.toml references impls not registered in \
             backend-hip (KernelDescriptor table or DIRECT_CALL_KERNELS_GFX906): {unregistered:?}. \
             Either add a KernelDescriptor (shape-dispatch) or a DirectCallKernel \
             (single-impl per dtype)."
        );
    }

    /// Reverse roundtrip: every cert JSON under `certs/hip/gfx906/` must
    /// be registered in one of the three catalogs (shape-dispatch,
    /// direct-call, bench-reference). Catches the inverse drift: a cert
    /// lingering after its kernel was removed.
    #[test]
    fn every_cert_on_disk_is_registered() {
        let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let cert_dir = repo_root.join("certs/hip/gfx906");
        let entries = match std::fs::read_dir(&cert_dir) {
            Ok(e) => e,
            Err(e) => panic!("read_dir {}: {}", cert_dir.display(), e),
        };

        let mut registered: std::collections::HashSet<&'static str> =
            std::collections::HashSet::new();
        for table in [
            QMATMUL_GFX906,
            RMSNORM_GFX906,
            SWIGLU_GFX906,
            ROPE_GFX906,
            SOFTMAX_GFX906,
            ATTENTION_DECODE_GFX906,
            ATTENTION_PREFILL_GFX906,
            TOPK_GFX906,
            INDEXED_MOE_MMVQ_GFX906,
            MOE_COMBINE_GFX906,
            INDEXED_MOE_MMVQ_GATE_UP_GFX906,
            INDEXED_MOE_MMQ_GFX906,
            ROPE_NEOX_PARTIAL_GFX906,
            L2_NORM_GFX906,
            ADD_F16_GFX906,
            CAST_F16_F32_GFX906,
            CAST_F32_F16_GFX906,
            CAUSAL_CONV1D_GFX906,
            DENSE_GEMV_F32_F16_GFX906,
            GDN_ALPHA_BETA_GFX906,
            GDN_STATE_STEP_GFX906,
            QUANTIZE_F16_Q8_1_GFX906,
            RMSNORM_F32_GFX906,
            SCALE_F32_GFX906,
            SHARED_EXPERT_SCALE_GFX906,
            SILU_F32_GFX906,
            SPLIT_Q_GATE_GFX906,
            SWIGLU_F32_GFX906,
        ] {
            for d in table {
                registered.insert(d.impl_id);
            }
        }
        for d in DIRECT_CALL_KERNELS_GFX906 {
            registered.insert(d.impl_id);
        }
        for d in BENCH_REFERENCE_KERNELS_GFX906 {
            registered.insert(d.impl_id);
        }

        let mut orphans: Vec<String> = Vec::new();
        for entry in entries {
            let entry = entry.unwrap();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let Some(stem) = name.strip_suffix(".json") else {
                continue;
            };
            if !registered.contains(stem) {
                orphans.push(stem.to_string());
            }
        }
        orphans.sort();
        assert!(
            orphans.is_empty(),
            "cert files under certs/hip/gfx906/ without a matching registration: \
             {orphans:?}. Either delete the cert, or add a row to KernelDescriptor, \
             DIRECT_CALL_KERNELS_GFX906, or BENCH_REFERENCE_KERNELS_GFX906."
        );
    }

    #[test]
    fn rmsnorm_routes_by_output_dtype() {
        let f16 = dispatch_rmsnorm(QDtype::F16, QDtype::F16, 1).unwrap();
        assert_eq!(f16.impl_id, "rmsnorm_f16_gfx906");
        let q8 = dispatch_rmsnorm(QDtype::F16, QDtype::Q8_1, 1).unwrap();
        assert_eq!(q8.impl_id, "rmsnorm_q8_1_fused_gfx906");
    }

    #[test]
    fn swiglu_routes_f16() {
        let s = dispatch_swiglu(QDtype::F16, 128).unwrap();
        assert_eq!(s.impl_id, "swiglu_f16_gfx906");
    }
}
