//! MMVQ correctness sweep — one HIP device, one dtype at a time.
//! Mirrors the ad-hoc `crates/backend-hip/tests/mmvq_q*.rs` tests but runs
//! from the library / CLI and writes a `Cert` JSON under `certs/hip/gfx906/`.
//! grid per roadmap:
//! M ∈ {1, 8, 16, 128, 512} (row counts — we run M rows per shape)
//! K ∈ {2048, 5120, 15360, 128256}
//! N — not a kernel input for single-row MMVQ (one block per row).
//! Tolerance: `1e-2 * sqrt(K/128)` — captures F32 accumulation noise + Q8_1
//! activation quant round-trip. See `project_v1_3_q8_0_landed.md` for the
//! derivation.

#![cfg(any(feature = "hip", feature = "cuda"))]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "sweep harness — every unsafe block is a kernel launch or a memcpy_async \
              over buffers allocated locally in the same function and freed before \
              return; invariant is uniform across all sites."
)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, LaunchCfg, Stream};
use flambeau_quant::{
    dequantize_into, BlockIq1M, BlockIq1S, BlockIq2S, BlockIq2Xs, BlockIq2Xxs, BlockIq3S,
    BlockIq3Xxs, BlockIq4Nl, BlockIq4Xs, BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1,
    BlockQ5K, BlockQ5_0, BlockQ5_1, BlockQ6K, BlockQ8K, BlockQ8_0, BlockQ8_1, GgmlDType, QK8_0,
    QK_K,
};
use half::f16;

use crate::backend::{BenchBackend, LaunchArgs};
use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig_for, seeded_f32_range};

const QK8: usize = QK8_0;

/// `(weights_f32, activation_f32, gpu_result, cpu_reference)` — the four F32
/// buffers an MMVQ/MMQ correctness oracle compares.
type RefQuad = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Dtype tag selecting which weight block layout the sweep drives. We only
/// define the four dtypes here; `GgmlDType::from_wire` covers the rest.
/// `(weight dtype, kernel variant)` — selects one impl to certify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Q8_0,
    /// Templated 256-thread single-row Q4_0 DP4A
    /// (`mmvq_dp4a_row<Q4_0, 256>`). Same source on HIP + CUDA; the neutral
    /// impl_id certs the templated kernel on either backend.
    Q4_0,
    /// Thin-block 128-thread single-row Q4_0 DP4A
    /// (`mmvq_dp4a_row<Q4_0, 128>`) — sm_86 m=1 decode autotune candidate vs the
    /// 256-thread base. Distinct stem from the gfx906-only hand kernel.
    Q4_0T128,
    /// t128 + VDR=2 Q4_0 DP4A (`mmvq_dp4a_vdr_row<Q4_0, 128, 2>`) — 128
    /// threads, 2 int32 lanes/thread chaining 4 dp4a for tighter ILP.
    Q4_0T128VDR2,
    /// Templated single-row Q4_1 DP4A with the neutral cross-backend impl_id
    /// (the base `mmvq_q4_1` kernel; the suffixed `Q4_1` variant keeps the
    /// older gfx906 dispatch context).
    Q4_1Dp4a,
    /// Templated single-row Q5_0 DP4A (`mmvq_dp4a_row<Q5_0, 256>`).
    Q5_0,
    /// Templated single-row Q5_1 DP4A (`mmvq_dp4a_row<Q5_1, 256>`).
    Q5_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    /// r2 multi-row variant of Q2_K.
    Q2KR2,
    /// DP4A 256-thread Q2_K MMVQ (port of llama.cpp's
    /// `vec_dot_q2_K_q8_1_impl_mmvq`). Two DP4A calls per (iqs, i):
    /// one for the 2-bit packed values, one for the broadcast-min
    /// term (m × Σu via dp4a(m_packed, u, 0)). Same 64-threads-per-
    /// super-block / BLOCKS_PER_ITER=4 layout as Q3KDp4a.
    Q2KDp4a,
    /// r2 multi-row DP4A Q2_K MMVQ (Lever 1 sibling of Q3KR2Dp4a —
    /// same wave64 / 2 rows-per-block layout).
    Q2KR2Dp4a,
    /// r2 multi-row variant of Q3_K.
    Q3KR2,
    /// DP4A 256-thread Q3_K MMVQ (port of llama.cpp's
    /// `vec_dot_q3_K_q8_1_impl_mmvq`). 64 threads per super-block,
    /// 4 super-blocks per iter. Splits the 2-bit + 1-bit-hmask decode
    /// into two DP4A calls (one for low bits, one for hmask bit) to
    /// avoid the int32 byte-borrow corruption that __vsubss4 would
    /// sidestep on NVIDIA.
    Q3KDp4a,
    Q3KDp4aT128Vdr2,
    /// r2 multi-row DP4A Q3_K MMVQ (Lever 1 — refactor of
    /// `Q3KDp4a` to wave64 single-warp / 2 rows per block, half-warp
    /// reduce, no LDS rendezvous, activation reuse across rows).
    Q3KR2Dp4a,
    /// r2 multi-row variant of Q4_K (2 output rows per wave64).
    Q4KR2,
    /// DP4A r2 multi-row Q4_K MMVQ (port of llama.cpp's
    /// `vec_dot_q4_K_q8_1_impl_vmmq`). Same shape as Q4KR2 but with
    /// `__builtin_amdgcn_sdot4` SIMD int8 dot products instead of
    /// per-element FP32 multiplies. Mirrors the MoE
    /// `indexed_moe_mmvq_q4_k_r2_dp4a` body.
    Q4KR2Dp4a,
    /// Single-row dp4a, 128 threads, llama.cpp GCN layout (VDR 2 / VDR 4).
    Q4KDp4aT128Vdr2,
    Q5KDp4aT128Vdr2,
    Iq4XsDp4aT128Vdr4,
    /// r2 multi-row variant of Q5_K.
    Q5KR2,
    /// DP4A 256-thread Q5_K MMVQ (port of llama.cpp's
    /// `vec_dot_q5_K_q8_1_impl_vmmq`). Replaces per-element FP32
    /// multiplies with `__builtin_amdgcn_sdot4` SIMD int8 dot products.
    Q5KDp4a,
    /// Wave64 r2 dp4a Q5_K MMVQ — dense analog of the MoE-side M-c
    /// kernel. 2 output rows per block, 32 lanes per row, half-warp
    /// DPP reduce. Cuts launch count in half vs the 256-thread
    /// single-row variant on the large dense projections (attn_qkv,
    /// attn_gate, etc.).
    Q5KR2Dp4a,
    /// r4 multi-row variant of Q6_K (4 output rows per wave64).
    Q6KR4,
    /// DP4A single-row Q6_K variant. Cooperative 64 lanes across
    /// 2 super-blocks, inner DP4A via `__builtin_amdgcn_sdot4`. Runtime
    /// intercepts the base Q6_K row when this kernel is loaded.
    Q6KDP4A,
    /// Q4_1 legacy quant with per-block min offset. Single row per
    /// 256-thread block, DP4A inner loop.
    Q4_1,
    /// r2 multi-row Q4_1 (64 threads, 2 rows/block, half-warp
    /// DPP reduce). NULL — lost DP4A parallelism. Kept for regression.
    Q4_1R2,
    /// DP4A r2 multi-row Q4_1 (256 threads, 2 rows/block).
    /// Halves grid + shares Y across rows; preserves DP4A.
    /// NULL — register pressure regressed.
    Q4_1R2DP4A,
    /// thin-block 128-thread single-row Q4_1 DP4A.
    Q4_1T128,
    /// Thin-block 128-thread single-row Q8_0 DP4A — analog of Q4_1T128
    /// for Q8_0 weights — targets the gfx906 latency-bound regime where Q8_0
    /// MMVQ is the dominant kernel on Coder-30B-Q8_0 / 27B-Q8_0 decode.
    Q8_0T128,
    /// t128 + VDR=2 combined.
    Q8_0T128VDR2,
    /// r2 multi-row Q8_0 (`mmvq_dp4a_nrow_row<Q8_0, 64, 2>`): one 64-thread
    /// block computes 2 output rows (32 lanes each), sharing the Q8_1
    /// activation reads. Autotune config variant vs the single-row dp4a.
    Q8_0R2,
    /// r4 multi-row Q8_0 (`mmvq_dp4a_nrow_row<Q8_0, 64, 4>`): 4 output rows
    /// per 64-thread block (16 lanes each), quarter-warp reduce.
    Q8_0R4,
    /// Templated 256-thread single-row Q8_0 DP4A (`mmvq_dp4a_row<Q8_0, 256>`).
    /// The cross-backend pilot kernel — certified on both gfx906 and sm_86.
    Q8_0Dp4a,
    /// native IQ4_NL single-row MMVQ.
    Iq4Nl,
    /// native IQ4_NL r2 multi-row MMVQ.
    Iq4NlR2,
    /// DP4A IQ4_NL MMVQ (port of llama.cpp's
    /// `vec_dot_iq4_nl_q8_1`). 256 threads/block, single row, mirrors
    /// `mmvq_q4_0` launch shape (4 lanes per 32-elem block, 64 blocks
    /// per outer iter). Uses the same `__builtin_amdgcn_perm` LUT
    /// helper as IQ4_XS dp4a.
    Iq4NlDp4a,
    Iq4NlDp4aT128Vdr2,
    /// native IQ4_XS single-row MMVQ.
    Iq4Xs,
    /// native IQ4_XS r2 multi-row MMVQ.
    Iq4XsR2,
    /// DP4A IQ4_XS MMVQ (port of llama.cpp's
    /// `vec_dot_iq4_xs_q8_1`). 256 threads, single row, 32 threads
    /// per super-block. Uses `__builtin_amdgcn_perm` for 4-way LUT
    /// lookup into `kvalues_iq4nl`, then dp4a inner.
    Iq4XsDp4a,
    /// native IQ3_XXS single-row MMVQ.
    Iq3Xxs,
    /// native IQ3_XXS r2 multi-row MMVQ.
    Iq3XxsR2,
    /// DP4A IQ3_XXS MMVQ (port of llama.cpp's
    /// `vec_dot_iq3_xxs_q8_1`). Uses IQ3XXS_GRID + unpack_ksigns
    /// (popcount-parity 8th-bit extension) + the apply_signs helper
    /// from IQ3_S.
    Iq3XxsDp4a,
    /// native IQ3_S single-row MMVQ.
    Iq3S,
    /// native IQ3_S r2 multi-row MMVQ.
    Iq3SR2,
    /// DP4A IQ3_S MMVQ (port of llama.cpp's
    /// `vec_dot_iq3_s_q8_1`). First codebook kernel: looks up 9-bit
    /// indices into `IQ3S_GRID` for 4-i8-packed magnitudes, applies
    /// per-byte sign from packed bits (manual __vsub4 emulation on
    /// gfx906), dp4a vs Q8_1. 256 threads/block, single row, 32 threads
    /// per super-block × 8 super-blocks per iter.
    Iq3SDp4a,
    Iq3SDp4aT128Vdr2,
    /// native IQ2_XXS single-row MMVQ.
    Iq2Xxs,
    Iq2XxsR2,
    /// DP4A IQ2_XXS MMVQ (port of llama.cpp's
    /// `vec_dot_iq2_xxs_q8_1`). Per sub-block 8 qs bytes split into 4
    /// codebook indices + 4 × 7-bit signs + 4-bit scale (top of aux32).
    /// One scale per sub-block (no ls0/ls1 split).
    Iq2XxsDp4a,
    Iq2Xs,
    Iq2XsR2,
    /// DP4A IQ2_XS MMVQ (port of llama.cpp's
    /// `vec_dot_iq2_xs_q8_1`). Each qs entry is uint16: low 9 bits
    /// index IQ2XS_GRID, high 7 bits are signs (popcount-parity for
    /// 8th bit). Reuses apply_signs + unpack_ksigns helpers.
    Iq2XsDp4a,
    Iq2S,
    Iq2SR2,
    /// DP4A IQ2_S MMVQ (port of llama.cpp's
    /// `vec_dot_iq2_s_q8_1`). IQ2S_GRID is uint64[1024] (8 i8 per
    /// entry); per (iqs, l_grp) lane: one grid lookup, two dp4as,
    /// two sub-block scales (ls0/ls1) over l_grp split.
    Iq2SDp4a,
    Iq1S,
    Iq1SR2,
    /// DP4A IQ1_S MMVQ (port of llama.cpp's
    /// `vec_dot_iq1_s_q8_1`). No sign mask; codebook stores signed i8
    /// magnitudes directly in IQ1S_GRID (uint64 of 8 packed i8).
    /// Per-sub-block delta offset (±IQ1_DELTA) introduces a bias term
    /// computed via dp4a(0x01010101, u, 0). VDR=1 (no x2 step on iqs).
    Iq1SDp4a,
    Iq1M,
    Iq1MR2,
    /// DP4A IQ1_M MMVQ (port of llama.cpp's
    /// `vec_dot_iq1_m_q8_1`). Closes the IQ family. Same IQ1S_GRID +
    /// delta pattern as IQ1_S but with per-half-sub-block scales
    /// (sc0/sc1) and per-l_grp delta sign from qh nibbles. d itself
    /// is reassembled from 4 nibbles spread across the 4 u16 scales
    /// words via iq1m_reassemble_d.
    Iq1MDp4a,
}

impl Dtype {
    pub fn name(self) -> &'static str {
        match self {
            Dtype::Q8_0 | Dtype::Q8_0T128 | Dtype::Q8_0T128VDR2 | Dtype::Q8_0R2 | Dtype::Q8_0R4 | Dtype::Q8_0Dp4a => "Q8_0",
            Dtype::Q4_0 | Dtype::Q4_0T128 | Dtype::Q4_0T128VDR2 => "Q4_0",
            Dtype::Q5_0 => "Q5_0",
            Dtype::Q5_1 => "Q5_1",
            Dtype::Q2K | Dtype::Q2KR2 | Dtype::Q2KDp4a | Dtype::Q2KR2Dp4a => "Q2_K",
            Dtype::Q3K | Dtype::Q3KR2 | Dtype::Q3KDp4a | Dtype::Q3KDp4aT128Vdr2 | Dtype::Q3KR2Dp4a => "Q3_K",
            Dtype::Q4K | Dtype::Q4KR2 | Dtype::Q4KR2Dp4a | Dtype::Q4KDp4aT128Vdr2 => "Q4_K",
            Dtype::Q5K | Dtype::Q5KR2 | Dtype::Q5KDp4a | Dtype::Q5KR2Dp4a | Dtype::Q5KDp4aT128Vdr2 => "Q5_K",
            Dtype::Q6K | Dtype::Q6KR4 | Dtype::Q6KDP4A => "Q6_K",
            Dtype::Q8K => "Q8_K",
            Dtype::Q4_1 | Dtype::Q4_1Dp4a | Dtype::Q4_1R2 | Dtype::Q4_1R2DP4A | Dtype::Q4_1T128 => {
                "Q4_1"
            }
            Dtype::Iq4Nl | Dtype::Iq4NlR2 | Dtype::Iq4NlDp4a | Dtype::Iq4NlDp4aT128Vdr2 => "IQ4_NL",
            Dtype::Iq4Xs | Dtype::Iq4XsR2 | Dtype::Iq4XsDp4a | Dtype::Iq4XsDp4aT128Vdr4 => "IQ4_XS",
            Dtype::Iq3Xxs | Dtype::Iq3XxsR2 | Dtype::Iq3XxsDp4a => "IQ3_XXS",
            Dtype::Iq3S | Dtype::Iq3SR2 | Dtype::Iq3SDp4a | Dtype::Iq3SDp4aT128Vdr2 => "IQ3_S",
            Dtype::Iq2Xxs | Dtype::Iq2XxsR2 | Dtype::Iq2XxsDp4a => "IQ2_XXS",
            Dtype::Iq2Xs | Dtype::Iq2XsR2 | Dtype::Iq2XsDp4a => "IQ2_XS",
            Dtype::Iq2S | Dtype::Iq2SR2 | Dtype::Iq2SDp4a => "IQ2_S",
            Dtype::Iq1S | Dtype::Iq1SR2 | Dtype::Iq1SDp4a => "IQ1_S",
            Dtype::Iq1M | Dtype::Iq1MR2 | Dtype::Iq1MDp4a => "IQ1_M",
        }
    }

    pub fn ggml(self) -> GgmlDType {
        match self {
            Dtype::Q8_0 | Dtype::Q8_0T128 | Dtype::Q8_0T128VDR2 | Dtype::Q8_0R2 | Dtype::Q8_0R4 | Dtype::Q8_0Dp4a => {
                GgmlDType::Q8_0
            }
            Dtype::Q4_0 | Dtype::Q4_0T128 | Dtype::Q4_0T128VDR2 => GgmlDType::Q4_0,
            Dtype::Q5_0 => GgmlDType::Q5_0,
            Dtype::Q5_1 => GgmlDType::Q5_1,
            Dtype::Q2K | Dtype::Q2KR2 | Dtype::Q2KDp4a | Dtype::Q2KR2Dp4a => GgmlDType::Q2K,
            Dtype::Q3K | Dtype::Q3KR2 | Dtype::Q3KDp4a | Dtype::Q3KDp4aT128Vdr2 | Dtype::Q3KR2Dp4a => GgmlDType::Q3K,
            Dtype::Q4K | Dtype::Q4KR2 | Dtype::Q4KR2Dp4a | Dtype::Q4KDp4aT128Vdr2 => GgmlDType::Q4K,
            Dtype::Q5K | Dtype::Q5KR2 | Dtype::Q5KDp4a | Dtype::Q5KR2Dp4a | Dtype::Q5KDp4aT128Vdr2 => GgmlDType::Q5K,
            Dtype::Q6K | Dtype::Q6KR4 | Dtype::Q6KDP4A => GgmlDType::Q6K,
            Dtype::Q8K => GgmlDType::Q8K,
            Dtype::Q4_1 | Dtype::Q4_1Dp4a | Dtype::Q4_1R2 | Dtype::Q4_1R2DP4A | Dtype::Q4_1T128 => {
                GgmlDType::Q4_1
            }
            Dtype::Iq4Nl | Dtype::Iq4NlR2 | Dtype::Iq4NlDp4a | Dtype::Iq4NlDp4aT128Vdr2 => GgmlDType::Iq4Nl,
            Dtype::Iq4Xs | Dtype::Iq4XsR2 | Dtype::Iq4XsDp4a | Dtype::Iq4XsDp4aT128Vdr4 => GgmlDType::Iq4Xs,
            Dtype::Iq3Xxs | Dtype::Iq3XxsR2 | Dtype::Iq3XxsDp4a => GgmlDType::Iq3Xxs,
            Dtype::Iq3S | Dtype::Iq3SR2 | Dtype::Iq3SDp4a | Dtype::Iq3SDp4aT128Vdr2 => GgmlDType::Iq3S,
            Dtype::Iq2Xxs | Dtype::Iq2XxsR2 | Dtype::Iq2XxsDp4a => GgmlDType::Iq2Xxs,
            Dtype::Iq2Xs | Dtype::Iq2XsR2 | Dtype::Iq2XsDp4a => GgmlDType::Iq2Xs,
            Dtype::Iq2S | Dtype::Iq2SR2 | Dtype::Iq2SDp4a => GgmlDType::Iq2S,
            Dtype::Iq1S | Dtype::Iq1SR2 | Dtype::Iq1SDp4a => GgmlDType::Iq1S,
            Dtype::Iq1M | Dtype::Iq1MR2 | Dtype::Iq1MDp4a => GgmlDType::Iq1M,
        }
    }

    fn impl_id(self) -> &'static str {
        match self {
            Dtype::Q8_0 => "qmatmul_q8_0_mmvq_single_row_gfx906",
            Dtype::Q8_0Dp4a => "qmatmul_q8_0_mmvq_dp4a",
            Dtype::Q4_0 => "qmatmul_q4_0_mmvq_dp4a",
            Dtype::Q4_0T128 => "qmatmul_q4_0_mmvq_t128_dp4a",
            Dtype::Q4_0T128VDR2 => "qmatmul_q4_0_mmvq_t128_vdr2",
            Dtype::Q4_1Dp4a => "qmatmul_q4_1_mmvq_dp4a",
            Dtype::Q5_0 => "qmatmul_q5_0_mmvq_dp4a",
            Dtype::Q5_1 => "qmatmul_q5_1_mmvq_dp4a",
            Dtype::Q2K => "qmatmul_q2_K_mmvq_single_row_gfx906",
            Dtype::Q2KR2 => "qmatmul_q2_K_mmvq_nw1_r2_gfx906",
            Dtype::Q2KDp4a => "qmatmul_q2_k_mmvq_dp4a",
            Dtype::Q2KR2Dp4a => "qmatmul_q2_K_mmvq_r2_dp4a_gfx906",
            Dtype::Q3K => "qmatmul_q3_K_mmvq_single_row_gfx906",
            Dtype::Q3KR2 => "qmatmul_q3_K_mmvq_nw1_r2_gfx906",
            Dtype::Q3KDp4a => "qmatmul_q3_k_mmvq_dp4a",
            Dtype::Q3KDp4aT128Vdr2 => "qmatmul_q3_k_mmvq_dp4a_t128_vdr2",
            Dtype::Q3KR2Dp4a => "qmatmul_q3_K_mmvq_r2_dp4a_gfx906",
            Dtype::Q4K => "qmatmul_q4_K_mmvq_single_row_gfx906",
            Dtype::Q5K => "qmatmul_q5_K_mmvq_single_row_gfx906",
            Dtype::Q6K => "qmatmul_q6_K_mmvq_single_row_gfx906",
            Dtype::Q8K => "qmatmul_q8_k_mmvq_single_row",
            Dtype::Q4KR2 => "qmatmul_q4_K_mmvq_nw1_r2_gfx906",
            Dtype::Q4KR2Dp4a => "qmatmul_q4_k_mmvq_dp4a",
            Dtype::Iq4XsDp4aT128Vdr4 => "qmatmul_iq4_xs_mmvq_dp4a_t128_vdr4",
            Dtype::Q5KDp4aT128Vdr2 => "qmatmul_q5_k_mmvq_dp4a_t128_vdr2",
            Dtype::Q4KDp4aT128Vdr2 => "qmatmul_q4_k_mmvq_dp4a_t128_vdr2",
            Dtype::Q5KR2 => "qmatmul_q5_K_mmvq_nw1_r2_gfx906",
            Dtype::Q5KDp4a => "qmatmul_q5_k_mmvq_dp4a",
            Dtype::Q5KR2Dp4a => "qmatmul_q5_K_mmvq_r2_dp4a_gfx906",
            Dtype::Q6KR4 => "qmatmul_q6_K_mmvq_nw1_r4_gfx906",
            Dtype::Q6KDP4A => "qmatmul_q6_k_mmvq_dp4a",
            Dtype::Q4_1 => "qmatmul_q4_1_mmvq_dp4a_gfx906",
            Dtype::Q4_1R2 => "qmatmul_q4_1_mmvq_nw1_r2_gfx906",
            Dtype::Q4_1R2DP4A => "qmatmul_q4_1_mmvq_r2_dp4a_gfx906",
            Dtype::Q4_1T128 => "qmatmul_q4_1_mmvq_t128_gfx906",
            Dtype::Q8_0T128 => "qmatmul_q8_0_mmvq_t128",
            Dtype::Q8_0T128VDR2 => "qmatmul_q8_0_mmvq_t128_vdr2",
            Dtype::Q8_0R2 => "qmatmul_q8_0_mmvq_r2",
            Dtype::Q8_0R4 => "qmatmul_q8_0_mmvq_r4",
            Dtype::Iq4Nl => "qmatmul_iq4_nl_mmvq_single_row_gfx906",
            Dtype::Iq4NlR2 => "qmatmul_iq4_nl_mmvq_nw1_r2_gfx906",
            Dtype::Iq4NlDp4a => "qmatmul_iq4_nl_mmvq_dp4a",
            Dtype::Iq4NlDp4aT128Vdr2 => "qmatmul_iq4_nl_mmvq_dp4a_t128_vdr2",
            Dtype::Iq4Xs => "qmatmul_iq4_xs_mmvq_single_row_gfx906",
            Dtype::Iq4XsR2 => "qmatmul_iq4_xs_mmvq_nw1_r2_gfx906",
            Dtype::Iq4XsDp4a => "qmatmul_iq4_xs_mmvq_dp4a",
            Dtype::Iq3Xxs => "qmatmul_iq3_xxs_mmvq_single_row_gfx906",
            Dtype::Iq3XxsR2 => "qmatmul_iq3_xxs_mmvq_nw1_r2_gfx906",
            Dtype::Iq3XxsDp4a => "qmatmul_iq3_xxs_mmvq_dp4a",
            Dtype::Iq3S => "qmatmul_iq3_s_mmvq_single_row_gfx906",
            Dtype::Iq3SR2 => "qmatmul_iq3_s_mmvq_nw1_r2_gfx906",
            Dtype::Iq3SDp4a => "qmatmul_iq3_s_mmvq_dp4a",
            Dtype::Iq3SDp4aT128Vdr2 => "qmatmul_iq3_s_mmvq_dp4a_t128_vdr2",
            Dtype::Iq2Xxs => "qmatmul_iq2_xxs_mmvq_single_row_gfx906",
            Dtype::Iq2XxsR2 => "qmatmul_iq2_xxs_mmvq_nw1_r2_gfx906",
            Dtype::Iq2XxsDp4a => "qmatmul_iq2_xxs_mmvq_dp4a",
            Dtype::Iq2Xs => "qmatmul_iq2_xs_mmvq_single_row_gfx906",
            Dtype::Iq2XsR2 => "qmatmul_iq2_xs_mmvq_nw1_r2_gfx906",
            Dtype::Iq2XsDp4a => "qmatmul_iq2_xs_mmvq_dp4a",
            Dtype::Iq2S => "qmatmul_iq2_s_mmvq_single_row_gfx906",
            Dtype::Iq2SR2 => "qmatmul_iq2_s_mmvq_nw1_r2_gfx906",
            Dtype::Iq2SDp4a => "qmatmul_iq2_s_mmvq_dp4a",
            Dtype::Iq1S => "qmatmul_iq1_s_mmvq_single_row_gfx906",
            Dtype::Iq1SR2 => "qmatmul_iq1_s_mmvq_nw1_r2_gfx906",
            Dtype::Iq1SDp4a => "qmatmul_iq1_s_mmvq_dp4a",
            Dtype::Iq1M => "qmatmul_iq1_m_mmvq_single_row_gfx906",
            Dtype::Iq1MR2 => "qmatmul_iq1_m_mmvq_nw1_r2_gfx906",
            Dtype::Iq1MDp4a => "qmatmul_iq1_m_mmvq_dp4a",
        }
    }

    fn kernel_stem(self) -> &'static str {
        match self {
            Dtype::Q8_0 => "mmvq_q8_0",
            Dtype::Q8_0Dp4a => "mmvq_q8_0_dp4a",
            Dtype::Q4_0 => "mmvq_q4_0",
            Dtype::Q4_0T128 => "mmvq_q4_0_t128_dp4a",
            Dtype::Q4_0T128VDR2 => "mmvq_q4_0_t128_vdr2",
            Dtype::Q4_1Dp4a => "mmvq_q4_1",
            Dtype::Q5_0 => "mmvq_q5_0",
            Dtype::Q5_1 => "mmvq_q5_1",
            Dtype::Q8_0T128 => "mmvq_q8_0_t128",
            Dtype::Q8_0T128VDR2 => "mmvq_q8_0_t128_vdr2",
            Dtype::Q8_0R2 => "mmvq_q8_0_r2",
            Dtype::Q8_0R4 => "mmvq_q8_0_r4",
            Dtype::Q2K => "mmvq_q2_k",
            Dtype::Q2KR2 => "mmvq_q2_k_r2",
            Dtype::Q2KDp4a => "mmvq_q2_k_dp4a",
            Dtype::Q2KR2Dp4a => "mmvq_q2_k_r2_dp4a",
            Dtype::Q3K => "mmvq_q3_k",
            Dtype::Q3KR2 => "mmvq_q3_k_r2",
            Dtype::Q3KDp4a => "mmvq_q3_k_dp4a",
            Dtype::Q3KDp4aT128Vdr2 => "mmvq_q3_k_dp4a_t128_vdr2",
            Dtype::Q3KR2Dp4a => "mmvq_q3_k_r2_dp4a",
            Dtype::Q4K => "mmvq_q4_k",
            Dtype::Q5K => "mmvq_q5_k",
            Dtype::Q6K => "mmvq_q6_k",
            Dtype::Q8K => "mmvq_q8_k",
            Dtype::Q4KR2 => "mmvq_q4_k_r2",
            Dtype::Q4KR2Dp4a => "mmvq_q4_k_r2_dp4a",
            Dtype::Iq4XsDp4aT128Vdr4 => "mmvq_iq4_xs_dp4a_t128_vdr4",
            Dtype::Q5KDp4aT128Vdr2 => "mmvq_q5_k_dp4a_t128_vdr2",
            Dtype::Q4KDp4aT128Vdr2 => "mmvq_q4_k_dp4a_t128_vdr2",
            Dtype::Q5KR2 => "mmvq_q5_k_r2",
            Dtype::Q5KDp4a => "mmvq_q5_k_dp4a",
            Dtype::Q5KR2Dp4a => "mmvq_q5_k_r2_dp4a",
            Dtype::Q6KR4 => "mmvq_q6_k_r4",
            Dtype::Q6KDP4A => "mmvq_q6_k_dp4a",
            Dtype::Q4_1 => "mmvq_q4_1",
            Dtype::Q4_1R2 => "mmvq_q4_1_r2",
            Dtype::Q4_1R2DP4A => "mmvq_q4_1_r2_dp4a",
            Dtype::Q4_1T128 => "mmvq_q4_1_t128",
            Dtype::Iq4Nl => "mmvq_iq4_nl",
            Dtype::Iq4NlR2 => "mmvq_iq4_nl_r2",
            Dtype::Iq4NlDp4a => "mmvq_iq4_nl_dp4a",
            Dtype::Iq4NlDp4aT128Vdr2 => "mmvq_iq4_nl_dp4a_t128_vdr2",
            Dtype::Iq4Xs => "mmvq_iq4_xs",
            Dtype::Iq4XsR2 => "mmvq_iq4_xs_r2",
            Dtype::Iq4XsDp4a => "mmvq_iq4_xs_dp4a",
            Dtype::Iq3Xxs => "mmvq_iq3_xxs",
            Dtype::Iq3XxsR2 => "mmvq_iq3_xxs_r2",
            Dtype::Iq3XxsDp4a => "mmvq_iq3_xxs_dp4a",
            Dtype::Iq3S => "mmvq_iq3_s",
            Dtype::Iq3SR2 => "mmvq_iq3_s_r2",
            Dtype::Iq3SDp4a => "mmvq_iq3_s_dp4a",
            Dtype::Iq3SDp4aT128Vdr2 => "mmvq_iq3_s_dp4a_t128_vdr2",
            Dtype::Iq2Xxs => "mmvq_iq2_xxs",
            Dtype::Iq2XxsR2 => "mmvq_iq2_xxs_r2",
            Dtype::Iq2XxsDp4a => "mmvq_iq2_xxs_dp4a",
            Dtype::Iq2Xs => "mmvq_iq2_xs",
            Dtype::Iq2XsR2 => "mmvq_iq2_xs_r2",
            Dtype::Iq2XsDp4a => "mmvq_iq2_xs_dp4a",
            Dtype::Iq2S => "mmvq_iq2_s",
            Dtype::Iq2SR2 => "mmvq_iq2_s_r2",
            Dtype::Iq2SDp4a => "mmvq_iq2_s_dp4a",
            Dtype::Iq1S => "mmvq_iq1_s",
            Dtype::Iq1SR2 => "mmvq_iq1_s_r2",
            Dtype::Iq1SDp4a => "mmvq_iq1_s_dp4a",
            Dtype::Iq1M => "mmvq_iq1_m",
            Dtype::Iq1MR2 => "mmvq_iq1_m_r2",
            Dtype::Iq1MDp4a => "mmvq_iq1_m_dp4a",
        }
    }

    fn kernel_entry(self) -> &'static str {
        match self {
            Dtype::Q8_0 => "flambeau_mmvq_q8_0_q8_1",
            Dtype::Q8_0Dp4a => "flambeau_mmvq_q8_0_dp4a_q8_1",
            Dtype::Q4_0 => "flambeau_mmvq_q4_0_q8_1",
            Dtype::Q4_0T128 => "flambeau_mmvq_q4_0_t128_dp4a_q8_1",
            Dtype::Q4_0T128VDR2 => "flambeau_mmvq_q4_0_t128_vdr2_q8_1",
            Dtype::Q4_1Dp4a => "flambeau_mmvq_q4_1_q8_1",
            Dtype::Q5_0 => "flambeau_mmvq_q5_0_q8_1",
            Dtype::Q5_1 => "flambeau_mmvq_q5_1_q8_1",
            Dtype::Q2K => "flambeau_mmvq_q2_K_q8_1",
            Dtype::Q2KR2 => "flambeau_mmvq_q2_K_r2_q8_1",
            Dtype::Q2KDp4a => "flambeau_mmvq_q2_k_dp4a_q8_1",
            Dtype::Q2KR2Dp4a => "flambeau_mmvq_q2_k_r2_dp4a_q8_1",
            Dtype::Q3K => "flambeau_mmvq_q3_k_q8_1",
            Dtype::Q3KR2 => "flambeau_mmvq_q3_k_r2_q8_1",
            Dtype::Q3KDp4a => "flambeau_mmvq_q3_k_dp4a_q8_1",
            Dtype::Q3KDp4aT128Vdr2 => "flambeau_mmvq_q3_k_dp4a_t128_vdr2_q8_1",
            Dtype::Q3KR2Dp4a => "flambeau_mmvq_q3_k_r2_dp4a_q8_1",
            Dtype::Q8K => "flambeau_mmvq_q8_K_q8_1",
            Dtype::Q4K => "flambeau_mmvq_q4_k_q8_1",
            Dtype::Q5K => "flambeau_mmvq_q5_k_q8_1",
            Dtype::Q6K => "flambeau_mmvq_q6_k_q8_1",
            Dtype::Q4KR2 => "flambeau_mmvq_q4_k_r2_q8_1",
            Dtype::Q4KR2Dp4a => "flambeau_mmvq_q4_k_r2_dp4a_q8_1",
            Dtype::Iq4XsDp4aT128Vdr4 => "flambeau_mmvq_iq4_xs_dp4a_t128_vdr4_q8_1",
            Dtype::Q5KDp4aT128Vdr2 => "flambeau_mmvq_q5_k_dp4a_t128_vdr2_q8_1",
            Dtype::Q4KDp4aT128Vdr2 => "flambeau_mmvq_q4_k_dp4a_t128_vdr2_q8_1",
            Dtype::Q5KR2 => "flambeau_mmvq_q5_k_r2_q8_1",
            Dtype::Q5KDp4a => "flambeau_mmvq_q5_k_dp4a_q8_1",
            Dtype::Q5KR2Dp4a => "flambeau_mmvq_q5_k_r2_dp4a_q8_1",
            Dtype::Q6KR4 => "flambeau_mmvq_q6_k_r4_q8_1",
            Dtype::Q6KDP4A => "flambeau_mmvq_q6_k_dp4a_q8_1",
            Dtype::Q4_1 => "flambeau_mmvq_q4_1_q8_1",
            Dtype::Q4_1R2 => "flambeau_mmvq_q4_1_r2_q8_1",
            Dtype::Q4_1R2DP4A => "flambeau_mmvq_q4_1_r2_dp4a_q8_1",
            Dtype::Q4_1T128 => "flambeau_mmvq_q4_1_t128_q8_1",
            Dtype::Q8_0T128 => "flambeau_mmvq_q8_0_t128_q8_1",
            Dtype::Q8_0T128VDR2 => "flambeau_mmvq_q8_0_t128_vdr2_q8_1",
            Dtype::Q8_0R2 => "flambeau_mmvq_q8_0_r2_q8_1",
            Dtype::Q8_0R4 => "flambeau_mmvq_q8_0_r4_q8_1",
            Dtype::Iq4Nl => "flambeau_mmvq_iq4_nl_q8_1",
            Dtype::Iq4NlR2 => "flambeau_mmvq_iq4_nl_r2_q8_1",
            Dtype::Iq4NlDp4a => "flambeau_mmvq_iq4_nl_dp4a_q8_1",
            Dtype::Iq4NlDp4aT128Vdr2 => "flambeau_mmvq_iq4_nl_dp4a_t128_vdr2_q8_1",
            Dtype::Iq4Xs => "flambeau_mmvq_iq4_xs_q8_1",
            Dtype::Iq4XsR2 => "flambeau_mmvq_iq4_xs_r2_q8_1",
            Dtype::Iq4XsDp4a => "flambeau_mmvq_iq4_xs_dp4a_q8_1",
            Dtype::Iq3Xxs => "flambeau_mmvq_iq3_xxs_q8_1",
            Dtype::Iq3XxsR2 => "flambeau_mmvq_iq3_xxs_r2_q8_1",
            Dtype::Iq3XxsDp4a => "flambeau_mmvq_iq3_xxs_dp4a_q8_1",
            Dtype::Iq3S => "flambeau_mmvq_iq3_s_q8_1",
            Dtype::Iq3SR2 => "flambeau_mmvq_iq3_s_r2_q8_1",
            Dtype::Iq3SDp4a => "flambeau_mmvq_iq3_s_dp4a_q8_1",
            Dtype::Iq3SDp4aT128Vdr2 => "flambeau_mmvq_iq3_s_dp4a_t128_vdr2_q8_1",
            Dtype::Iq2Xxs => "flambeau_mmvq_iq2_xxs_q8_1",
            Dtype::Iq2XxsR2 => "flambeau_mmvq_iq2_xxs_r2_q8_1",
            Dtype::Iq2XxsDp4a => "flambeau_mmvq_iq2_xxs_dp4a_q8_1",
            Dtype::Iq2Xs => "flambeau_mmvq_iq2_xs_q8_1",
            Dtype::Iq2XsR2 => "flambeau_mmvq_iq2_xs_r2_q8_1",
            Dtype::Iq2XsDp4a => "flambeau_mmvq_iq2_xs_dp4a_q8_1",
            Dtype::Iq2S => "flambeau_mmvq_iq2_s_q8_1",
            Dtype::Iq2SR2 => "flambeau_mmvq_iq2_s_r2_q8_1",
            Dtype::Iq2SDp4a => "flambeau_mmvq_iq2_s_dp4a_q8_1",
            Dtype::Iq1S => "flambeau_mmvq_iq1_s_q8_1",
            Dtype::Iq1SR2 => "flambeau_mmvq_iq1_s_r2_q8_1",
            Dtype::Iq1SDp4a => "flambeau_mmvq_iq1_s_dp4a_q8_1",
            Dtype::Iq1M => "flambeau_mmvq_iq1_m_q8_1",
            Dtype::Iq1MR2 => "flambeau_mmvq_iq1_m_r2_q8_1",
            Dtype::Iq1MDp4a => "flambeau_mmvq_iq1_m_dp4a_q8_1",
        }
    }

    fn block_size_bytes(self) -> usize {
        match self {
            Dtype::Q8_0 | Dtype::Q8_0T128 | Dtype::Q8_0T128VDR2 | Dtype::Q8_0R2 | Dtype::Q8_0R4 | Dtype::Q8_0Dp4a => {
                std::mem::size_of::<BlockQ8_0>()
            }
            Dtype::Q4_0 | Dtype::Q4_0T128 | Dtype::Q4_0T128VDR2 => std::mem::size_of::<BlockQ4_0>(),
            Dtype::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Dtype::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            Dtype::Q2K | Dtype::Q2KR2 | Dtype::Q2KDp4a | Dtype::Q2KR2Dp4a => std::mem::size_of::<BlockQ2K>(),
            Dtype::Q3K | Dtype::Q3KR2 | Dtype::Q3KDp4a | Dtype::Q3KDp4aT128Vdr2 | Dtype::Q3KR2Dp4a => std::mem::size_of::<BlockQ3K>(),
            Dtype::Q4K | Dtype::Q4KR2 | Dtype::Q4KR2Dp4a | Dtype::Q4KDp4aT128Vdr2 => std::mem::size_of::<BlockQ4K>(),
            Dtype::Q5K | Dtype::Q5KR2 | Dtype::Q5KDp4a | Dtype::Q5KR2Dp4a | Dtype::Q5KDp4aT128Vdr2 => std::mem::size_of::<BlockQ5K>(),
            Dtype::Q6K | Dtype::Q6KR4 | Dtype::Q6KDP4A => std::mem::size_of::<BlockQ6K>(),
            Dtype::Q8K => std::mem::size_of::<BlockQ8K>(),
            Dtype::Q4_1 | Dtype::Q4_1Dp4a | Dtype::Q4_1R2 | Dtype::Q4_1R2DP4A | Dtype::Q4_1T128 => {
                std::mem::size_of::<BlockQ4_1>()
            }
            Dtype::Iq4Nl | Dtype::Iq4NlR2 | Dtype::Iq4NlDp4a | Dtype::Iq4NlDp4aT128Vdr2 => std::mem::size_of::<BlockIq4Nl>(),
            Dtype::Iq4Xs | Dtype::Iq4XsR2 | Dtype::Iq4XsDp4a | Dtype::Iq4XsDp4aT128Vdr4 => std::mem::size_of::<BlockIq4Xs>(),
            Dtype::Iq3Xxs | Dtype::Iq3XxsR2 | Dtype::Iq3XxsDp4a => std::mem::size_of::<BlockIq3Xxs>(),
            Dtype::Iq3S | Dtype::Iq3SR2 | Dtype::Iq3SDp4a | Dtype::Iq3SDp4aT128Vdr2 => std::mem::size_of::<BlockIq3S>(),
            Dtype::Iq2Xxs | Dtype::Iq2XxsR2 | Dtype::Iq2XxsDp4a => std::mem::size_of::<BlockIq2Xxs>(),
            Dtype::Iq2Xs | Dtype::Iq2XsR2 | Dtype::Iq2XsDp4a => std::mem::size_of::<BlockIq2Xs>(),
            Dtype::Iq2S | Dtype::Iq2SR2 | Dtype::Iq2SDp4a => std::mem::size_of::<BlockIq2S>(),
            Dtype::Iq1S | Dtype::Iq1SR2 | Dtype::Iq1SDp4a => std::mem::size_of::<BlockIq1S>(),
            Dtype::Iq1M | Dtype::Iq1MR2 | Dtype::Iq1MDp4a => std::mem::size_of::<BlockIq1M>(),
        }
    }

    fn block_elem_count(self) -> usize {
        self.ggml().block_size()
    }

    fn launch_threads(self) -> u32 {
        match self {
            Dtype::Q8_0 | Dtype::Q8_0Dp4a | Dtype::Q4_0 | Dtype::Q4_1 | Dtype::Q4_1Dp4a | Dtype::Q5_0 | Dtype::Q5_1 | Dtype::Q4_1R2DP4A | Dtype::Q8K | Dtype::Q5KDp4a | Dtype::Q3KDp4a | Dtype::Q2KDp4a | Dtype::Iq4XsDp4a | Dtype::Iq4NlDp4a | Dtype::Iq3SDp4a | Dtype::Iq3XxsDp4a | Dtype::Iq2SDp4a | Dtype::Iq2XsDp4a | Dtype::Iq2XxsDp4a | Dtype::Iq1SDp4a | Dtype::Iq1MDp4a => 256,
            Dtype::Q4_1T128 | Dtype::Q8_0T128 | Dtype::Q8_0T128VDR2 | Dtype::Q4_0T128 | Dtype::Q4_0T128VDR2 => 128,
            Dtype::Q4KDp4aT128Vdr2 | Dtype::Q5KDp4aT128Vdr2 | Dtype::Iq4XsDp4aT128Vdr4 | Dtype::Iq3SDp4aT128Vdr2 | Dtype::Iq4NlDp4aT128Vdr2 | Dtype::Q3KDp4aT128Vdr2 => 128,
            _ => 64,
        }
    }

    /// How many output rows one thread block computes. Used to compute
    /// `gridDim.x` = `n_rows / rows_per_block` (rounded up).
    fn rows_per_block(self) -> u32 {
        match self {
            Dtype::Q6KR4 | Dtype::Q8_0R4 => 4,
            Dtype::Q8_0R2
            | Dtype::Q2KR2
            | Dtype::Q2KR2Dp4a
            | Dtype::Q3KR2
            | Dtype::Q3KR2Dp4a
            | Dtype::Q4KR2
            | Dtype::Q4KR2Dp4a
            | Dtype::Q5KR2
            | Dtype::Q4_1R2
            | Dtype::Q4_1R2DP4A
            | Dtype::Iq4NlR2
            | Dtype::Iq4XsR2
            | Dtype::Iq3XxsR2
            | Dtype::Iq3SR2
            | Dtype::Iq2XxsR2
            | Dtype::Iq2XsR2
            | Dtype::Iq2SR2
            | Dtype::Iq1SR2
            | Dtype::Iq1MR2 => 2,
            // Q6KDP4A is a single-row kernel (inner cooperative across 2 super-blocks).
            _ => 1,
        }
    }
}

/// Sweep spec — one `Cert` per call. M values define the row count; K values
/// the per-row contracted dimension.
#[derive(Debug, Clone)]
pub struct SweepSpec {
    pub dtype: Dtype,
    pub m_grid: Vec<usize>,
    pub k_grid: Vec<usize>,
    pub seed: u64,
}

impl SweepSpec {
    pub fn v1_3_default(dtype: Dtype) -> Self {
        Self {
            dtype,
            m_grid: vec![1, 8, 16, 128, 512],
            k_grid: vec![2048, 5120, 15360],
            seed: 0xC0FFEE,
        }
    }
}

pub fn run_sweep<B: BenchBackend>(spec: &SweepSpec, repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;

    // Capture the kernel's static PMC once per sweep — it's shape-independent
    // (no runtime specialisation), so one query covers every `ShapeResult`.
    let pmc = capture_static_pmc::<B>(&dev, spec.dtype)?;
    // The F16-output entry of the same kernel, when the module has one: it
    // is certified with the F32 entry (same body, only the final store
    // differs) and must reproduce the saturated, rounded F32 output exactly.
    let entry_f16: &'static str =
        Box::leak(format!("{}_f16", spec.dtype.kernel_entry()).into_boxed_str());
    let mut f16_checked = false;

    let mut results = Vec::new();
    for &m in &spec.m_grid {
        for &k in &spec.k_grid {
            let seed = spec
                .seed
                .wrapping_add((m as u64).wrapping_mul(0x1234567))
                .wrapping_add((k as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let (got, reference, f16_mismatches) =
                run_shape::<B>(&dev, spec.dtype, entry_f16, m, k, seed)?;
            let max_rel_err = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt());
            let tolerance = cert_tol(k);
            f16_checked |= f16_mismatches.is_some();
            if let Some(bad) = f16_mismatches.filter(|&bad| bad > 0) {
                tracing::warn!(
                    target: "flambeau_bench::sweep_mmvq",
                    dtype = spec.dtype.name(),
                    m, k, bad,
                    "F16-output entry differs from the saturated F32 output"
                );
            }
            results.push(ShapeResult {
                m,
                k,
                n: k, // single-row MMVQ: N is irrelevant; mirror K for the index
                seed,
                max_rel_err,
                tolerance,
                pass: max_rel_err <= tolerance && f16_mismatches.unwrap_or(0) == 0,
            });
            tracing::info!(
                target: "flambeau_bench::sweep_mmvq",
                dtype = spec.dtype.name(),
                m, k,
                max_rel_err,
                tolerance,
                "shape result"
            );
        }
    }

    let pass = results.iter().all(|r| r.pass);

    let rig = rig_for(arch);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: spec.dtype.impl_id().to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "qmatmul_mmvq".to_string(),
        dtype_weight: spec.dtype.name().to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: if f16_checked {
            "|err| <= 3e-2 * max(|ref|, sqrt(k)); _f16 entry == F32 saturated to ±65504 and rounded"
                .to_string()
        } else {
            "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string()
        },
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig,
        pmc: Some(pmc),
    };

    let written = cert.write_to_disk(repo_root)?;
    tracing::info!(
        target: "flambeau_bench::sweep_mmvq",
        cert = %written.display(),
        pass = cert.pass,
        "cert written"
    );

    Ok(cert)
}

// ---- fused gate+up sweep --------------------------------------------------

/// Weight dtype for the fused gate+up MMVQ sweep. A separate enum from `Dtype`
/// because the op signature differs (two weight matrices in, two outputs out);
/// only Q8_0 is templated so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateUpDtype {
    Q8_0,
    Q4_0T128,
    Q5K,
}

impl GateUpDtype {
    pub fn name(self) -> &'static str {
        match self {
            GateUpDtype::Q8_0 => "Q8_0",
            GateUpDtype::Q4_0T128 => "Q4_0",
            GateUpDtype::Q5K => "Q5_K",
        }
    }
    fn ggml(self) -> GgmlDType {
        match self {
            GateUpDtype::Q8_0 => GgmlDType::Q8_0,
            GateUpDtype::Q4_0T128 => GgmlDType::Q4_0,
            GateUpDtype::Q5K => GgmlDType::Q5K,
        }
    }
    fn weight_dtype(self) -> Dtype {
        match self {
            GateUpDtype::Q8_0 => Dtype::Q8_0Dp4a,
            GateUpDtype::Q4_0T128 => Dtype::Q4_0,
            GateUpDtype::Q5K => Dtype::Q5K,
        }
    }
    fn impl_id(self) -> &'static str {
        match self {
            GateUpDtype::Q8_0 => "qmatmul_q8_0_gate_up_t128_vdr2",
            GateUpDtype::Q4_0T128 => "qmatmul_q4_0_gate_up_t128_dp4a",
            GateUpDtype::Q5K => "qmatmul_q5_k_gate_up_dp4a",
        }
    }
    fn kernel_stem(self) -> &'static str {
        match self {
            GateUpDtype::Q8_0 => "mmvq_q8_0_gate_up_t128_vdr2",
            GateUpDtype::Q4_0T128 => "mmvq_q4_0_gate_up_t128_dp4a",
            GateUpDtype::Q5K => "mmvq_q5_k_gate_up_dp4a",
        }
    }
    fn kernel_entry(self) -> &'static str {
        match self {
            GateUpDtype::Q8_0 => "flambeau_mmvq_q8_0_gate_up_t128_vdr2_q8_1",
            GateUpDtype::Q4_0T128 => "flambeau_mmvq_q4_0_gate_up_t128_dp4a_q8_1",
            GateUpDtype::Q5K => "flambeau_mmvq_q5_k_gate_up_dp4a_q8_1",
        }
    }
    fn launch_threads(self) -> u32 {
        match self {
            GateUpDtype::Q8_0 | GateUpDtype::Q4_0T128 => 128,
            GateUpDtype::Q5K => 256,
        }
    }
}

pub fn run_gate_up_sweep<B: BenchBackend>(
    dtype: GateUpDtype,
    repo_root: &Path,
    arch: &str,
) -> Result<Cert> {
    let dev = B::new_device(0)?;

    let bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let module = B::load(&dev, bytes)?;
    let kernel = B::kernel(&module, dtype.kernel_entry())?;
    let pmc = B::attributes(&kernel)?;

    let m_grid = vec![1usize, 8, 16, 128, 512];
    let k_grid = vec![2048usize, 5120, 15360];
    let seed = 0xC0FFEE_u64;

    let mut results = Vec::new();
    for &m in &m_grid {
        for &k in &k_grid {
            let s = seed
                .wrapping_add((m as u64).wrapping_mul(0x1234567))
                .wrapping_add((k as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let (got_g, got_u, ref_g, ref_u) = run_shape_gate_up::<B>(&dev, dtype, m, k, s)?;
            let floor = (k as f32).sqrt();
            let err_g = max_rel_err_with_floor(&got_g, &ref_g, floor);
            let err_u = max_rel_err_with_floor(&got_u, &ref_u, floor);
            let max_rel_err = err_g.max(err_u);
            let tolerance = cert_tol(k);
            results.push(ShapeResult {
                m,
                k,
                n: k,
                seed: s,
                max_rel_err,
                tolerance,
                pass: max_rel_err <= tolerance,
            });
            tracing::info!(
                target: "flambeau_bench::sweep_mmvq",
                dtype = dtype.name(), m, k, max_rel_err, tolerance, "gate_up shape result"
            );
        }
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: dtype.impl_id().to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "qmatmul_gate_up_mmvq".to_string(),
        dtype_weight: dtype.name().to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    let written = cert.write_to_disk(repo_root)?;
    tracing::info!(
        target: "flambeau_bench::sweep_mmvq",
        cert = %written.display(), pass = cert.pass, "gate_up cert written"
    );
    Ok(cert)
}

/// One (m, k) gate+up shape: gate and up use the same row count `m` (the dense
/// FFN case). Returns `(gate_got, up_got, gate_ref, up_ref)`.
fn run_shape_gate_up<B: BenchBackend>(
    dev: &B::Device,
    dtype: GateUpDtype,
    m: usize,
    k: usize,
    seed: u64,
) -> Result<RefQuad> {
    let q_bytes =
        B::image("quantize_q8_1").ok_or_else(|| anyhow!("quantize_q8_1 not compiled"))?;
    let m_bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let q_module = B::load(dev, q_bytes)?;
    let m_module = B::load(dev, m_bytes)?;
    let k_quantize = B::kernel(&q_module, "flambeau_quantize_row_q8_1")?;
    let k_mmvq = B::kernel(&m_module, dtype.kernel_entry())?;

    let block_elems = dtype.ggml().block_size();
    let block_bytes = dtype.weight_dtype().block_size_bytes();
    assert_eq!(k % block_elems, 0, "k {k} must be multiple of block size {block_elems}");
    let weights_blocks = m * (k / block_elems);

    let gate_raw = tame_scales(dtype.weight_dtype(), seeded_bytes(seed, weights_blocks * block_bytes));
    let up_raw =
        tame_scales(dtype.weight_dtype(), seeded_bytes(seed ^ 0xDEAD_BEEF, weights_blocks * block_bytes));
    let y_f32 = seeded_f32_range(seed.wrapping_add(0xA5A5A5A5), k, -1.0, 1.0);

    let mut gate_dq = vec![0.0f32; m * k];
    let mut up_dq = vec![0.0f32; m * k];
    dequantize_into(dtype.ggml(), &gate_raw, &mut gate_dq)?;
    dequantize_into(dtype.ggml(), &up_raw, &mut up_dq)?;

    let d_gate = alloc_and_upload(dev, gate_raw.as_slice());
    let d_up = alloc_and_upload(dev, up_raw.as_slice());
    let d_y_f32 = alloc_and_upload(dev, y_f32.as_slice());

    let y_q8_1_blocks = k / QK8;
    let y_q8_1_bytes = y_q8_1_blocks * std::mem::size_of::<BlockQ8_1>();
    let d_y_q8_1 = dev.alloc(y_q8_1_bytes)?;
    {
        let stream = dev.default_stream();
        let n_elems = k as i32;
        let d_y_f32_ptr: u64 = d_y_f32.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_y_f32_ptr);
        args.push_arg(&d_y_q8_1_ptr);
        args.push_arg(&n_elems);
        B::launch(&k_quantize, stream, LaunchCfg::one_d(y_q8_1_blocks as u32, QK8 as u32), args)?;
        stream.synchronize()?;
    }

    let d_gate_out = dev.alloc(m * 4)?;
    let d_up_out = dev.alloc(m * 4)?;
    {
        let stream = dev.default_stream();
        let n_rows = m as i32;
        let n_units = (k / block_elems) as i32;
        let gw_ptr: u64 = d_gate.as_usize() as u64;
        let uw_ptr: u64 = d_up.as_usize() as u64;
        let y_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let go_ptr: u64 = d_gate_out.as_usize() as u64;
        let uo_ptr: u64 = d_up_out.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&gw_ptr);
        args.push_arg(&uw_ptr);
        args.push_arg(&y_ptr);
        args.push_arg(&go_ptr);
        args.push_arg(&uo_ptr);
        args.push_arg(&n_rows);
        args.push_arg(&n_rows);
        args.push_arg(&n_units);
        B::launch(&k_mmvq, stream, LaunchCfg::one_d(m as u32, dtype.launch_threads()), args)?;
        stream.synchronize()?;
    }

    let mut got_g = vec![0.0f32; m];
    let mut got_u = vec![0.0f32; m];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_g.as_mut_ptr() as usize),
            d_gate_out,
            m * 4,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_u.as_mut_ptr() as usize),
            d_up_out,
            m * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_gate, gate_raw.len())?;
        dev.dealloc(d_up, up_raw.len())?;
        dev.dealloc(d_y_f32, y_f32.len() * 4)?;
        dev.dealloc(d_y_q8_1, y_q8_1_bytes)?;
        dev.dealloc(d_gate_out, m * 4)?;
        dev.dealloc(d_up_out, m * 4)?;
    }

    let y_rt = quantize_q8_1_roundtrip(&y_f32);
    let ref_g = reference_matmul(&gate_dq, &y_rt, m, k);
    let ref_u = reference_matmul(&up_dq, &y_rt, m, k);
    Ok((got_g, got_u, ref_g, ref_u))
}

/// Fused K+V Q4_0 decode with F16 destination (`mmvq_q4_0_kv_f16dst`): K and V
/// are both Q4_0, share the row count + the Q8_1 activation, and write F16. Same
/// per-row dot as the dense `gate_up` (Q4_0 dp4a + bias), two outputs.
pub fn run_kv_f16dst_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let stem = "mmvq_q4_0_kv_f16dst_dp4a";
    let entry = "flambeau_mmvq_q4_0_kv_f16dst_dp4a_q8_1";
    let bytes = B::image(stem).ok_or_else(|| anyhow!("{stem} not compiled"))?;
    let module = B::load(&dev, bytes)?;
    let pmc = B::attributes(&B::kernel(&module, entry)?)?;

    let m_grid = vec![1usize, 8, 16, 128, 512];
    let k_grid = vec![2048usize, 5120, 15360];
    let seed = 0xC0FFEE_u64;

    let mut results = Vec::new();
    for &m in &m_grid {
        for &k in &k_grid {
            let s = seed
                .wrapping_add((m as u64).wrapping_mul(0x1234567))
                .wrapping_add((k as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let (got_k, got_v, ref_k, ref_v) = run_shape_kv_f16dst::<B>(&dev, m, k, s)?;
            let floor = (k as f32).sqrt();
            let max_rel_err = max_rel_err_with_floor(&got_k, &ref_k, floor)
                .max(max_rel_err_with_floor(&got_v, &ref_v, floor));
            let tolerance = cert_tol(k);
            results.push(ShapeResult {
                m,
                k,
                n: k,
                seed: s,
                max_rel_err,
                tolerance,
                pass: max_rel_err <= tolerance,
            });
        }
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "mmvq_q4_0_kv_f16dst".to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "qmatmul_kv_decode_mmvq".to_string(),
        dtype_weight: "Q4_0".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))  (fused K+V Q4_0, F16 dst)"
            .to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

/// One (m, k) shape for the fused K+V F16-dst kernel. Returns
/// `(k_got, v_got, k_ref, v_ref)` with the device outputs widened from F16.
fn run_shape_kv_f16dst<B: BenchBackend>(
    dev: &B::Device,
    m: usize,
    k: usize,
    seed: u64,
) -> Result<RefQuad> {
    let q_bytes = B::image("quantize_q8_1").ok_or_else(|| anyhow!("quantize_q8_1 not compiled"))?;
    let m_bytes = B::image("mmvq_q4_0_kv_f16dst_dp4a")
        .ok_or_else(|| anyhow!("mmvq_q4_0_kv_f16dst_dp4a not compiled"))?;
    let q_module = B::load(dev, q_bytes)?;
    let m_module = B::load(dev, m_bytes)?;
    let k_quantize = B::kernel(&q_module, "flambeau_quantize_row_q8_1")?;
    let k_mmvq = B::kernel(&m_module, "flambeau_mmvq_q4_0_kv_f16dst_dp4a_q8_1")?;

    let wd = Dtype::Q4_0;
    let block_elems = GgmlDType::Q4_0.block_size();
    let block_bytes = wd.block_size_bytes();
    assert_eq!(k % block_elems, 0, "k {k} must be multiple of block size {block_elems}");
    let weights_blocks = m * (k / block_elems);

    let k_raw = tame_scales(wd, seeded_bytes(seed, weights_blocks * block_bytes));
    let v_raw = tame_scales(wd, seeded_bytes(seed ^ 0xDEAD_BEEF, weights_blocks * block_bytes));
    let y_f32 = seeded_f32_range(seed.wrapping_add(0xA5A5A5A5), k, -1.0, 1.0);

    let mut k_dq = vec![0.0f32; m * k];
    let mut v_dq = vec![0.0f32; m * k];
    dequantize_into(GgmlDType::Q4_0, &k_raw, &mut k_dq)?;
    dequantize_into(GgmlDType::Q4_0, &v_raw, &mut v_dq)?;

    let d_k = alloc_and_upload(dev, k_raw.as_slice());
    let d_v = alloc_and_upload(dev, v_raw.as_slice());
    let d_y_f32 = alloc_and_upload(dev, y_f32.as_slice());

    let y_q8_1_blocks = k / QK8;
    let y_q8_1_bytes = y_q8_1_blocks * std::mem::size_of::<BlockQ8_1>();
    let d_y_q8_1 = dev.alloc(y_q8_1_bytes)?;
    {
        let stream = dev.default_stream();
        let n_elems = k as i32;
        let d_y_f32_ptr: u64 = d_y_f32.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_y_f32_ptr);
        args.push_arg(&d_y_q8_1_ptr);
        args.push_arg(&n_elems);
        B::launch(&k_quantize, stream, LaunchCfg::one_d(y_q8_1_blocks as u32, QK8 as u32), args)?;
        stream.synchronize()?;
    }

    let d_k_out = dev.alloc(m * 2)?;
    let d_v_out = dev.alloc(m * 2)?;
    {
        let stream = dev.default_stream();
        let n_rows = m as i32;
        let n_units = (k / block_elems) as i32;
        let kw_ptr: u64 = d_k.as_usize() as u64;
        let vw_ptr: u64 = d_v.as_usize() as u64;
        let y_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let ko_ptr: u64 = d_k_out.as_usize() as u64;
        let vo_ptr: u64 = d_v_out.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&kw_ptr);
        args.push_arg(&vw_ptr);
        args.push_arg(&y_ptr);
        args.push_arg(&ko_ptr);
        args.push_arg(&vo_ptr);
        args.push_arg(&n_rows);
        args.push_arg(&n_units);
        B::launch(&k_mmvq, stream, LaunchCfg::one_d(m as u32, 256), args)?;
        stream.synchronize()?;
    }

    let mut k_bytes = vec![0u8; m * 2];
    let mut v_bytes = vec![0u8; m * 2];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(k_bytes.as_mut_ptr() as usize),
            d_k_out,
            m * 2,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(v_bytes.as_mut_ptr() as usize),
            d_v_out,
            m * 2,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_k, k_raw.len())?;
        dev.dealloc(d_v, v_raw.len())?;
        dev.dealloc(d_y_f32, y_f32.len() * 4)?;
        dev.dealloc(d_y_q8_1, y_q8_1_bytes)?;
        dev.dealloc(d_k_out, m * 2)?;
        dev.dealloc(d_v_out, m * 2)?;
    }

    let widen = |b: &[u8]| -> Vec<f32> {
        (0..m).map(|i| f16::from_le_bytes([b[2 * i], b[2 * i + 1]]).to_f32()).collect()
    };
    let y_rt = quantize_q8_1_roundtrip(&y_f32);
    let ref_k = reference_matmul(&k_dq, &y_rt, m, k);
    let ref_v = reference_matmul(&v_dq, &y_rt, m, k);
    Ok((widen(&k_bytes), widen(&v_bytes), ref_k, ref_v))
}

// ---- batched (N-slot) sweep -----------------------------------------------

/// The `_n1` .. `_n8` entry names of a batched MMVQ cell.
macro_rules! batched_entries {
    ($base:literal) => {
        [
            concat!($base, "_n1"),
            concat!($base, "_n2"),
            concat!($base, "_n3"),
            concat!($base, "_n4"),
            concat!($base, "_n5"),
            concat!($base, "_n6"),
            concat!($base, "_n7"),
            concat!($base, "_n8"),
        ]
    };
}

/// Weight dtype for the batched MMVQ sweep (one weight row × N activation
/// columns). Separate from `Dtype` because the kernel has N=2/3/4 entries and a
/// slot-major output; only Q8_0 is templated so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchedDtype {
    Q8_0,
    Q4_0,
    Q4_1,
    Q5K,
    Q8_0RowTile,
    Q4_0RowTile,
    /// Row-tiled dp4a batched MMVQ (`mmvq_dp4a_batched_k.cuh`), N = 2..8.
    Q4KDp4aRowTile,
    Q5KDp4aRowTile,
    Iq4XsDp4aRowTile,
    Iq3SDp4aRowTile,
    Iq3XxsDp4aRowTile,
    Iq2SDp4aRowTile,
    Q6KDp4aRowTile,
    /// R10s layout (`mmvq_vdr_batched.cuh`), N = 1..8.
    Iq4XsT128Vdr4Batched,
    /// R10s layout (`mmvq_vdr_batched.cuh`), N = 1..8.
    Q5KT128Vdr2Batched,
    /// R10s layout (`mmvq_vdr_batched.cuh`), N = 1..8.
    Q4KT128Vdr2Batched,
    /// R10s layout (`mmvq_vdr_batched.cuh`), N = 1..8.
    Q6KVdr2Batched,
    /// R10s layout (`mmvq_vdr_batched.cuh`), N = 1..8.
    Q3KT128Vdr2Batched,
    /// R10s layout (`mmvq_vdr_batched.cuh`), N = 1..8.
    Iq4NlT128Vdr2Batched,
}

impl BatchedDtype {
    pub fn name(self) -> &'static str {
        match self {
            BatchedDtype::Q8_0 | BatchedDtype::Q8_0RowTile => "Q8_0",
            BatchedDtype::Q4_0 | BatchedDtype::Q4_0RowTile => "Q4_0",
            BatchedDtype::Q4_1 => "Q4_1",
            BatchedDtype::Q4KDp4aRowTile => "Q4_K",
            BatchedDtype::Q5K | BatchedDtype::Q5KDp4aRowTile => "Q5_K",
            BatchedDtype::Iq4XsDp4aRowTile => "IQ4_XS",
            BatchedDtype::Iq3SDp4aRowTile => "IQ3_S",
            BatchedDtype::Iq3XxsDp4aRowTile => "IQ3_XXS",
            BatchedDtype::Iq2SDp4aRowTile => "IQ2_S",
            BatchedDtype::Q6KDp4aRowTile => "Q6_K",
            BatchedDtype::Iq4NlT128Vdr2Batched => "IQ4_NL",
            BatchedDtype::Q3KT128Vdr2Batched => "Q3_K",
            BatchedDtype::Q6KVdr2Batched => "Q6_K",
            BatchedDtype::Q4KT128Vdr2Batched => "Q4_K",
            BatchedDtype::Q5KT128Vdr2Batched => "Q5_K",
            BatchedDtype::Iq4XsT128Vdr4Batched => "IQ4_XS",
        }
    }
    fn weight_dtype(self) -> Dtype {
        match self {
            BatchedDtype::Q8_0 | BatchedDtype::Q8_0RowTile => Dtype::Q8_0Dp4a,
            BatchedDtype::Q4_0 | BatchedDtype::Q4_0RowTile => Dtype::Q4_0,
            BatchedDtype::Q4_1 => Dtype::Q4_1Dp4a,
            BatchedDtype::Q4KDp4aRowTile => Dtype::Q4K,
            BatchedDtype::Q5K | BatchedDtype::Q5KDp4aRowTile => Dtype::Q5K,
            BatchedDtype::Iq4XsDp4aRowTile => Dtype::Iq4XsDp4a,
            BatchedDtype::Iq3SDp4aRowTile => Dtype::Iq3SDp4a,
            BatchedDtype::Iq3XxsDp4aRowTile => Dtype::Iq3XxsDp4a,
            BatchedDtype::Iq2SDp4aRowTile => Dtype::Iq2SDp4a,
            BatchedDtype::Q6KDp4aRowTile => Dtype::Q6KDP4A,
            BatchedDtype::Iq4NlT128Vdr2Batched => Dtype::Iq4NlDp4a,
            BatchedDtype::Q3KT128Vdr2Batched => Dtype::Q3KDp4a,
            BatchedDtype::Q6KVdr2Batched => Dtype::Q6KDP4A,
            BatchedDtype::Q4KT128Vdr2Batched => Dtype::Q4K,
            BatchedDtype::Q5KT128Vdr2Batched => Dtype::Q5K,
            BatchedDtype::Iq4XsT128Vdr4Batched => Dtype::Iq4XsDp4a,
        }
    }
    fn impl_id(self) -> &'static str {
        match self {
            BatchedDtype::Q8_0 => "qmatmul_q8_0_mmvq_batched",
            BatchedDtype::Q4_0 => "qmatmul_q4_0_mmvq_batched",
            BatchedDtype::Q4_1 => "qmatmul_q4_1_mmvq_batched",
            BatchedDtype::Q5K => "qmatmul_q5_k_r2_mmvq_batched",
            BatchedDtype::Q8_0RowTile => "qmatmul_q8_0_mmvq_row_tile_batched",
            BatchedDtype::Q4_0RowTile => "qmatmul_q4_0_mmvq_row_tile_batched",
            BatchedDtype::Q4KDp4aRowTile => "qmatmul_q4_k_mmvq_dp4a_row_tile_batched",
            BatchedDtype::Q5KDp4aRowTile => "qmatmul_q5_k_mmvq_dp4a_row_tile_batched",
            BatchedDtype::Iq4XsDp4aRowTile => "qmatmul_iq4_xs_mmvq_dp4a_row_tile_batched",
            BatchedDtype::Iq3SDp4aRowTile => "qmatmul_iq3_s_mmvq_dp4a_row_tile_batched",
            BatchedDtype::Iq3XxsDp4aRowTile => "qmatmul_iq3_xxs_mmvq_dp4a_row_tile_batched",
            BatchedDtype::Iq2SDp4aRowTile => "qmatmul_iq2_s_mmvq_dp4a_row_tile_batched",
            BatchedDtype::Q6KDp4aRowTile => "qmatmul_q6_k_mmvq_dp4a_row_tile_batched",
            BatchedDtype::Iq4NlT128Vdr2Batched => "qmatmul_iq4_nl_mmvq_dp4a_t128_vdr2_batched",
            BatchedDtype::Q3KT128Vdr2Batched => "qmatmul_q3_k_mmvq_dp4a_t128_vdr2_batched",
            BatchedDtype::Q6KVdr2Batched => "qmatmul_q6_k_mmvq_dp4a_vdr2_batched",
            BatchedDtype::Q4KT128Vdr2Batched => "qmatmul_q4_k_mmvq_dp4a_t128_vdr2_batched",
            BatchedDtype::Q5KT128Vdr2Batched => "qmatmul_q5_k_mmvq_dp4a_t128_vdr2_batched",
            BatchedDtype::Iq4XsT128Vdr4Batched => "qmatmul_iq4_xs_mmvq_dp4a_t128_vdr4_batched",
        }
    }
    fn kernel_stem(self) -> &'static str {
        match self {
            BatchedDtype::Q8_0 => "mmvq_q8_0_batched",
            BatchedDtype::Q4_0 => "mmvq_q4_0_batched",
            BatchedDtype::Q4_1 => "mmvq_q4_1_batched",
            BatchedDtype::Q5K => "mmvq_q5_k_r2_batched",
            BatchedDtype::Q8_0RowTile => "mmvq_q8_0_row_tile_batched",
            BatchedDtype::Q4_0RowTile => "mmvq_q4_0_row_tile_batched",
            BatchedDtype::Q4KDp4aRowTile => "mmvq_q4_k_dp4a_batched",
            BatchedDtype::Q5KDp4aRowTile => "mmvq_q5_k_dp4a_batched",
            BatchedDtype::Iq4XsDp4aRowTile => "mmvq_iq4_xs_dp4a_batched",
            BatchedDtype::Iq3SDp4aRowTile => "mmvq_iq3_s_dp4a_batched",
            BatchedDtype::Iq3XxsDp4aRowTile => "mmvq_iq3_xxs_dp4a_batched",
            BatchedDtype::Iq2SDp4aRowTile => "mmvq_iq2_s_dp4a_batched",
            BatchedDtype::Q6KDp4aRowTile => "mmvq_q6_k_dp4a_batched",
            BatchedDtype::Iq4NlT128Vdr2Batched => "mmvq_iq4_nl_dp4a_t128_vdr2_batched",
            BatchedDtype::Q3KT128Vdr2Batched => "mmvq_q3_k_dp4a_t128_vdr2_batched",
            BatchedDtype::Q6KVdr2Batched => "mmvq_q6_k_dp4a_vdr2_batched",
            BatchedDtype::Q4KT128Vdr2Batched => "mmvq_q4_k_dp4a_t128_vdr2_batched",
            BatchedDtype::Q5KT128Vdr2Batched => "mmvq_q5_k_dp4a_t128_vdr2_batched",
            BatchedDtype::Iq4XsT128Vdr4Batched => "mmvq_iq4_xs_dp4a_t128_vdr4_batched",
        }
    }
    /// The R10s-layout batched kernels (`mmvq_vdr_batched.cuh`).
    fn is_vdr_batched(self) -> bool {
        matches!(self, BatchedDtype::Iq4XsT128Vdr4Batched | BatchedDtype::Iq4NlT128Vdr2Batched | BatchedDtype::Q3KT128Vdr2Batched | BatchedDtype::Q6KVdr2Batched | BatchedDtype::Q4KT128Vdr2Batched | BatchedDtype::Q5KT128Vdr2Batched)
    }
    fn launch_threads(self) -> u32 {
        match self {
            BatchedDtype::Q8_0
            | BatchedDtype::Q4_0
            | BatchedDtype::Q4_1
            | BatchedDtype::Q8_0RowTile
            | BatchedDtype::Q4_0RowTile
            | BatchedDtype::Q4KDp4aRowTile
            | BatchedDtype::Q5KDp4aRowTile
            | BatchedDtype::Iq4XsDp4aRowTile
            | BatchedDtype::Iq3SDp4aRowTile
            | BatchedDtype::Iq3XxsDp4aRowTile
            | BatchedDtype::Iq2SDp4aRowTile
            | BatchedDtype::Q6KDp4aRowTile => 256,
            BatchedDtype::Q5K => 64,
            BatchedDtype::Iq4NlT128Vdr2Batched => 32,
            BatchedDtype::Q3KT128Vdr2Batched => 32,
            BatchedDtype::Q6KVdr2Batched => 32,
            BatchedDtype::Q4KT128Vdr2Batched => 32,
            BatchedDtype::Q5KT128Vdr2Batched => 32,
            BatchedDtype::Iq4XsT128Vdr4Batched => 32,
        }
    }
    /// Output rows computed per block: 2 for the r2 Q5_K kernel, 8 for the
    /// 32-lane-group Q5_K row-tile (same on both backends), 4 for the
    /// Q8_0/Q4_0 row-tiles (over-launches harmlessly on wave32's R=8), 1 else.
    /// (threads, rows per block) of the `n_slots`-column entry.
    fn launch_shape(self, n_slots: usize) -> (u32, u32) {
        if self.is_vdr_batched() {
            let _ = n_slots;
            flambeau_ops::qmatmul::mmvq_batched_shape(self.impl_id()).expect("batched recipe")
        } else {
            (self.launch_threads(), self.rows_per_block())
        }
    }
    fn rows_per_block(self) -> u32 {
        match self {
            BatchedDtype::Q5K => 2,
            BatchedDtype::Q8_0RowTile | BatchedDtype::Q4_0RowTile => 4,
            BatchedDtype::Q4KDp4aRowTile
            | BatchedDtype::Q5KDp4aRowTile
            | BatchedDtype::Iq4XsDp4aRowTile
            | BatchedDtype::Iq3SDp4aRowTile
            | BatchedDtype::Iq3XxsDp4aRowTile
            | BatchedDtype::Iq2SDp4aRowTile
            | BatchedDtype::Q6KDp4aRowTile => 8,
            _ => 1,
        }
    }
    /// Activation-column counts the dtype has kernel entries for.
    fn n_grid(self) -> &'static [usize] {
        match self {
            BatchedDtype::Q4KDp4aRowTile
            | BatchedDtype::Q5KDp4aRowTile
            | BatchedDtype::Iq4XsDp4aRowTile
            | BatchedDtype::Iq3SDp4aRowTile
            | BatchedDtype::Iq3XxsDp4aRowTile
            | BatchedDtype::Iq2SDp4aRowTile
            | BatchedDtype::Q6KDp4aRowTile => &[2, 3, 4, 5, 6, 7, 8],
            BatchedDtype::Iq4XsT128Vdr4Batched => &[1, 2, 3, 4, 5, 6, 7, 8],
            BatchedDtype::Q5KT128Vdr2Batched => &[1, 2, 3, 4, 5, 6, 7, 8],
            BatchedDtype::Q4KT128Vdr2Batched => &[1, 2, 3, 4, 5, 6, 7, 8],
            BatchedDtype::Q6KVdr2Batched => &[1, 2, 3, 4, 5, 6, 7, 8],
            BatchedDtype::Q3KT128Vdr2Batched => &[1, 2, 3, 4, 5, 6, 7, 8],
            BatchedDtype::Iq4NlT128Vdr2Batched => &[1, 2, 3, 4, 5, 6, 7, 8],
            _ => &[2, 3, 4],
        }
    }
    fn kernel_entry(self, n_slots: usize) -> &'static str {
        match (self, n_slots) {
            (BatchedDtype::Q8_0, 2) => "flambeau_mmvq_q8_0_q8_1_batched_n2",
            (BatchedDtype::Q8_0, 3) => "flambeau_mmvq_q8_0_q8_1_batched_n3",
            (BatchedDtype::Q8_0, 4) => "flambeau_mmvq_q8_0_q8_1_batched_n4",
            (BatchedDtype::Q4_0, 2) => "flambeau_mmvq_q4_0_q8_1_batched_n2",
            (BatchedDtype::Q4_0, 3) => "flambeau_mmvq_q4_0_q8_1_batched_n3",
            (BatchedDtype::Q4_0, 4) => "flambeau_mmvq_q4_0_q8_1_batched_n4",
            (BatchedDtype::Q4_1, 2) => "flambeau_mmvq_q4_1_q8_1_batched_n2",
            (BatchedDtype::Q4_1, 3) => "flambeau_mmvq_q4_1_q8_1_batched_n3",
            (BatchedDtype::Q4_1, 4) => "flambeau_mmvq_q4_1_q8_1_batched_n4",
            (BatchedDtype::Q5K, 2) => "flambeau_mmvq_q5_k_r2_q8_1_batched_n2",
            (BatchedDtype::Q5K, 3) => "flambeau_mmvq_q5_k_r2_q8_1_batched_n3",
            (BatchedDtype::Q5K, 4) => "flambeau_mmvq_q5_k_r2_q8_1_batched_n4",
            (BatchedDtype::Q8_0RowTile, 2) => "flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n2",
            (BatchedDtype::Q8_0RowTile, 3) => "flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n3",
            (BatchedDtype::Q8_0RowTile, 4) => "flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n4",
            (BatchedDtype::Q4_0RowTile, 2) => "flambeau_mmvq_q4_0_row_tile_dp4a_q8_1_batched_n2",
            (BatchedDtype::Q4_0RowTile, 3) => "flambeau_mmvq_q4_0_row_tile_dp4a_q8_1_batched_n3",
            (BatchedDtype::Q4_0RowTile, 4) => "flambeau_mmvq_q4_0_row_tile_dp4a_q8_1_batched_n4",
            (BatchedDtype::Q4KDp4aRowTile, 2) => "flambeau_mmvq_q4_k_dp4a_q8_1_batched_n2",
            (BatchedDtype::Q4KDp4aRowTile, 3) => "flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3",
            (BatchedDtype::Q4KDp4aRowTile, 4) => "flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4",
            (BatchedDtype::Q4KDp4aRowTile, 5) => "flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5",
            (BatchedDtype::Q4KDp4aRowTile, 6) => "flambeau_mmvq_q4_k_dp4a_q8_1_batched_n6",
            (BatchedDtype::Q4KDp4aRowTile, 7) => "flambeau_mmvq_q4_k_dp4a_q8_1_batched_n7",
            (BatchedDtype::Q4KDp4aRowTile, 8) => "flambeau_mmvq_q4_k_dp4a_q8_1_batched_n8",
            (BatchedDtype::Q5KDp4aRowTile, 2) => "flambeau_mmvq_q5_k_dp4a_q8_1_batched_n2",
            (BatchedDtype::Q5KDp4aRowTile, 3) => "flambeau_mmvq_q5_k_dp4a_q8_1_batched_n3",
            (BatchedDtype::Q5KDp4aRowTile, 4) => "flambeau_mmvq_q5_k_dp4a_q8_1_batched_n4",
            (BatchedDtype::Q5KDp4aRowTile, 5) => "flambeau_mmvq_q5_k_dp4a_q8_1_batched_n5",
            (BatchedDtype::Q5KDp4aRowTile, 6) => "flambeau_mmvq_q5_k_dp4a_q8_1_batched_n6",
            (BatchedDtype::Q5KDp4aRowTile, 7) => "flambeau_mmvq_q5_k_dp4a_q8_1_batched_n7",
            (BatchedDtype::Q5KDp4aRowTile, 8) => "flambeau_mmvq_q5_k_dp4a_q8_1_batched_n8",
            (BatchedDtype::Iq4XsDp4aRowTile, 2) => "flambeau_mmvq_iq4_xs_dp4a_q8_1_batched_n2",
            (BatchedDtype::Iq4XsDp4aRowTile, 3) => "flambeau_mmvq_iq4_xs_dp4a_q8_1_batched_n3",
            (BatchedDtype::Iq4XsDp4aRowTile, 4) => "flambeau_mmvq_iq4_xs_dp4a_q8_1_batched_n4",
            (BatchedDtype::Iq4XsDp4aRowTile, 5) => "flambeau_mmvq_iq4_xs_dp4a_q8_1_batched_n5",
            (BatchedDtype::Iq4XsDp4aRowTile, 6) => "flambeau_mmvq_iq4_xs_dp4a_q8_1_batched_n6",
            (BatchedDtype::Iq4XsDp4aRowTile, 7) => "flambeau_mmvq_iq4_xs_dp4a_q8_1_batched_n7",
            (BatchedDtype::Iq4XsDp4aRowTile, 8) => "flambeau_mmvq_iq4_xs_dp4a_q8_1_batched_n8",
            (BatchedDtype::Iq3SDp4aRowTile, 2) => "flambeau_mmvq_iq3_s_dp4a_q8_1_batched_n2",
            (BatchedDtype::Iq3SDp4aRowTile, 3) => "flambeau_mmvq_iq3_s_dp4a_q8_1_batched_n3",
            (BatchedDtype::Iq3SDp4aRowTile, 4) => "flambeau_mmvq_iq3_s_dp4a_q8_1_batched_n4",
            (BatchedDtype::Iq3SDp4aRowTile, 5) => "flambeau_mmvq_iq3_s_dp4a_q8_1_batched_n5",
            (BatchedDtype::Iq3SDp4aRowTile, 6) => "flambeau_mmvq_iq3_s_dp4a_q8_1_batched_n6",
            (BatchedDtype::Iq3SDp4aRowTile, 7) => "flambeau_mmvq_iq3_s_dp4a_q8_1_batched_n7",
            (BatchedDtype::Iq3SDp4aRowTile, 8) => "flambeau_mmvq_iq3_s_dp4a_q8_1_batched_n8",
            (BatchedDtype::Iq3XxsDp4aRowTile, 2) => "flambeau_mmvq_iq3_xxs_dp4a_q8_1_batched_n2",
            (BatchedDtype::Iq3XxsDp4aRowTile, 3) => "flambeau_mmvq_iq3_xxs_dp4a_q8_1_batched_n3",
            (BatchedDtype::Iq3XxsDp4aRowTile, 4) => "flambeau_mmvq_iq3_xxs_dp4a_q8_1_batched_n4",
            (BatchedDtype::Iq3XxsDp4aRowTile, 5) => "flambeau_mmvq_iq3_xxs_dp4a_q8_1_batched_n5",
            (BatchedDtype::Iq3XxsDp4aRowTile, 6) => "flambeau_mmvq_iq3_xxs_dp4a_q8_1_batched_n6",
            (BatchedDtype::Iq3XxsDp4aRowTile, 7) => "flambeau_mmvq_iq3_xxs_dp4a_q8_1_batched_n7",
            (BatchedDtype::Iq3XxsDp4aRowTile, 8) => "flambeau_mmvq_iq3_xxs_dp4a_q8_1_batched_n8",
            (BatchedDtype::Iq2SDp4aRowTile, 2) => "flambeau_mmvq_iq2_s_dp4a_q8_1_batched_n2",
            (BatchedDtype::Iq2SDp4aRowTile, 3) => "flambeau_mmvq_iq2_s_dp4a_q8_1_batched_n3",
            (BatchedDtype::Iq2SDp4aRowTile, 4) => "flambeau_mmvq_iq2_s_dp4a_q8_1_batched_n4",
            (BatchedDtype::Iq2SDp4aRowTile, 5) => "flambeau_mmvq_iq2_s_dp4a_q8_1_batched_n5",
            (BatchedDtype::Iq2SDp4aRowTile, 6) => "flambeau_mmvq_iq2_s_dp4a_q8_1_batched_n6",
            (BatchedDtype::Iq2SDp4aRowTile, 7) => "flambeau_mmvq_iq2_s_dp4a_q8_1_batched_n7",
            (BatchedDtype::Iq2SDp4aRowTile, 8) => "flambeau_mmvq_iq2_s_dp4a_q8_1_batched_n8",
            (BatchedDtype::Q6KDp4aRowTile, 2) => "flambeau_mmvq_q6_k_dp4a_q8_1_batched_n2",
            (BatchedDtype::Q6KDp4aRowTile, 3) => "flambeau_mmvq_q6_k_dp4a_q8_1_batched_n3",
            (BatchedDtype::Q6KDp4aRowTile, 4) => "flambeau_mmvq_q6_k_dp4a_q8_1_batched_n4",
            (BatchedDtype::Q6KDp4aRowTile, 5) => "flambeau_mmvq_q6_k_dp4a_q8_1_batched_n5",
            (BatchedDtype::Q6KDp4aRowTile, 6) => "flambeau_mmvq_q6_k_dp4a_q8_1_batched_n6",
            (BatchedDtype::Q6KDp4aRowTile, 7) => "flambeau_mmvq_q6_k_dp4a_q8_1_batched_n7",
            (BatchedDtype::Q6KDp4aRowTile, 8) => "flambeau_mmvq_q6_k_dp4a_q8_1_batched_n8",
            (BatchedDtype::Iq4XsT128Vdr4Batched, n) => batched_entries!("flambeau_mmvq_iq4_xs_dp4a_t128_vdr4_batched_q8_1")[n - 1],
            (BatchedDtype::Q5KT128Vdr2Batched, n) => batched_entries!("flambeau_mmvq_q5_k_dp4a_t128_vdr2_batched_q8_1")[n - 1],
            (BatchedDtype::Q4KT128Vdr2Batched, n) => batched_entries!("flambeau_mmvq_q4_k_dp4a_t128_vdr2_batched_q8_1")[n - 1],
            (BatchedDtype::Q6KVdr2Batched, n) => batched_entries!("flambeau_mmvq_q6_k_dp4a_vdr2_batched_q8_1")[n - 1],
            (BatchedDtype::Q3KT128Vdr2Batched, n) => batched_entries!("flambeau_mmvq_q3_k_dp4a_t128_vdr2_batched_q8_1")[n - 1],
            (BatchedDtype::Iq4NlT128Vdr2Batched, n) => batched_entries!("flambeau_mmvq_iq4_nl_dp4a_t128_vdr2_batched_q8_1")[n - 1],
            _ => unreachable!("batched n_slots has no kernel entry"),
        }
    }
}

pub fn run_batched_sweep<B: BenchBackend>(
    dtype: BatchedDtype,
    repo_root: &Path,
    arch: &str,
) -> Result<Cert> {
    let dev = B::new_device(0)?;

    let bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let module = B::load(&dev, bytes)?;
    let kernel = B::kernel(&module, dtype.kernel_entry(2))?;
    let pmc = B::attributes(&kernel)?;

    let m_grid = vec![1usize, 8, 16, 128, 512];
    let k_grid = vec![2048usize, 5120, 15360];
    let n_grid = dtype.n_grid().to_vec();
    let seed = 0xC0FFEE_u64;

    let mut results = Vec::new();
    for &m in &m_grid {
        for &k in &k_grid {
            for &n in &n_grid {
                let s = seed
                    .wrapping_add((m as u64).wrapping_mul(0x1234567))
                    .wrapping_add((k as u64).wrapping_mul(0x9E3779B97F4A7C15))
                    .wrapping_add((n as u64).wrapping_mul(0x85EBCA6B));
                let (got, reference) = run_shape_batched::<B>(&dev, dtype, m, k, n, s)?;
                let max_rel_err = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt());
                let tolerance = cert_tol(k);
                results.push(ShapeResult {
                    m,
                    k,
                    n,
                    seed: s,
                    max_rel_err,
                    tolerance,
                    pass: max_rel_err <= tolerance,
                });
                tracing::info!(
                    target: "flambeau_bench::sweep_mmvq",
                    dtype = dtype.name(), m, k, n, max_rel_err, tolerance, "batched shape result"
                );
            }
        }
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: dtype.impl_id().to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "qmatmul_batched_mmvq".to_string(),
        dtype_weight: dtype.name().to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    let written = cert.write_to_disk(repo_root)?;
    tracing::info!(
        target: "flambeau_bench::sweep_mmvq",
        cert = %written.display(), pass = cert.pass, "batched cert written"
    );
    Ok(cert)
}

/// One (m, k, n_slots) batched shape. `n_slots` activation columns (each a
/// distinct k-vector) are quantized slot-major; the kernel writes the slot-major
/// `dst[n_slots][m]`. Returns `(got, reference)` flattened slot-major.
fn run_shape_batched<B: BenchBackend>(
    dev: &B::Device,
    dtype: BatchedDtype,
    m: usize,
    k: usize,
    n_slots: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let q_bytes =
        B::image("quantize_q8_1").ok_or_else(|| anyhow!("quantize_q8_1 not compiled"))?;
    let m_bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let q_module = B::load(dev, q_bytes)?;
    let m_module = B::load(dev, m_bytes)?;
    let k_quantize = B::kernel(&q_module, "flambeau_quantize_row_q8_1")?;
    let k_mmvq = B::kernel(&m_module, dtype.kernel_entry(n_slots))?;

    let wd = dtype.weight_dtype();
    let block_elems = wd.block_elem_count();
    let block_bytes = wd.block_size_bytes();
    assert_eq!(k % block_elems, 0, "k {k} must be multiple of block size {block_elems}");
    let n_blocks = k / block_elems;

    let weights_raw = tame_scales(wd, seeded_bytes(seed, m * n_blocks * block_bytes));
    let mut weights_dequant = vec![0.0f32; m * k];
    dequantize_into(wd.ggml(), &weights_raw, &mut weights_dequant)?;
    let d_x = alloc_and_upload(dev, weights_raw.as_slice());

    // Per-slot activations: quantize each into a slot-major Q8_1 buffer.
    let q8_1_blocks_per_slot = k / QK8;
    let slot_bytes = q8_1_blocks_per_slot * std::mem::size_of::<BlockQ8_1>();
    let d_y_q8_1 = dev.alloc(n_slots * slot_bytes)?;
    let mut y_rt_slots: Vec<Vec<f32>> = Vec::with_capacity(n_slots);
    for s in 0..n_slots {
        let y_f32 = seeded_f32_range(seed.wrapping_add(0xA5A5 ^ (s as u64)), k, -1.0, 1.0);
        let d_y_f32 = alloc_and_upload(dev, y_f32.as_slice());
        {
            let stream = dev.default_stream();
            let n_elems = k as i32;
            let d_y_f32_ptr: u64 = d_y_f32.as_usize() as u64;
            let d_slot_ptr: u64 = (d_y_q8_1.as_usize() + s * slot_bytes) as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&d_y_f32_ptr);
            args.push_arg(&d_slot_ptr);
            args.push_arg(&n_elems);
            B::launch(&k_quantize, stream, LaunchCfg::one_d(q8_1_blocks_per_slot as u32, QK8 as u32), args)?;
            stream.synchronize()?;
        }
        unsafe { dev.dealloc(d_y_f32, y_f32.len() * 4)?; }
        y_rt_slots.push(quantize_q8_1_roundtrip(&y_f32));
    }

    let d_dst = dev.alloc(n_slots * m * 4)?;
    {
        let stream = dev.default_stream();
        let n_rows = m as i32;
        let n_units = n_blocks as i32;
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_y_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let d_dst_ptr: u64 = d_dst.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_x_ptr);
        args.push_arg(&d_y_ptr);
        args.push_arg(&d_dst_ptr);
        args.push_arg(&n_rows);
        args.push_arg(&n_units);
        let (threads, rows_per_block) = dtype.launch_shape(n_slots);
        let grid = (m as u32).div_ceil(rows_per_block);
        B::launch(&k_mmvq, stream, LaunchCfg::one_d(grid, threads), args)?;
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_slots * m];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            n_slots * m * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_x, weights_raw.len())?;
        dev.dealloc(d_y_q8_1, n_slots * slot_bytes)?;
        dev.dealloc(d_dst, n_slots * m * 4)?;
    }

    // Reference, slot-major to match dst[s*m + r].
    let mut reference = vec![0.0f32; n_slots * m];
    for s in 0..n_slots {
        let r = reference_matmul(&weights_dequant, &y_rt_slots[s], m, k);
        reference[s * m..(s + 1) * m].copy_from_slice(&r);
    }
    Ok((got, reference))
}

// ---- PMC capture ----------------------------------------------------------

fn capture_static_pmc<B: BenchBackend>(dev: &B::Device, dtype: Dtype) -> Result<PmcSnapshot> {
    let bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let module = B::load(dev, bytes)?;
    let kernel = B::kernel(&module, dtype.kernel_entry())?;
    let pmc = B::attributes(&kernel)?;

    tracing::info!(
        target: "flambeau_bench::sweep_mmvq",
        dtype = dtype.name(),
        vgpr = pmc.vgpr_count.unwrap_or(0),
        waves_per_simd = pmc.waves_per_simd.unwrap_or(0),
        "static PMC captured"
    );

    Ok(pmc)
}

// ---- per-shape runner -----------------------------------------------------

/// Returns the F32 output, the CPU reference, and — when the module has
/// `entry_f16` — how many rows of the F16-output entry differ from the F32
/// output saturated to ±65504 and rounded to F16.
fn run_shape<B: BenchBackend>(
    dev: &B::Device,
    dtype: Dtype,
    entry_f16: &'static str,
    m: usize,
    k: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>, Option<usize>)> {
    let q_bytes =
        B::image("quantize_q8_1").ok_or_else(|| anyhow!("quantize_q8_1 not compiled"))?;
    let m_bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let q_module = B::load(dev, q_bytes)?;
    let m_module = B::load(dev, m_bytes)?;
    let k_quantize = B::kernel(&q_module, "flambeau_quantize_row_q8_1")?;
    let k_mmvq = B::kernel(&m_module, dtype.kernel_entry())?;
    let k_mmvq_f16 = B::kernel(&m_module, entry_f16).ok();

    let block_elems = dtype.block_elem_count();
    assert_eq!(
        k % block_elems,
        0,
        "k {k} must be multiple of block size {block_elems} for {}",
        dtype.name()
    );

    let weights_blocks = m * (k / block_elems);
    let block_bytes = dtype.block_size_bytes();

    // Generate random weights (raw bit pattern — the kernel operates on bytes).
    let weights_raw = seeded_bytes(seed, weights_blocks * block_bytes);
    let weights_raw = tame_scales(dtype, weights_raw);
    let y_f32 = seeded_f32_range(seed.wrapping_add(0xA5A5A5A5), k, -1.0, 1.0);

    // Dequantise weights on CPU for the reference matmul.
    let total_elems = m * k;
    let mut weights_dequant = vec![0.0f32; total_elems];
    dequantize_into(dtype.ggml(), &weights_raw, &mut weights_dequant)?;

    // Upload inputs.
    let d_x = alloc_and_upload(dev, weights_raw.as_slice());
    let d_y_f32 = alloc_and_upload(dev, y_f32.as_slice());

    // Quantise activation to Q8_1 on device.
    let y_q8_1_blocks = k / QK8;
    let y_q8_1_bytes = y_q8_1_blocks * std::mem::size_of::<BlockQ8_1>();
    let d_y_q8_1 = dev.alloc(y_q8_1_bytes)?;
    {
        let stream = dev.default_stream();
        let n_elems = k as i32;
        let d_y_f32_ptr: u64 = d_y_f32.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_y_f32_ptr);
        args.push_arg(&d_y_q8_1_ptr);
        args.push_arg(&n_elems);
        let cfg = LaunchCfg::one_d(y_q8_1_blocks as u32, QK8 as u32);
        B::launch(&k_quantize, stream, cfg, args)?;
        stream.synchronize()?;
    }

    // MMVQ launch.
    let d_dst = dev.alloc(m * 4)?;
    {
        let stream = dev.default_stream();
        let m_i = m as i32;
        let n_units = (k / block_elems) as i32;
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let d_dst_ptr: u64 = d_dst.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_x_ptr);
        args.push_arg(&d_y_q8_1_ptr);
        args.push_arg(&d_dst_ptr);
        args.push_arg(&m_i);
        args.push_arg(&n_units);
        let rpb = dtype.rows_per_block();
        let grid = (m as u32).div_ceil(rpb);
        let cfg = LaunchCfg::one_d(grid, dtype.launch_threads());
        B::launch(&k_mmvq, stream, cfg, args)?;
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; m];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            m * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    let f16_mismatches = match k_mmvq_f16 {
        None => None,
        Some(k_f16) => {
            let d_dst_f16 = dev.alloc(m * 2)?;
            {
                let stream = dev.default_stream();
                let m_i = m as i32;
                let n_units = (k / block_elems) as i32;
                let d_x_ptr: u64 = d_x.as_usize() as u64;
                let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
                let d_dst_ptr: u64 = d_dst_f16.as_usize() as u64;
                let mut args = <B::Args<'_>>::default();
                args.push_arg(&d_x_ptr);
                args.push_arg(&d_y_q8_1_ptr);
                args.push_arg(&d_dst_ptr);
                args.push_arg(&m_i);
                args.push_arg(&n_units);
                let grid = (m as u32).div_ceil(dtype.rows_per_block());
                let cfg = LaunchCfg::one_d(grid, dtype.launch_threads());
                B::launch(&k_f16, stream, cfg, args)?;
                stream.synchronize()?;
            }
            let mut got_f16 = vec![0u16; m];
            unsafe {
                dev.memcpy_async(
                    dev.default_stream(),
                    CopyDirection::DeviceToHost,
                    DevicePtr(got_f16.as_mut_ptr() as usize),
                    d_dst_f16,
                    m * 2,
                )?;
            }
            dev.default_stream().synchronize()?;
            unsafe {
                dev.dealloc(d_dst_f16, m * 2)?;
            }
            let bad = got
                .iter()
                .zip(&got_f16)
                .filter(|(&v, &h)| f16::from_f32(v.clamp(-65504.0, 65504.0)).to_bits() != h)
                .count();
            Some(bad)
        }
    };

    unsafe {
        dev.dealloc(d_x, weights_raw.len())?;
        dev.dealloc(d_y_f32, y_f32.len() * 4)?;
        dev.dealloc(d_y_q8_1, y_q8_1_bytes)?;
        dev.dealloc(d_dst, m * 4)?;
    }

    let y_rt = quantize_q8_1_roundtrip(&y_f32);
    let reference = reference_matmul(&weights_dequant, &y_rt, m, k);
    Ok((got, reference, f16_mismatches))
}

// ---- fused gate+up batched (N-slot) sweep ---------------------------------

/// Weight dtype for the fused gate+up batched MMVQ sweep (two weight matrices,
/// N activation columns, two slot-major outputs). Only Q4_0 (the row-tiled
/// combo) is templated so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateUpBatchedDtype {
    Q4_0RowTile,
}

impl GateUpBatchedDtype {
    pub fn name(self) -> &'static str {
        match self {
            GateUpBatchedDtype::Q4_0RowTile => "Q4_0",
        }
    }
    fn weight_dtype(self) -> Dtype {
        match self {
            GateUpBatchedDtype::Q4_0RowTile => Dtype::Q4_0,
        }
    }
    fn impl_id(self) -> &'static str {
        match self {
            GateUpBatchedDtype::Q4_0RowTile => "qmatmul_q4_0_gate_up_row_tile_mmvq_batched",
        }
    }
    fn kernel_stem(self) -> &'static str {
        match self {
            GateUpBatchedDtype::Q4_0RowTile => "mmvq_q4_0_gate_up_row_tile_batched",
        }
    }
    fn kernel_entry(self, n_slots: usize) -> &'static str {
        match (self, n_slots) {
            (GateUpBatchedDtype::Q4_0RowTile, 2) => "flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n2",
            (GateUpBatchedDtype::Q4_0RowTile, 3) => "flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n3",
            (GateUpBatchedDtype::Q4_0RowTile, 4) => "flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n4",
            _ => unreachable!("gate_up batched n_slots outside [2,4]"),
        }
    }
    /// Wave64 rows/block (over-launches harmlessly on wave32's R=8).
    fn rows_per_block(self) -> u32 {
        4
    }
}

pub fn run_gate_up_batched_sweep<B: BenchBackend>(
    dtype: GateUpBatchedDtype,
    repo_root: &Path,
    arch: &str,
) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let module = B::load(&dev, bytes)?;
    let kernel = B::kernel(&module, dtype.kernel_entry(2))?;
    let pmc = B::attributes(&kernel)?;

    let m_grid = vec![1usize, 8, 16, 128, 512];
    let k_grid = vec![2048usize, 5120, 15360];
    let n_grid = vec![2usize, 3, 4];
    let seed = 0xC0FFEE_u64;

    let mut results = Vec::new();
    for &m in &m_grid {
        for &k in &k_grid {
            for &n in &n_grid {
                let s = seed
                    .wrapping_add((m as u64).wrapping_mul(0x1234567))
                    .wrapping_add((k as u64).wrapping_mul(0x9E3779B97F4A7C15))
                    .wrapping_add((n as u64).wrapping_mul(0x85EBCA6B));
                let (got, reference) = run_shape_gate_up_batched::<B>(&dev, dtype, m, k, n, s)?;
                let max_rel_err = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt());
                let tolerance = cert_tol(k);
                results.push(ShapeResult {
                    m,
                    k,
                    n,
                    seed: s,
                    max_rel_err,
                    tolerance,
                    pass: max_rel_err <= tolerance,
                });
            }
        }
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: dtype.impl_id().to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "qmatmul_gate_up_batched_mmvq".to_string(),
        dtype_weight: dtype.name().to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    let written = cert.write_to_disk(repo_root)?;
    tracing::info!(
        target: "flambeau_bench::sweep_mmvq",
        cert = %written.display(), pass = cert.pass, "gate_up batched cert written"
    );
    Ok(cert)
}

/// One (m, k, n_slots) fused gate+up batched shape (n_rows_gate = n_rows_up =
/// m). Returns `(got, reference)`, each the concatenation of the slot-major
/// gate output `[n_slots][m]` followed by the up output `[n_slots][m]`.
fn run_shape_gate_up_batched<B: BenchBackend>(
    dev: &B::Device,
    dtype: GateUpBatchedDtype,
    m: usize,
    k: usize,
    n_slots: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let q_bytes =
        B::image("quantize_q8_1").ok_or_else(|| anyhow!("quantize_q8_1 not compiled"))?;
    let m_bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let q_module = B::load(dev, q_bytes)?;
    let m_module = B::load(dev, m_bytes)?;
    let k_quantize = B::kernel(&q_module, "flambeau_quantize_row_q8_1")?;
    let k_mmvq = B::kernel(&m_module, dtype.kernel_entry(n_slots))?;

    let wd = dtype.weight_dtype();
    let block_elems = wd.block_elem_count();
    let block_bytes = wd.block_size_bytes();
    assert_eq!(k % block_elems, 0, "k {k} must be multiple of block size {block_elems}");
    let n_blocks = k / block_elems;

    let gate_raw = tame_scales(wd, seeded_bytes(seed, m * n_blocks * block_bytes));
    let up_raw = tame_scales(wd, seeded_bytes(seed ^ 0xDEAD_BEEF, m * n_blocks * block_bytes));
    let mut gate_dq = vec![0.0f32; m * k];
    let mut up_dq = vec![0.0f32; m * k];
    dequantize_into(wd.ggml(), &gate_raw, &mut gate_dq)?;
    dequantize_into(wd.ggml(), &up_raw, &mut up_dq)?;
    let d_gate = alloc_and_upload(dev, gate_raw.as_slice());
    let d_up = alloc_and_upload(dev, up_raw.as_slice());

    let q8_1_blocks_per_slot = k / QK8;
    let slot_bytes = q8_1_blocks_per_slot * std::mem::size_of::<BlockQ8_1>();
    let d_y_q8_1 = dev.alloc(n_slots * slot_bytes)?;
    let mut y_rt_slots: Vec<Vec<f32>> = Vec::with_capacity(n_slots);
    for s in 0..n_slots {
        let y_f32 = seeded_f32_range(seed.wrapping_add(0xA5A5 ^ (s as u64)), k, -1.0, 1.0);
        let d_y_f32 = alloc_and_upload(dev, y_f32.as_slice());
        {
            let stream = dev.default_stream();
            let n_elems = k as i32;
            let d_y_f32_ptr: u64 = d_y_f32.as_usize() as u64;
            let d_slot_ptr: u64 = (d_y_q8_1.as_usize() + s * slot_bytes) as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&d_y_f32_ptr);
            args.push_arg(&d_slot_ptr);
            args.push_arg(&n_elems);
            B::launch(&k_quantize, stream, LaunchCfg::one_d(q8_1_blocks_per_slot as u32, QK8 as u32), args)?;
            stream.synchronize()?;
        }
        unsafe { dev.dealloc(d_y_f32, y_f32.len() * 4)?; }
        y_rt_slots.push(quantize_q8_1_roundtrip(&y_f32));
    }

    let d_gate_out = dev.alloc(n_slots * m * 4)?;
    let d_up_out = dev.alloc(n_slots * m * 4)?;
    {
        let stream = dev.default_stream();
        let n_rows = m as i32;
        let n_units = n_blocks as i32;
        let gw_ptr: u64 = d_gate.as_usize() as u64;
        let uw_ptr: u64 = d_up.as_usize() as u64;
        let y_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let go_ptr: u64 = d_gate_out.as_usize() as u64;
        let uo_ptr: u64 = d_up_out.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&gw_ptr);
        args.push_arg(&uw_ptr);
        args.push_arg(&y_ptr);
        args.push_arg(&go_ptr);
        args.push_arg(&uo_ptr);
        args.push_arg(&n_rows);
        args.push_arg(&n_rows);
        args.push_arg(&n_units);
        let grid = (m as u32).div_ceil(dtype.rows_per_block());
        B::launch(&k_mmvq, stream, LaunchCfg::one_d(grid, 256), args)?;
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; 2 * n_slots * m];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_gate_out,
            n_slots * m * 4,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize + n_slots * m * 4),
            d_up_out,
            n_slots * m * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_gate, gate_raw.len())?;
        dev.dealloc(d_up, up_raw.len())?;
        dev.dealloc(d_y_q8_1, n_slots * slot_bytes)?;
        dev.dealloc(d_gate_out, n_slots * m * 4)?;
        dev.dealloc(d_up_out, n_slots * m * 4)?;
    }

    let mut reference = vec![0.0f32; 2 * n_slots * m];
    for s in 0..n_slots {
        let rg = reference_matmul(&gate_dq, &y_rt_slots[s], m, k);
        let ru = reference_matmul(&up_dq, &y_rt_slots[s], m, k);
        reference[s * m..(s + 1) * m].copy_from_slice(&rg);
        reference[n_slots * m + s * m..n_slots * m + (s + 1) * m].copy_from_slice(&ru);
    }
    Ok((got, reference))
}

// ---- MMQ oracle (batched dot, 2D grid) sweep ------------------------------

/// Weight dtype for the MMQ correctness-oracle sweep (`dst[batch,row] =
/// Σ_k x[row,k]·y[batch,k]`, one element per `{n_rows,n_batches}` grid block).
/// The cross-backend MMQ pilot — only Q8_0 so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmqOracleDtype {
    Q8_0,
}

impl MmqOracleDtype {
    pub fn name(self) -> &'static str {
        match self {
            MmqOracleDtype::Q8_0 => "Q8_0",
        }
    }
    fn weight_dtype(self) -> Dtype {
        match self {
            MmqOracleDtype::Q8_0 => Dtype::Q8_0Dp4a,
        }
    }
    fn impl_id(self) -> &'static str {
        match self {
            MmqOracleDtype::Q8_0 => "qmatmul_q8_0_mmq_oracle",
        }
    }
    fn kernel_stem(self) -> &'static str {
        match self {
            MmqOracleDtype::Q8_0 => "mmq_q8_0_oracle",
        }
    }
    fn kernel_entry(self) -> &'static str {
        match self {
            MmqOracleDtype::Q8_0 => "flambeau_mmq_q8_0_oracle_q8_1",
        }
    }
}

pub fn run_mmq_oracle_sweep<B: BenchBackend>(
    dtype: MmqOracleDtype,
    repo_root: &Path,
    arch: &str,
) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let module = B::load(&dev, bytes)?;
    let kernel = B::kernel(&module, dtype.kernel_entry())?;
    let pmc = B::attributes(&kernel)?;

    let m_grid = vec![1usize, 8, 16, 128, 512];
    let k_grid = vec![2048usize, 5120, 15360];
    let n_grid = vec![1usize, 2, 4];
    let seed = 0xC0FFEE_u64;

    let mut results = Vec::new();
    for &m in &m_grid {
        for &k in &k_grid {
            for &n in &n_grid {
                let s = seed
                    .wrapping_add((m as u64).wrapping_mul(0x1234567))
                    .wrapping_add((k as u64).wrapping_mul(0x9E3779B97F4A7C15))
                    .wrapping_add((n as u64).wrapping_mul(0xDEADBEEF));
                let (got, reference) = run_shape_mmq_oracle::<B>(&dev, dtype, m, k, n, s)?;
                let max_rel_err = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt());
                let tolerance = cert_tol(k);
                results.push(ShapeResult {
                    m,
                    k,
                    n,
                    seed: s,
                    max_rel_err,
                    tolerance,
                    pass: max_rel_err <= tolerance,
                });
            }
        }
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: dtype.impl_id().to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "qmatmul_mmq".to_string(),
        dtype_weight: dtype.name().to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    let written = cert.write_to_disk(repo_root)?;
    tracing::info!(
        target: "flambeau_bench::sweep_mmvq",
        cert = %written.display(), pass = cert.pass, "mmq oracle cert written"
    );
    Ok(cert)
}

/// One (m=n_rows, k, n=n_batches) oracle shape. Returns `(got, reference)`,
/// batch-major `[n_batches][n_rows]`.
fn run_shape_mmq_oracle<B: BenchBackend>(
    dev: &B::Device,
    dtype: MmqOracleDtype,
    m: usize,
    k: usize,
    n_batches: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let q_bytes =
        B::image("quantize_q8_1").ok_or_else(|| anyhow!("quantize_q8_1 not compiled"))?;
    let m_bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let q_module = B::load(dev, q_bytes)?;
    let m_module = B::load(dev, m_bytes)?;
    let k_quantize = B::kernel(&q_module, "flambeau_quantize_row_q8_1")?;
    let k_mmq = B::kernel(&m_module, dtype.kernel_entry())?;

    let wd = dtype.weight_dtype();
    let block_elems = wd.block_elem_count();
    let block_bytes = wd.block_size_bytes();
    assert_eq!(k % block_elems, 0, "k {k} must be multiple of block size {block_elems}");
    let n_blocks = k / block_elems;

    let weights_raw = tame_scales(wd, seeded_bytes(seed, m * n_blocks * block_bytes));
    let mut weights_dequant = vec![0.0f32; m * k];
    dequantize_into(wd.ggml(), &weights_raw, &mut weights_dequant)?;
    let d_x = alloc_and_upload(dev, weights_raw.as_slice());

    let q8_1_blocks_per_batch = k / QK8;
    let batch_bytes = q8_1_blocks_per_batch * std::mem::size_of::<BlockQ8_1>();
    let d_y_q8_1 = dev.alloc(n_batches * batch_bytes)?;
    let mut y_rt_batches: Vec<Vec<f32>> = Vec::with_capacity(n_batches);
    for bt in 0..n_batches {
        let y_f32 = seeded_f32_range(seed.wrapping_add(0xA5A5 ^ (bt as u64)), k, -1.0, 1.0);
        let d_y_f32 = alloc_and_upload(dev, y_f32.as_slice());
        {
            let stream = dev.default_stream();
            let n_elems = k as i32;
            let d_y_f32_ptr: u64 = d_y_f32.as_usize() as u64;
            let d_batch_ptr: u64 = (d_y_q8_1.as_usize() + bt * batch_bytes) as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&d_y_f32_ptr);
            args.push_arg(&d_batch_ptr);
            args.push_arg(&n_elems);
            B::launch(&k_quantize, stream, LaunchCfg::one_d(q8_1_blocks_per_batch as u32, QK8 as u32), args)?;
            stream.synchronize()?;
        }
        unsafe { dev.dealloc(d_y_f32, y_f32.len() * 4)?; }
        y_rt_batches.push(quantize_q8_1_roundtrip(&y_f32));
    }

    let d_dst = dev.alloc(n_batches * m * 4)?;
    {
        let stream = dev.default_stream();
        let n_rows = m as i32;
        let n_batch_i = n_batches as i32;
        let n_units = n_blocks as i32;
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_y_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let d_dst_ptr: u64 = d_dst.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_x_ptr);
        args.push_arg(&d_y_ptr);
        args.push_arg(&d_dst_ptr);
        args.push_arg(&n_rows);
        args.push_arg(&n_batch_i);
        args.push_arg(&n_units);
        let cfg = LaunchCfg { grid: (m as u32, n_batches as u32, 1), block: (256, 1, 1), shared_bytes: 0 };
        B::launch(&k_mmq, stream, cfg, args)?;
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_batches * m];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            n_batches * m * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_x, weights_raw.len())?;
        dev.dealloc(d_y_q8_1, n_batches * batch_bytes)?;
        dev.dealloc(d_dst, n_batches * m * 4)?;
    }

    let mut reference = vec![0.0f32; n_batches * m];
    for bt in 0..n_batches {
        let r = reference_matmul(&weights_dequant, &y_rt_batches[bt], m, k);
        reference[bt * m..(bt + 1) * m].copy_from_slice(&r);
    }
    Ok((got, reference))
}

// ---- MMQ tiled GEMM sweep -------------------------------------------------

/// Weight dtype for the single-pass tiled MMQ GEMM sweep (`dst[col,row] =
/// Σ_k x[row,k]·y[col,k]`, a `MMQ_Y×TILE_N` output tile per block).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmqGemmDtype {
    Q4_0Wave64,
    Q4_1Wave64,
    Q5_0Wave64,
    Q5_1Wave64,
    Q6KWave64,
    Q4KWave64,
    Q5KWave64,
    Q2KWave64,
    Q3KWave64,
    Q8KWave64,
    Iq4NlWave64,
    Iq4XsWave64,
    Iq2XxsWave64,
    Iq2XsWave64,
    Iq2SWave64,
    Iq3XxsWave64,
    Iq3SWave64,
    Iq1SWave64,
    Iq1MWave64,
    Q8_0Wave64Tile16,
}

impl MmqGemmDtype {
    pub fn name(self) -> &'static str {
        match self {
            MmqGemmDtype::Q4_0Wave64 => "Q4_0",
            MmqGemmDtype::Q4_1Wave64 => "Q4_1",
            MmqGemmDtype::Q5_0Wave64 => "Q5_0",
            MmqGemmDtype::Q5_1Wave64 => "Q5_1",
            MmqGemmDtype::Q6KWave64 => "Q6_K",
            MmqGemmDtype::Q4KWave64 => "Q4_K",
            MmqGemmDtype::Q5KWave64 => "Q5_K",
            MmqGemmDtype::Q2KWave64 => "Q2_K",
            MmqGemmDtype::Q3KWave64 => "Q3_K",
            MmqGemmDtype::Q8KWave64 => "Q8_K",
            MmqGemmDtype::Iq4NlWave64 => "IQ4_NL",
            MmqGemmDtype::Iq4XsWave64 => "IQ4_XS",
            MmqGemmDtype::Iq2XxsWave64 => "IQ2_XXS",
            MmqGemmDtype::Iq2XsWave64 => "IQ2_XS",
            MmqGemmDtype::Iq2SWave64 => "IQ2_S",
            MmqGemmDtype::Iq3XxsWave64 => "IQ3_XXS",
            MmqGemmDtype::Iq3SWave64 => "IQ3_S",
            MmqGemmDtype::Iq1SWave64 => "IQ1_S",
            MmqGemmDtype::Iq1MWave64 => "IQ1_M",
            MmqGemmDtype::Q8_0Wave64Tile16 => "Q8_0",
        }
    }
    fn weight_dtype(self) -> Dtype {
        match self {
            MmqGemmDtype::Q4_0Wave64 => Dtype::Q4_0,
            MmqGemmDtype::Q4_1Wave64 => Dtype::Q4_1Dp4a,
            MmqGemmDtype::Q5_0Wave64 => Dtype::Q5_0,
            MmqGemmDtype::Q5_1Wave64 => Dtype::Q5_1,
            MmqGemmDtype::Q6KWave64 => Dtype::Q6K,
            MmqGemmDtype::Q4KWave64 => Dtype::Q4K,
            MmqGemmDtype::Q5KWave64 => Dtype::Q5K,
            MmqGemmDtype::Q2KWave64 => Dtype::Q2K,
            MmqGemmDtype::Q3KWave64 => Dtype::Q3K,
            MmqGemmDtype::Q8KWave64 => Dtype::Q8K,
            MmqGemmDtype::Iq4NlWave64 => Dtype::Iq4Nl,
            MmqGemmDtype::Iq4XsWave64 => Dtype::Iq4Xs,
            MmqGemmDtype::Iq2XxsWave64 => Dtype::Iq2Xxs,
            MmqGemmDtype::Iq2XsWave64 => Dtype::Iq2Xs,
            MmqGemmDtype::Iq2SWave64 => Dtype::Iq2S,
            MmqGemmDtype::Iq3XxsWave64 => Dtype::Iq3Xxs,
            MmqGemmDtype::Iq3SWave64 => Dtype::Iq3S,
            MmqGemmDtype::Iq1SWave64 => Dtype::Iq1S,
            MmqGemmDtype::Iq1MWave64 => Dtype::Iq1M,
            MmqGemmDtype::Q8_0Wave64Tile16 => Dtype::Q8_0Dp4a,
        }
    }
    fn impl_id(self) -> &'static str {
        match self {
            MmqGemmDtype::Q4_0Wave64 => "qmatmul_q4_0_mmq_wave64",
            MmqGemmDtype::Q4_1Wave64 => "qmatmul_q4_1_mmq_wave64",
            MmqGemmDtype::Q5_0Wave64 => "qmatmul_q5_0_mmq_wave64",
            MmqGemmDtype::Q5_1Wave64 => "qmatmul_q5_1_mmq_wave64",
            MmqGemmDtype::Q6KWave64 => "qmatmul_q6_k_mmq_wave64",
            MmqGemmDtype::Q4KWave64 => "qmatmul_q4_k_mmq_wave64",
            MmqGemmDtype::Q5KWave64 => "qmatmul_q5_k_mmq_wave64",
            MmqGemmDtype::Q2KWave64 => "qmatmul_q2_k_mmq_wave64",
            MmqGemmDtype::Q3KWave64 => "qmatmul_q3_k_mmq_wave64",
            MmqGemmDtype::Q8KWave64 => "qmatmul_q8_k_mmq_wave64",
            MmqGemmDtype::Iq4NlWave64 => "qmatmul_iq4_nl_mmq_wave64",
            MmqGemmDtype::Iq4XsWave64 => "qmatmul_iq4_xs_mmq_wave64",
            MmqGemmDtype::Iq2XxsWave64 => "qmatmul_iq2_xxs_mmq_wave64",
            MmqGemmDtype::Iq2XsWave64 => "qmatmul_iq2_xs_mmq_wave64",
            MmqGemmDtype::Iq2SWave64 => "qmatmul_iq2_s_mmq_wave64",
            MmqGemmDtype::Iq3XxsWave64 => "qmatmul_iq3_xxs_mmq_wave64",
            MmqGemmDtype::Iq3SWave64 => "qmatmul_iq3_s_mmq_wave64",
            MmqGemmDtype::Iq1SWave64 => "qmatmul_iq1_s_mmq_wave64",
            MmqGemmDtype::Iq1MWave64 => "qmatmul_iq1_m_mmq_wave64",
            MmqGemmDtype::Q8_0Wave64Tile16 => "qmatmul_q8_0_mmq_wave64_tile16",
        }
    }
    fn kernel_stem(self) -> &'static str {
        match self {
            MmqGemmDtype::Q4_0Wave64 => "mmq_q4_0_wave64",
            MmqGemmDtype::Q4_1Wave64 => "mmq_q4_1_wave64",
            MmqGemmDtype::Q5_0Wave64 => "mmq_q5_0_wave64",
            MmqGemmDtype::Q5_1Wave64 => "mmq_q5_1_wave64",
            MmqGemmDtype::Q6KWave64 => "mmq_q6_K_wave64",
            MmqGemmDtype::Q4KWave64 => "mmq_q4_K_wave64",
            MmqGemmDtype::Q5KWave64 => "mmq_q5_K_wave64",
            MmqGemmDtype::Q2KWave64 => "mmq_q2_K_wave64",
            MmqGemmDtype::Q3KWave64 => "mmq_q3_K_wave64",
            MmqGemmDtype::Q8KWave64 => "mmq_q8_K_wave64",
            MmqGemmDtype::Iq4NlWave64 => "mmq_iq4_nl_wave64",
            MmqGemmDtype::Iq4XsWave64 => "mmq_iq4_xs_wave64",
            MmqGemmDtype::Iq2XxsWave64 => "mmq_iq2_xxs_wave64",
            MmqGemmDtype::Iq2XsWave64 => "mmq_iq2_xs_wave64",
            MmqGemmDtype::Iq2SWave64 => "mmq_iq2_s_wave64",
            MmqGemmDtype::Iq3XxsWave64 => "mmq_iq3_xxs_wave64",
            MmqGemmDtype::Iq3SWave64 => "mmq_iq3_s_wave64",
            MmqGemmDtype::Iq1SWave64 => "mmq_iq1_s_wave64",
            MmqGemmDtype::Iq1MWave64 => "mmq_iq1_m_wave64",
            MmqGemmDtype::Q8_0Wave64Tile16 => "mmq_q8_0_wave64_tile16",
        }
    }
    fn kernel_entry(self) -> &'static str {
        match self {
            MmqGemmDtype::Q4_0Wave64 => "flambeau_mmq_q4_0_wave64_q8_1",
            MmqGemmDtype::Q4_1Wave64 => "flambeau_mmq_q4_1_wave64_q8_1",
            MmqGemmDtype::Q5_0Wave64 => "flambeau_mmq_q5_0_wave64_q8_1",
            MmqGemmDtype::Q5_1Wave64 => "flambeau_mmq_q5_1_wave64_q8_1",
            MmqGemmDtype::Q6KWave64 => "flambeau_mmq_q6_K_wave64_q8_1",
            MmqGemmDtype::Q4KWave64 => "flambeau_mmq_q4_K_wave64_q8_1",
            MmqGemmDtype::Q5KWave64 => "flambeau_mmq_q5_K_wave64_q8_1",
            MmqGemmDtype::Q2KWave64 => "flambeau_mmq_q2_K_wave64_q8_1",
            MmqGemmDtype::Q3KWave64 => "flambeau_mmq_q3_K_wave64_q8_1",
            MmqGemmDtype::Q8KWave64 => "flambeau_mmq_q8_K_wave64_q8_1",
            MmqGemmDtype::Iq4NlWave64 => "flambeau_mmq_iq4_nl_wave64_q8_1",
            MmqGemmDtype::Iq4XsWave64 => "flambeau_mmq_iq4_xs_wave64_q8_1",
            MmqGemmDtype::Iq2XxsWave64 => "flambeau_mmq_iq2_xxs_wave64_q8_1",
            MmqGemmDtype::Iq2XsWave64 => "flambeau_mmq_iq2_xs_wave64_q8_1",
            MmqGemmDtype::Iq2SWave64 => "flambeau_mmq_iq2_s_wave64_q8_1",
            MmqGemmDtype::Iq3XxsWave64 => "flambeau_mmq_iq3_xxs_wave64_q8_1",
            MmqGemmDtype::Iq3SWave64 => "flambeau_mmq_iq3_s_wave64_q8_1",
            MmqGemmDtype::Iq1SWave64 => "flambeau_mmq_iq1_s_wave64_q8_1",
            MmqGemmDtype::Iq1MWave64 => "flambeau_mmq_iq1_m_wave64_q8_1",
            MmqGemmDtype::Q8_0Wave64Tile16 => "flambeau_mmq_q8_0_wave64_tile16_q8_1",
        }
    }
    /// (MMQ_Y rows/block, TILE_N cols/block) — the grid tiling. Block thread
    /// count equals MMQ_Y (each single-pass wave64 thread owns one weight row).
    fn tile(self) -> (u32, u32) {
        match self {
            MmqGemmDtype::Q8_0Wave64Tile16 => (64, 16),
            _ => (64, 8),
        }
    }
}

pub fn run_mmq_gemm_sweep<B: BenchBackend>(
    dtype: MmqGemmDtype,
    repo_root: &Path,
    arch: &str,
) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let module = B::load(&dev, bytes)?;
    let kernel = B::kernel(&module, dtype.kernel_entry())?;
    let pmc = B::attributes(&kernel)?;

    // (n_rows = weight rows / output dim, n_cols = prefill token batch, k).
    let rows_grid = vec![64usize, 256, 4096];
    let cols_grid = vec![1usize, 8, 16];
    let k_grid = vec![2048usize, 4096];
    let seed = 0xC0FFEE_u64;

    let mut results = Vec::new();
    for &n_rows in &rows_grid {
        for &n_cols in &cols_grid {
            for &k in &k_grid {
                let s = seed
                    .wrapping_add((n_rows as u64).wrapping_mul(0x1234567))
                    .wrapping_add((n_cols as u64).wrapping_mul(0x85EBCA6B))
                    .wrapping_add((k as u64).wrapping_mul(0x9E3779B97F4A7C15));
                let (got, reference) = run_shape_mmq_gemm::<B>(&dev, dtype, n_rows, n_cols, k, s)?;
                let max_rel_err = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt());
                let tolerance = cert_tol(k);
                results.push(ShapeResult {
                    m: n_rows,
                    k,
                    n: n_cols,
                    seed: s,
                    max_rel_err,
                    tolerance,
                    pass: max_rel_err <= tolerance,
                });
            }
        }
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: dtype.impl_id().to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "qmatmul_mmq".to_string(),
        dtype_weight: dtype.name().to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    let written = cert.write_to_disk(repo_root)?;
    tracing::info!(
        target: "flambeau_bench::sweep_mmvq",
        cert = %written.display(), pass = cert.pass, "mmq gemm cert written"
    );
    Ok(cert)
}

/// One (n_rows, n_cols, k) tiled MMQ GEMM shape. Weight `[n_rows][k]` Q8_0,
/// activation `[n_cols][k]` Q8_1, output `dst[col*n_rows + row]`. Returns
/// `(got, reference)`.
fn run_shape_mmq_gemm<B: BenchBackend>(
    dev: &B::Device,
    dtype: MmqGemmDtype,
    n_rows: usize,
    n_cols: usize,
    k: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let q_bytes =
        B::image("quantize_q8_1").ok_or_else(|| anyhow!("quantize_q8_1 not compiled"))?;
    let m_bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let q_module = B::load(dev, q_bytes)?;
    let m_module = B::load(dev, m_bytes)?;
    let k_quantize = B::kernel(&q_module, "flambeau_quantize_row_q8_1")?;
    let k_mmq = B::kernel(&m_module, dtype.kernel_entry())?;

    let wd = dtype.weight_dtype();
    let block_elems = wd.block_elem_count();
    let block_bytes = wd.block_size_bytes();
    assert_eq!(k % block_elems, 0, "k {k} must be multiple of block size {block_elems}");
    let n_blocks = k / block_elems;

    let weights_raw = tame_scales(wd, seeded_bytes(seed, n_rows * n_blocks * block_bytes));
    let mut w_dequant = vec![0.0f32; n_rows * k];
    dequantize_into(wd.ggml(), &weights_raw, &mut w_dequant)?;
    let d_x = alloc_and_upload(dev, weights_raw.as_slice());

    // Activation: n_cols rows of k, quantized flat → [n_cols][k/QK8] Q8_1 blocks.
    let act_f32 = seeded_f32_range(seed.wrapping_add(0xA5A5A5A5), n_cols * k, -1.0, 1.0);
    let d_act = alloc_and_upload(dev, act_f32.as_slice());
    let total_y_blocks = n_cols * k / QK8;
    let y_bytes = total_y_blocks * std::mem::size_of::<BlockQ8_1>();
    let d_y_q8_1 = dev.alloc(y_bytes)?;
    {
        let stream = dev.default_stream();
        let n_elems = (n_cols * k) as i32;
        let d_act_ptr: u64 = d_act.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_act_ptr);
        args.push_arg(&d_y_q8_1_ptr);
        args.push_arg(&n_elems);
        B::launch(&k_quantize, stream, LaunchCfg::one_d(total_y_blocks as u32, QK8 as u32), args)?;
        stream.synchronize()?;
    }

    let d_dst = dev.alloc(n_cols * n_rows * 4)?;
    {
        let stream = dev.default_stream();
        let ncols_x = k as i32;
        let nrows_x = n_rows as i32;
        let ncols_y = n_cols as i32;
        let nrows_y = k as i32;
        let nrows_dst = n_rows as i32;
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_y_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let d_dst_ptr: u64 = d_dst.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_x_ptr);
        args.push_arg(&d_y_ptr);
        args.push_arg(&d_dst_ptr);
        args.push_arg(&ncols_x);
        args.push_arg(&nrows_x);
        args.push_arg(&ncols_y);
        args.push_arg(&nrows_y);
        args.push_arg(&nrows_dst);
        let (mmq_y, tile_n) = dtype.tile();
        let cfg = LaunchCfg {
            grid: ((n_rows as u32).div_ceil(mmq_y), (n_cols as u32).div_ceil(tile_n), 1),
            block: (mmq_y, 1, 1),
            shared_bytes: 0,
        };
        B::launch(&k_mmq, stream, cfg, args)?;
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_cols * n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            n_cols * n_rows * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_x, weights_raw.len())?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y_q8_1, y_bytes)?;
        dev.dealloc(d_dst, n_cols * n_rows * 4)?;
    }

    // Reference: dst[col*n_rows + row] = w_dequant[row] · y_rt[col].
    let mut reference = vec![0.0f32; n_cols * n_rows];
    for c in 0..n_cols {
        let y_rt = quantize_q8_1_roundtrip(&act_f32[c * k..(c + 1) * k]);
        let col_out = reference_matmul(&w_dequant, &y_rt, n_rows, k);
        for row in 0..n_rows {
            reference[c * n_rows + row] = col_out[row];
        }
    }
    Ok((got, reference))
}

/// Indexed-MoE MMVQ kernels (decode GEMV with per-token expert routing). Same
/// inner product as the dense mmvq family but with the 8-arg MoE ABI
/// `(w, y, expert_ids, dst, n_rows, n_tokens, top_k, n_blocks_per_row)` and a
/// (n_rows, n_tokens*top_k, 1) grid — so a distinct harness from `MmqGemmDtype`.
#[derive(Clone, Copy)]
pub enum MoeMmvqDtype {
    Q8_0,
    Q4_0,
    Q4_1,
    Q4KR2,
    Q4KR4,
    Q5KR2,
    Q6KR2,
    Q3KR2,
    Iq4NlR2,
    Iq4XsR2,
    Iq2XxsR2,
    Iq2XsR2,
    Iq2SR2,
    Iq3XxsR2,
    Iq3SR2,
    Iq1SR2,
    Iq1MR2,
}

/// Qwen3.6-35B-A3B expert projections, `(n_rows, k)`: gate / up
/// (moe_intermediate × hidden) and down (hidden × moe_intermediate).
const MOE_35B_GATE_UP: (usize, usize) = (768, 2048);
const MOE_35B_DOWN: (usize, usize) = (2048, 768);
const MOE_35B_TOP_K: usize = 8;
const MOE_35B_EXPERTS: usize = 256;

/// `(n_tokens, top_k, n_rows, k)` for every `m` of `m_grid` over `shapes`.
fn moe_cases(m_grid: &[usize], shapes: &[(usize, usize)]) -> Vec<(usize, usize, usize, usize)> {
    m_grid
        .iter()
        .flat_map(|&m| shapes.iter().map(move |&(n_rows, k)| (m, MOE_35B_TOP_K, n_rows, k)))
        .collect()
}

impl MoeMmvqDtype {
    /// Sweep shapes. The Q4_K r2 / r4 kernels cover the token range of
    /// their gfx906 dispatch rows (r2: m = 1..31, r4: m >= 32).
    fn cases(self) -> Vec<(usize, usize, usize, usize)> {
        match self {
            MoeMmvqDtype::Q4KR2 => moe_cases(&[1, 8, 31], &[MOE_35B_GATE_UP, MOE_35B_DOWN]),
            MoeMmvqDtype::Q4KR4 => moe_cases(&[32, 64, 128], &[MOE_35B_GATE_UP, MOE_35B_DOWN]),
            _ => vec![(1, 4, 256, 2048), (4, 4, 256, 2048), (1, 8, 2048, 768)],
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            MoeMmvqDtype::Q8_0 => "Q8_0",
            MoeMmvqDtype::Q4_0 => "Q4_0",
            MoeMmvqDtype::Q4_1 => "Q4_1",
            MoeMmvqDtype::Q4KR2 | MoeMmvqDtype::Q4KR4 => "Q4_K",
            MoeMmvqDtype::Q5KR2 => "Q5_K",
            MoeMmvqDtype::Q6KR2 => "Q6_K",
            MoeMmvqDtype::Q3KR2 => "Q3_K",
            MoeMmvqDtype::Iq4NlR2 => "IQ4_NL",
            MoeMmvqDtype::Iq4XsR2 => "IQ4_XS",
            MoeMmvqDtype::Iq2XxsR2 => "IQ2_XXS",
            MoeMmvqDtype::Iq2XsR2 => "IQ2_XS",
            MoeMmvqDtype::Iq2SR2 => "IQ2_S",
            MoeMmvqDtype::Iq3XxsR2 => "IQ3_XXS",
            MoeMmvqDtype::Iq3SR2 => "IQ3_S",
            MoeMmvqDtype::Iq1SR2 => "IQ1_S",
            MoeMmvqDtype::Iq1MR2 => "IQ1_M",
        }
    }
    fn weight_dtype(self) -> Dtype {
        match self {
            MoeMmvqDtype::Q8_0 => Dtype::Q8_0Dp4a,
            MoeMmvqDtype::Q4_0 => Dtype::Q4_0,
            MoeMmvqDtype::Q4_1 => Dtype::Q4_1Dp4a,
            MoeMmvqDtype::Q4KR2 | MoeMmvqDtype::Q4KR4 => Dtype::Q4K,
            MoeMmvqDtype::Q5KR2 => Dtype::Q5K,
            MoeMmvqDtype::Q6KR2 => Dtype::Q6K,
            MoeMmvqDtype::Q3KR2 => Dtype::Q3K,
            MoeMmvqDtype::Iq4NlR2 => Dtype::Iq4Nl,
            MoeMmvqDtype::Iq4XsR2 => Dtype::Iq4Xs,
            MoeMmvqDtype::Iq2XxsR2 => Dtype::Iq2Xxs,
            MoeMmvqDtype::Iq2XsR2 => Dtype::Iq2Xs,
            MoeMmvqDtype::Iq2SR2 => Dtype::Iq2S,
            MoeMmvqDtype::Iq3XxsR2 => Dtype::Iq3Xxs,
            MoeMmvqDtype::Iq3SR2 => Dtype::Iq3S,
            MoeMmvqDtype::Iq1SR2 => Dtype::Iq1S,
            MoeMmvqDtype::Iq1MR2 => Dtype::Iq1M,
        }
    }
    fn impl_id(self) -> &'static str {
        match self {
            MoeMmvqDtype::Q8_0 => "indexed_moe_mmvq_q8_0",
            MoeMmvqDtype::Q4_0 => "indexed_moe_mmvq_q4_0",
            MoeMmvqDtype::Q4_1 => "indexed_moe_mmvq_q4_1",
            MoeMmvqDtype::Q4KR2 => "indexed_moe_mmvq_q4_k_r2",
            MoeMmvqDtype::Q4KR4 => "indexed_moe_mmvq_q4_k_r4",
            MoeMmvqDtype::Q5KR2 => "indexed_moe_mmvq_q5_k_r2",
            MoeMmvqDtype::Q6KR2 => "indexed_moe_mmvq_q6_k_r2",
            MoeMmvqDtype::Q3KR2 => "indexed_moe_mmvq_q3_k_r2",
            MoeMmvqDtype::Iq4NlR2 => "indexed_moe_mmvq_iq4_nl_r2",
            MoeMmvqDtype::Iq4XsR2 => "indexed_moe_mmvq_iq4_xs_r2",
            MoeMmvqDtype::Iq2XxsR2 => "indexed_moe_mmvq_iq2_xxs_r2",
            MoeMmvqDtype::Iq2XsR2 => "indexed_moe_mmvq_iq2_xs_r2",
            MoeMmvqDtype::Iq2SR2 => "indexed_moe_mmvq_iq2_s_r2",
            MoeMmvqDtype::Iq3XxsR2 => "indexed_moe_mmvq_iq3_xxs_r2",
            MoeMmvqDtype::Iq3SR2 => "indexed_moe_mmvq_iq3_s_r2",
            MoeMmvqDtype::Iq1SR2 => "indexed_moe_mmvq_iq1_s_r2",
            MoeMmvqDtype::Iq1MR2 => "indexed_moe_mmvq_iq1_m_r2",
        }
    }
    fn kernel_stem(self) -> &'static str {
        match self {
            MoeMmvqDtype::Q8_0 => "indexed_moe_mmvq_q8_0",
            MoeMmvqDtype::Q4_0 => "indexed_moe_mmvq_q4_0",
            MoeMmvqDtype::Q4_1 => "indexed_moe_mmvq_q4_1",
            MoeMmvqDtype::Q4KR2 => "indexed_moe_mmvq_q4_k_r2_dp4a",
            MoeMmvqDtype::Q4KR4 => "indexed_moe_mmvq_q4_k_r4_dp4a",
            MoeMmvqDtype::Q5KR2 => "indexed_moe_mmvq_q5_k_r2_dp4a",
            MoeMmvqDtype::Q6KR2 => "indexed_moe_mmvq_q6_k_r2_dp4a",
            MoeMmvqDtype::Q3KR2 => "indexed_moe_mmvq_q3_k_r2_dp4a",
            MoeMmvqDtype::Iq4NlR2 => "indexed_moe_mmvq_iq4_nl_r2_dp4a",
            MoeMmvqDtype::Iq4XsR2 => "indexed_moe_mmvq_iq4_xs_r2_dp4a",
            MoeMmvqDtype::Iq2XxsR2 => "indexed_moe_mmvq_iq2_xxs_r2_dp4a",
            MoeMmvqDtype::Iq2XsR2 => "indexed_moe_mmvq_iq2_xs_r2_dp4a",
            MoeMmvqDtype::Iq2SR2 => "indexed_moe_mmvq_iq2_s_r2_dp4a",
            MoeMmvqDtype::Iq3XxsR2 => "indexed_moe_mmvq_iq3_xxs_r2_dp4a",
            MoeMmvqDtype::Iq3SR2 => "indexed_moe_mmvq_iq3_s_r2_dp4a",
            MoeMmvqDtype::Iq1SR2 => "indexed_moe_mmvq_iq1_s_r2_dp4a",
            MoeMmvqDtype::Iq1MR2 => "indexed_moe_mmvq_iq1_m_r2_dp4a",
        }
    }
    fn kernel_entry(self) -> &'static str {
        match self {
            MoeMmvqDtype::Q8_0 => "flambeau_indexed_moe_mmvq_q8_0_dp4a_q8_1",
            MoeMmvqDtype::Q4_0 => "flambeau_indexed_moe_mmvq_q4_0_q8_1",
            MoeMmvqDtype::Q4_1 => "flambeau_indexed_moe_mmvq_q4_1_q8_1",
            MoeMmvqDtype::Q4KR2 => "flambeau_indexed_moe_mmvq_q4_k_r2_dp4a_q8_1",
            MoeMmvqDtype::Q4KR4 => "flambeau_indexed_moe_mmvq_q4_k_r4_dp4a_q8_1",
            MoeMmvqDtype::Q5KR2 => "flambeau_indexed_moe_mmvq_q5_k_r2_dp4a_q8_1",
            MoeMmvqDtype::Q6KR2 => "flambeau_indexed_moe_mmvq_q6_k_r2_dp4a_q8_1",
            MoeMmvqDtype::Q3KR2 => "flambeau_indexed_moe_mmvq_q3_k_r2_dp4a_q8_1",
            MoeMmvqDtype::Iq4NlR2 => "flambeau_indexed_moe_mmvq_iq4_nl_r2_dp4a_q8_1",
            MoeMmvqDtype::Iq4XsR2 => "flambeau_indexed_moe_mmvq_iq4_xs_r2_dp4a_q8_1",
            MoeMmvqDtype::Iq2XxsR2 => "flambeau_indexed_moe_mmvq_iq2_xxs_r2_dp4a_q8_1",
            MoeMmvqDtype::Iq2XsR2 => "flambeau_indexed_moe_mmvq_iq2_xs_r2_dp4a_q8_1",
            MoeMmvqDtype::Iq2SR2 => "flambeau_indexed_moe_mmvq_iq2_s_r2_dp4a_q8_1",
            MoeMmvqDtype::Iq3XxsR2 => "flambeau_indexed_moe_mmvq_iq3_xxs_r2_dp4a_q8_1",
            MoeMmvqDtype::Iq3SR2 => "flambeau_indexed_moe_mmvq_iq3_s_r2_dp4a_q8_1",
            MoeMmvqDtype::Iq1SR2 => "flambeau_indexed_moe_mmvq_iq1_s_r2_dp4a_q8_1",
            MoeMmvqDtype::Iq1MR2 => "flambeau_indexed_moe_mmvq_iq1_m_r2_dp4a_q8_1",
        }
    }
    /// (threads/block, rows/block). Plain per-row kernels own one output row per
    /// block (256t); the r2/r4 K-quant kernels are one wave64 (64t) computing
    /// 2 / 4 output rows, shrinking grid.x by that factor.
    fn launch(self) -> (u32, u32) {
        match self {
            MoeMmvqDtype::Q8_0
            | MoeMmvqDtype::Q4_0
            | MoeMmvqDtype::Q4_1 => (256, 1),
            MoeMmvqDtype::Q4KR2
            | MoeMmvqDtype::Q5KR2
            | MoeMmvqDtype::Q6KR2
            | MoeMmvqDtype::Q3KR2
            | MoeMmvqDtype::Iq4NlR2
            | MoeMmvqDtype::Iq4XsR2
            | MoeMmvqDtype::Iq2XxsR2
            | MoeMmvqDtype::Iq2XsR2
            | MoeMmvqDtype::Iq2SR2
            | MoeMmvqDtype::Iq3XxsR2
            | MoeMmvqDtype::Iq3SR2
            | MoeMmvqDtype::Iq1SR2
            | MoeMmvqDtype::Iq1MR2 => (64, 2),
            MoeMmvqDtype::Q4KR4 => (64, 4),
        }
    }
}

pub fn run_moe_mmvq_sweep<B: BenchBackend>(
    dtype: MoeMmvqDtype,
    repo_root: &Path,
    arch: &str,
) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let pmc = {
        let bytes = B::image(dtype.kernel_stem())
            .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
        let module = B::load(&dev, bytes)?;
        let kernel = B::kernel(&module, dtype.kernel_entry())?;
        B::attributes(&kernel)?
    };

    // (n_tokens, top_k, n_rows = expert output dim, k = hidden).
    let n_experts = MOE_35B_EXPERTS;
    let cases = dtype.cases();

    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k) in cases {
        let seed = 0xDEC0DE_u64
            .wrapping_add((n_tokens as u64).wrapping_mul(53))
            .wrapping_add((top_k as u64).wrapping_mul(17))
            .wrapping_add((n_rows as u64).wrapping_mul(7));
        let (got, reference) =
            run_shape_moe_mmvq::<B>(&dev, dtype, n_experts, n_rows, k, top_k, n_tokens, seed)?;
        let max_rel_err = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt());
        let tolerance = cert_tol(k);
        results.push(ShapeResult {
            m: n_tokens,
            k,
            n: n_rows,
            seed,
            max_rel_err,
            tolerance,
            pass: max_rel_err <= tolerance,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: dtype.impl_id().to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "indexed_moe_mmvq".to_string(),
        dtype_weight: dtype.name().to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))  (indexed-MoE MMVQ)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    let written = cert.write_to_disk(repo_root)?;
    tracing::info!(
        target: "flambeau_bench::sweep_mmvq",
        cert = %written.display(), pass = cert.pass, "indexed-moe mmvq cert written"
    );
    Ok(cert)
}

/// Router ids for `n_pairs` (token, slot) pairs: a deterministic hash over
/// `[0, n_experts)`, with the first pair on expert 0 and the last on the
/// highest expert so both ends of the expert-offset range are exercised.
fn moe_expert_ids(n_pairs: usize, n_experts: usize, seed: u64) -> Vec<i32> {
    let mut ids: Vec<i32> = (0..n_pairs)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();
    if let Some(first) = ids.first_mut() {
        *first = 0;
    }
    if n_pairs > 1 {
        ids[n_pairs - 1] = n_experts as i32 - 1;
    }
    ids
}

/// Dequantised `[n_rows][k]` weights of each expert `expert_ids` selects;
/// `raw` holds every expert, `expert_bytes` apart.
fn dequant_selected_experts(
    wd: Dtype,
    raw: &[u8],
    expert_ids: &[i32],
    expert_bytes: usize,
    expert_elems: usize,
) -> Result<std::collections::BTreeMap<usize, Vec<f32>>> {
    let mut out = std::collections::BTreeMap::new();
    for &e in expert_ids {
        let e = e as usize;
        if out.contains_key(&e) {
            continue;
        }
        let mut dq = vec![0.0f32; expert_elems];
        dequantize_into(wd.ggml(), &raw[e * expert_bytes..(e + 1) * expert_bytes], &mut dq)?;
        out.insert(e, dq);
    }
    Ok(out)
}

/// One indexed-MoE MMVQ shape. Weight `[n_experts][n_rows][k]`, per-token
/// activation `[n_tokens][k]` Q8_1, router-selected `expert_ids[n_tokens][top_k]`,
/// output `dst[token, slot, row]`. Returns `(got, reference)`.
#[expect(clippy::too_many_arguments, reason = "MoE shape carries 7 independent dims")]
fn run_shape_moe_mmvq<B: BenchBackend>(
    dev: &B::Device,
    dtype: MoeMmvqDtype,
    n_experts: usize,
    n_rows: usize,
    k: usize,
    top_k: usize,
    n_tokens: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let q_bytes =
        B::image("quantize_q8_1").ok_or_else(|| anyhow!("quantize_q8_1 not compiled"))?;
    let m_bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let q_module = B::load(dev, q_bytes)?;
    let m_module = B::load(dev, m_bytes)?;
    let k_quantize = B::kernel(&q_module, "flambeau_quantize_row_q8_1")?;
    let k_mmvq = B::kernel(&m_module, dtype.kernel_entry())?;

    let wd = dtype.weight_dtype();
    let block_elems = wd.block_elem_count();
    let block_bytes = wd.block_size_bytes();
    assert_eq!(k % block_elems, 0, "k {k} must be multiple of block size {block_elems}");
    let n_blocks = k / block_elems;

    // Expert weights [n_experts][n_rows][k].
    let expert_bytes = n_rows * n_blocks * block_bytes;
    let weights_raw = tame_scales(wd, seeded_bytes(seed, n_experts * expert_bytes));
    let d_w = alloc_and_upload(dev, weights_raw.as_slice());

    // Per-token activation [n_tokens][k] → token-major Q8_1 [n_tokens][n_blocks].
    let act_f32 = seeded_f32_range(seed.wrapping_add(0xA5A5A5A5), n_tokens * k, -1.0, 1.0);
    let d_act = alloc_and_upload(dev, act_f32.as_slice());
    let total_y_blocks = n_tokens * k / QK8;
    let y_bytes = total_y_blocks * std::mem::size_of::<BlockQ8_1>();
    let d_y = dev.alloc(y_bytes)?;
    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k) as i32;
        let d_act_ptr: u64 = d_act.as_usize() as u64;
        let d_y_ptr: u64 = d_y.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_act_ptr);
        args.push_arg(&d_y_ptr);
        args.push_arg(&n_elems);
        B::launch(&k_quantize, stream, LaunchCfg::one_d(total_y_blocks as u32, QK8 as u32), args)?;
        stream.synchronize()?;
    }

    let expert_ids = moe_expert_ids(n_tokens * top_k, n_experts, seed);
    let d_ids = alloc_and_upload(dev, expert_ids.as_slice());
    let w_dequant =
        dequant_selected_experts(wd, &weights_raw, &expert_ids, expert_bytes, n_rows * k)?;

    let d_dst = dev.alloc(n_tokens * top_k * n_rows * 4)?;
    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_tokens_i = n_tokens as i32;
        let top_k_i = top_k as i32;
        let nb_i = n_blocks as i32;
        let d_w_ptr: u64 = d_w.as_usize() as u64;
        let d_y_ptr: u64 = d_y.as_usize() as u64;
        let d_ids_ptr: u64 = d_ids.as_usize() as u64;
        let d_dst_ptr: u64 = d_dst.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_w_ptr);
        args.push_arg(&d_y_ptr);
        args.push_arg(&d_ids_ptr);
        args.push_arg(&d_dst_ptr);
        args.push_arg(&n_rows_i);
        args.push_arg(&n_tokens_i);
        args.push_arg(&top_k_i);
        args.push_arg(&nb_i);
        let (threads, rows_per_block) = dtype.launch();
        let cfg = LaunchCfg {
            grid: ((n_rows as u32).div_ceil(rows_per_block), (n_tokens * top_k) as u32, 1),
            block: (threads, 1, 1),
            shared_bytes: 0,
        };
        B::launch(&k_mmvq, stream, cfg, args)?;
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_tokens * top_k * n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            n_tokens * top_k * n_rows * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_w, weights_raw.len())?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_bytes)?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_dst, n_tokens * top_k * n_rows * 4)?;
    }

    // Reference: per (token, slot, row) dot the selected expert's weight row
    // with the Q8_1-roundtripped token activation.
    let mut reference = vec![0.0f32; n_tokens * top_k * n_rows];
    for t in 0..n_tokens {
        let y_rt = quantize_q8_1_roundtrip(&act_f32[t * k..(t + 1) * k]);
        for slot in 0..top_k {
            let expert = expert_ids[t * top_k + slot] as usize;
            let col_out = reference_matmul(&w_dequant[&expert], &y_rt, n_rows, k);
            for row in 0..n_rows {
                reference[(t * top_k + slot) * n_rows + row] = col_out[row];
            }
        }
    }
    Ok((got, reference))
}

/// Fused gate+up indexed-MoE MMVQ kernels (10-arg ABI: gate_w, up_w, y,
/// expert_ids, gate_out, up_out, n_rows, n_tokens, top_k, n_blocks_per_row).
#[derive(Clone, Copy)]
pub enum MoeGateUpDtype {
    Q8_0,
    Q4_0,
    Q4KR4,
}

impl MoeGateUpDtype {
    /// Sweep shapes. Q4_K r4 serves every token count (its gfx906 dispatch
    /// row is m = any), so it covers the decode and prefill range.
    fn cases(self) -> Vec<(usize, usize, usize, usize)> {
        match self {
            MoeGateUpDtype::Q4KR4 => moe_cases(&[1, 8, 31, 32, 64, 128], &[MOE_35B_GATE_UP]),
            _ => vec![(1, 4, 256, 2048), (4, 4, 256, 2048)],
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            MoeGateUpDtype::Q8_0 => "Q8_0",
            MoeGateUpDtype::Q4_0 => "Q4_0",
            MoeGateUpDtype::Q4KR4 => "Q4_K",
        }
    }
    fn weight_dtype(self) -> Dtype {
        match self {
            MoeGateUpDtype::Q8_0 => Dtype::Q8_0Dp4a,
            MoeGateUpDtype::Q4_0 => Dtype::Q4_0,
            MoeGateUpDtype::Q4KR4 => Dtype::Q4K,
        }
    }
    fn impl_id(self) -> &'static str {
        match self {
            MoeGateUpDtype::Q8_0 => "indexed_moe_mmvq_q8_0_gate_up",
            MoeGateUpDtype::Q4_0 => "indexed_moe_mmvq_q4_0_gate_up",
            MoeGateUpDtype::Q4KR4 => "indexed_moe_mmvq_q4_k_gate_up_r4",
        }
    }
    fn kernel_stem(self) -> &'static str {
        match self {
            MoeGateUpDtype::Q8_0 => "indexed_moe_mmvq_q8_0_gate_up_dp4a",
            MoeGateUpDtype::Q4_0 => "indexed_moe_mmvq_q4_0_gate_up_dp4a",
            MoeGateUpDtype::Q4KR4 => "indexed_moe_mmvq_q4_k_gate_up_r4_dp4a",
        }
    }
    fn kernel_entry(self) -> &'static str {
        match self {
            MoeGateUpDtype::Q8_0 => "flambeau_indexed_moe_mmvq_q8_0_gate_up_dp4a_q8_1",
            MoeGateUpDtype::Q4_0 => "flambeau_indexed_moe_mmvq_q4_0_gate_up_dp4a_q8_1",
            MoeGateUpDtype::Q4KR4 => "flambeau_indexed_moe_mmvq_q4_k_gate_up_r4_dp4a_q8_1",
        }
    }
    /// Block thread count. The dp4a variants are 256t (4 lanes/block); the Q4_K
    /// scalar + r4 kernels are one wave64 (64t).
    fn launch_threads(self) -> u32 {
        match self {
            MoeGateUpDtype::Q8_0 | MoeGateUpDtype::Q4_0 => 256,
            MoeGateUpDtype::Q4KR4 => 64,
        }
    }
    /// Output rows computed per block (grid.x = ceil(n_rows / this)). The r4
    /// kernels do 4 rows/block; the rest one.
    fn rows_per_block(self) -> u32 {
        match self {
            MoeGateUpDtype::Q4KR4 => 4,
            _ => 1,
        }
    }
}

pub fn run_moe_gate_up_sweep<B: BenchBackend>(
    dtype: MoeGateUpDtype,
    repo_root: &Path,
    arch: &str,
) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let pmc = {
        let bytes = B::image(dtype.kernel_stem())
            .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
        let module = B::load(&dev, bytes)?;
        let kernel = B::kernel(&module, dtype.kernel_entry())?;
        B::attributes(&kernel)?
    };

    let n_experts = MOE_35B_EXPERTS;
    let cases = dtype.cases();

    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k) in cases {
        let seed = 0xDEC0DE_u64
            .wrapping_add((n_tokens as u64).wrapping_mul(53))
            .wrapping_add((top_k as u64).wrapping_mul(17))
            .wrapping_add((n_rows as u64).wrapping_mul(7));
        let (gg, gu, rg, ru) =
            run_shape_moe_gate_up::<B>(&dev, dtype, n_experts, n_rows, k, top_k, n_tokens, seed)?;
        let eg = max_rel_err_with_floor(&gg, &rg, (k as f32).sqrt());
        let eu = max_rel_err_with_floor(&gu, &ru, (k as f32).sqrt());
        let max_rel_err = eg.max(eu);
        let tolerance = cert_tol(k);
        results.push(ShapeResult {
            m: n_tokens,
            k,
            n: n_rows,
            seed,
            max_rel_err,
            tolerance,
            pass: max_rel_err <= tolerance,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: dtype.impl_id().to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "indexed_moe_mmvq_gate_up".to_string(),
        dtype_weight: dtype.name().to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))  (fused gate+up MoE MMVQ)"
            .to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    let written = cert.write_to_disk(repo_root)?;
    tracing::info!(
        target: "flambeau_bench::sweep_mmvq",
        cert = %written.display(), pass = cert.pass, "indexed-moe gate_up cert written"
    );
    Ok(cert)
}

/// One fused gate+up MoE MMVQ shape. Two expert weight tensors share a per-token
/// Q8_1 activation; returns `(got_gate, got_up, ref_gate, ref_up)`.
#[expect(clippy::too_many_arguments, reason = "MoE shape carries 7 independent dims")]
fn run_shape_moe_gate_up<B: BenchBackend>(
    dev: &B::Device,
    dtype: MoeGateUpDtype,
    n_experts: usize,
    n_rows: usize,
    k: usize,
    top_k: usize,
    n_tokens: usize,
    seed: u64,
) -> Result<RefQuad> {
    let q_bytes =
        B::image("quantize_q8_1").ok_or_else(|| anyhow!("quantize_q8_1 not compiled"))?;
    let m_bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let q_module = B::load(dev, q_bytes)?;
    let m_module = B::load(dev, m_bytes)?;
    let k_quantize = B::kernel(&q_module, "flambeau_quantize_row_q8_1")?;
    let k_mmvq = B::kernel(&m_module, dtype.kernel_entry())?;

    let wd = dtype.weight_dtype();
    let block_elems = wd.block_elem_count();
    let block_bytes = wd.block_size_bytes();
    assert_eq!(k % block_elems, 0, "k {k} must be multiple of block size {block_elems}");
    let n_blocks = k / block_elems;

    let expert_bytes = n_rows * n_blocks * block_bytes;
    let gate_raw = tame_scales(wd, seeded_bytes(seed, n_experts * expert_bytes));
    let up_raw = tame_scales(wd, seeded_bytes(seed ^ 0xDEAD_BEEF, n_experts * expert_bytes));
    let d_gate = alloc_and_upload(dev, gate_raw.as_slice());
    let d_up = alloc_and_upload(dev, up_raw.as_slice());

    let act_f32 = seeded_f32_range(seed.wrapping_add(0xA5A5A5A5), n_tokens * k, -1.0, 1.0);
    let d_act = alloc_and_upload(dev, act_f32.as_slice());
    let total_y_blocks = n_tokens * k / QK8;
    let y_bytes = total_y_blocks * std::mem::size_of::<BlockQ8_1>();
    let d_y = dev.alloc(y_bytes)?;
    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k) as i32;
        let d_act_ptr: u64 = d_act.as_usize() as u64;
        let d_y_ptr: u64 = d_y.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_act_ptr);
        args.push_arg(&d_y_ptr);
        args.push_arg(&n_elems);
        B::launch(&k_quantize, stream, LaunchCfg::one_d(total_y_blocks as u32, QK8 as u32), args)?;
        stream.synchronize()?;
    }

    let expert_ids = moe_expert_ids(n_tokens * top_k, n_experts, seed);
    let d_ids = alloc_and_upload(dev, expert_ids.as_slice());
    let gate_dq = dequant_selected_experts(wd, &gate_raw, &expert_ids, expert_bytes, n_rows * k)?;
    let up_dq = dequant_selected_experts(wd, &up_raw, &expert_ids, expert_bytes, n_rows * k)?;

    let out_elems = n_tokens * top_k * n_rows;
    let d_gate_out = dev.alloc(out_elems * 4)?;
    let d_up_out = dev.alloc(out_elems * 4)?;
    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_tokens_i = n_tokens as i32;
        let top_k_i = top_k as i32;
        let nb_i = n_blocks as i32;
        let gw_ptr: u64 = d_gate.as_usize() as u64;
        let uw_ptr: u64 = d_up.as_usize() as u64;
        let y_ptr: u64 = d_y.as_usize() as u64;
        let ids_ptr: u64 = d_ids.as_usize() as u64;
        let go_ptr: u64 = d_gate_out.as_usize() as u64;
        let uo_ptr: u64 = d_up_out.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&gw_ptr);
        args.push_arg(&uw_ptr);
        args.push_arg(&y_ptr);
        args.push_arg(&ids_ptr);
        args.push_arg(&go_ptr);
        args.push_arg(&uo_ptr);
        args.push_arg(&n_rows_i);
        args.push_arg(&n_tokens_i);
        args.push_arg(&top_k_i);
        args.push_arg(&nb_i);
        let cfg = LaunchCfg {
            grid: (
                (n_rows as u32).div_ceil(dtype.rows_per_block()),
                (n_tokens * top_k) as u32,
                1,
            ),
            block: (dtype.launch_threads(), 1, 1),
            shared_bytes: 0,
        };
        B::launch(&k_mmvq, stream, cfg, args)?;
        stream.synchronize()?;
    }

    let mut got_g = vec![0.0f32; out_elems];
    let mut got_u = vec![0.0f32; out_elems];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_g.as_mut_ptr() as usize),
            d_gate_out,
            out_elems * 4,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got_u.as_mut_ptr() as usize),
            d_up_out,
            out_elems * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_gate, gate_raw.len())?;
        dev.dealloc(d_up, up_raw.len())?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_bytes)?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_gate_out, out_elems * 4)?;
        dev.dealloc(d_up_out, out_elems * 4)?;
    }

    let mut ref_g = vec![0.0f32; out_elems];
    let mut ref_u = vec![0.0f32; out_elems];
    for t in 0..n_tokens {
        let y_rt = quantize_q8_1_roundtrip(&act_f32[t * k..(t + 1) * k]);
        for slot in 0..top_k {
            let expert = expert_ids[t * top_k + slot] as usize;
            let g_out = reference_matmul(&gate_dq[&expert], &y_rt, n_rows, k);
            let u_out = reference_matmul(&up_dq[&expert], &y_rt, n_rows, k);
            for row in 0..n_rows {
                ref_g[(t * top_k + slot) * n_rows + row] = g_out[row];
                ref_u[(t * top_k + slot) * n_rows + row] = u_out[row];
            }
        }
    }
    Ok((got_g, got_u, ref_g, ref_u))
}

/// Register-tiled prefill MoE MMQ kernels (tile8): MMQ_Y=64 weight rows × TILE_N=8
/// sorted slot-columns per block, each thread computing its row's 8 outputs from
/// registers (no LDS / no cross-thread comms). Experts are pre-sorted + padded-to-8
/// (`sorted_pair_idx_padded` / `padded_offsets`). `gate_up` variants fuse two
/// weights (13-arg ABI, per-token activation); `down` variants take one weight
/// (11-arg ABI, per-pair activation).
#[derive(Clone, Copy)]
pub enum MoeMmqTile8Dtype {
    Q8_0GateUp,
    Q8_0Down,
    Q4_0GateUp,
    Q4_0Down,
    Q4_1Down,
    Q4KGateUp,
    Q4KDown,
    Q5KGateUp,
    Q5KDown,
    Q6KGateUp,
    Q6KDown,
    Q3KGateUp,
    Q3KDown,
    Iq4NlGateUp,
    Iq4NlDown,
    Iq4XsGateUp,
    Iq4XsDown,
    Iq2XxsGateUp,
    Iq2XxsDown,
    Iq2XsGateUp,
    Iq2XsDown,
    Iq2SGateUp,
    Iq2SDown,
    Iq3XxsGateUp,
    Iq3XxsDown,
    Iq3SGateUp,
    Iq3SDown,
    Iq1SGateUp,
    Iq1SDown,
    Iq1MGateUp,
    Iq1MDown,
}

impl MoeMmqTile8Dtype {
    pub fn name(self) -> &'static str {
        use MoeMmqTile8Dtype::*;
        match self {
            Q8_0GateUp => "q8_0_gate_up", Q8_0Down => "q8_0_down",
            Q4_0GateUp => "q4_0_gate_up", Q4_0Down => "q4_0_down",
            Q4_1Down => "q4_1_down",
            Q4KGateUp => "q4_k_gate_up", Q4KDown => "q4_k_down",
            Q5KGateUp => "q5_k_gate_up", Q5KDown => "q5_k_down",
            Q6KGateUp => "q6_k_gate_up", Q6KDown => "q6_k_down",
            Q3KGateUp => "q3_k_gate_up", Q3KDown => "q3_k_down",
            Iq4NlGateUp => "iq4_nl_gate_up", Iq4NlDown => "iq4_nl_down",
            Iq4XsGateUp => "iq4_xs_gate_up", Iq4XsDown => "iq4_xs_down",
            Iq2XxsGateUp => "iq2_xxs_gate_up", Iq2XxsDown => "iq2_xxs_down",
            Iq2XsGateUp => "iq2_xs_gate_up", Iq2XsDown => "iq2_xs_down",
            Iq2SGateUp => "iq2_s_gate_up", Iq2SDown => "iq2_s_down",
            Iq3XxsGateUp => "iq3_xxs_gate_up", Iq3XxsDown => "iq3_xxs_down",
            Iq3SGateUp => "iq3_s_gate_up", Iq3SDown => "iq3_s_down",
            Iq1SGateUp => "iq1_s_gate_up", Iq1SDown => "iq1_s_down",
            Iq1MGateUp => "iq1_m_gate_up", Iq1MDown => "iq1_m_down",
        }
    }
    fn weight_dtype(self) -> Dtype {
        use MoeMmqTile8Dtype::*;
        match self {
            Q8_0GateUp | Q8_0Down => Dtype::Q8_0Dp4a,
            Q4_0GateUp | Q4_0Down => Dtype::Q4_0,
            Q4_1Down => Dtype::Q4_1Dp4a,
            Q4KGateUp | Q4KDown => Dtype::Q4K,
            Q5KGateUp | Q5KDown => Dtype::Q5K,
            Q6KGateUp | Q6KDown => Dtype::Q6K,
            Q3KGateUp | Q3KDown => Dtype::Q3K,
            Iq4NlGateUp | Iq4NlDown => Dtype::Iq4Nl,
            Iq4XsGateUp | Iq4XsDown => Dtype::Iq4Xs,
            Iq2XxsGateUp | Iq2XxsDown => Dtype::Iq2Xxs,
            Iq2XsGateUp | Iq2XsDown => Dtype::Iq2Xs,
            Iq2SGateUp | Iq2SDown => Dtype::Iq2S,
            Iq3XxsGateUp | Iq3XxsDown => Dtype::Iq3Xxs,
            Iq3SGateUp | Iq3SDown => Dtype::Iq3S,
            Iq1SGateUp | Iq1SDown => Dtype::Iq1S,
            Iq1MGateUp | Iq1MDown => Dtype::Iq1M,
        }
    }
    fn gate_up(self) -> bool {
        use MoeMmqTile8Dtype::*;
        matches!(self,
            Q8_0GateUp | Q4_0GateUp | Q4KGateUp | Q5KGateUp | Q6KGateUp | Q3KGateUp
            | Iq4NlGateUp | Iq4XsGateUp | Iq2XxsGateUp | Iq2XsGateUp | Iq2SGateUp
            | Iq3XxsGateUp | Iq3SGateUp | Iq1SGateUp | Iq1MGateUp)
    }
    fn impl_id(self) -> &'static str {
        use MoeMmqTile8Dtype::*;
        match self {
            Q8_0GateUp => "indexed_moe_mmq_q8_0_gate_up_tile8",
            Q8_0Down => "indexed_moe_mmq_q8_0_down_tile8",
            Q4_0GateUp => "indexed_moe_mmq_q4_0_gate_up_tile8",
            Q4_0Down => "indexed_moe_mmq_q4_0_down_tile8",
            Q4_1Down => "indexed_moe_mmq_q4_1_down_tile8",
            Q4KGateUp => "indexed_moe_mmq_q4_k_gate_up_tile8",
            Q4KDown => "indexed_moe_mmq_q4_k_down_tile8",
            Q5KGateUp => "indexed_moe_mmq_q5_k_gate_up_tile8",
            Q5KDown => "indexed_moe_mmq_q5_k_down_tile8",
            Q6KGateUp => "indexed_moe_mmq_q6_k_gate_up_tile8",
            Q6KDown => "indexed_moe_mmq_q6_k_down_tile8",
            Q3KGateUp => "indexed_moe_mmq_q3_k_gate_up_tile8",
            Q3KDown => "indexed_moe_mmq_q3_k_down_tile8",
            Iq4NlGateUp => "indexed_moe_mmq_iq4_nl_gate_up_tile8",
            Iq4NlDown => "indexed_moe_mmq_iq4_nl_down_tile8",
            Iq4XsGateUp => "indexed_moe_mmq_iq4_xs_gate_up_tile8",
            Iq4XsDown => "indexed_moe_mmq_iq4_xs_down_tile8",
            Iq2XxsGateUp => "indexed_moe_mmq_iq2_xxs_gate_up_tile8",
            Iq2XxsDown => "indexed_moe_mmq_iq2_xxs_down_tile8",
            Iq2XsGateUp => "indexed_moe_mmq_iq2_xs_gate_up_tile8",
            Iq2XsDown => "indexed_moe_mmq_iq2_xs_down_tile8",
            Iq2SGateUp => "indexed_moe_mmq_iq2_s_gate_up_tile8",
            Iq2SDown => "indexed_moe_mmq_iq2_s_down_tile8",
            Iq3XxsGateUp => "indexed_moe_mmq_iq3_xxs_gate_up_tile8",
            Iq3XxsDown => "indexed_moe_mmq_iq3_xxs_down_tile8",
            Iq3SGateUp => "indexed_moe_mmq_iq3_s_gate_up_tile8",
            Iq3SDown => "indexed_moe_mmq_iq3_s_down_tile8",
            Iq1SGateUp => "indexed_moe_mmq_iq1_s_gate_up_tile8",
            Iq1SDown => "indexed_moe_mmq_iq1_s_down_tile8",
            Iq1MGateUp => "indexed_moe_mmq_iq1_m_gate_up_tile8",
            Iq1MDown => "indexed_moe_mmq_iq1_m_down_tile8",
        }
    }
    fn kernel_stem(self) -> &'static str {
        use MoeMmqTile8Dtype::*;
        match self {
            Q8_0GateUp => "indexed_moe_mmq_q8_0_gate_up_tile8_dp4a",
            Q8_0Down => "indexed_moe_mmq_q8_0_down_tile8_dp4a",
            Q4_0GateUp => "indexed_moe_mmq_q4_0_gate_up_tile8_dp4a",
            Q4_0Down => "indexed_moe_mmq_q4_0_down_tile8_dp4a",
            Q4_1Down => "indexed_moe_mmq_q4_1_down_tile8_dp4a",
            Q4KGateUp => "indexed_moe_mmq_q4_k_gate_up_tile8_dp4a",
            Q4KDown => "indexed_moe_mmq_q4_k_down_tile8_dp4a",
            Q5KGateUp => "indexed_moe_mmq_q5_k_gate_up_tile8_dp4a",
            Q5KDown => "indexed_moe_mmq_q5_k_down_tile8_dp4a",
            Q6KGateUp => "indexed_moe_mmq_q6_k_gate_up_tile8_dp4a",
            Q6KDown => "indexed_moe_mmq_q6_k_down_tile8_dp4a",
            Q3KGateUp => "indexed_moe_mmq_q3_k_gate_up_tile8_dp4a",
            Q3KDown => "indexed_moe_mmq_q3_k_down_tile8_dp4a",
            Iq4NlGateUp => "indexed_moe_mmq_iq4_nl_gate_up_tile8_dp4a",
            Iq4NlDown => "indexed_moe_mmq_iq4_nl_down_tile8_dp4a",
            Iq4XsGateUp => "indexed_moe_mmq_iq4_xs_gate_up_tile8_dp4a",
            Iq4XsDown => "indexed_moe_mmq_iq4_xs_down_tile8_dp4a",
            Iq2XxsGateUp => "indexed_moe_mmq_iq2_xxs_gate_up_tile8_dp4a",
            Iq2XxsDown => "indexed_moe_mmq_iq2_xxs_down_tile8_dp4a",
            Iq2XsGateUp => "indexed_moe_mmq_iq2_xs_gate_up_tile8_dp4a",
            Iq2XsDown => "indexed_moe_mmq_iq2_xs_down_tile8_dp4a",
            Iq2SGateUp => "indexed_moe_mmq_iq2_s_gate_up_tile8_dp4a",
            Iq2SDown => "indexed_moe_mmq_iq2_s_down_tile8_dp4a",
            Iq3XxsGateUp => "indexed_moe_mmq_iq3_xxs_gate_up_tile8_dp4a",
            Iq3XxsDown => "indexed_moe_mmq_iq3_xxs_down_tile8_dp4a",
            Iq3SGateUp => "indexed_moe_mmq_iq3_s_gate_up_tile8_dp4a",
            Iq3SDown => "indexed_moe_mmq_iq3_s_down_tile8_dp4a",
            Iq1SGateUp => "indexed_moe_mmq_iq1_s_gate_up_tile8_dp4a",
            Iq1SDown => "indexed_moe_mmq_iq1_s_down_tile8_dp4a",
            Iq1MGateUp => "indexed_moe_mmq_iq1_m_gate_up_tile8_dp4a",
            Iq1MDown => "indexed_moe_mmq_iq1_m_down_tile8_dp4a",
        }
    }
    fn kernel_entry(self) -> &'static str {
        use MoeMmqTile8Dtype::*;
        match self {
            Q8_0GateUp => "flambeau_indexed_moe_mmq_q8_0_gate_up_tile8_dp4a_q8_1",
            Q8_0Down => "flambeau_indexed_moe_mmq_q8_0_down_tile8_dp4a_q8_1",
            Q4_0GateUp => "flambeau_indexed_moe_mmq_q4_0_gate_up_tile8_dp4a_q8_1",
            Q4_0Down => "flambeau_indexed_moe_mmq_q4_0_down_tile8_dp4a_q8_1",
            Q4_1Down => "flambeau_indexed_moe_mmq_q4_1_down_tile8_dp4a_q8_1",
            Q4KGateUp => "flambeau_indexed_moe_mmq_q4_k_gate_up_tile8_dp4a_q8_1",
            Q4KDown => "flambeau_indexed_moe_mmq_q4_k_down_tile8_dp4a_q8_1",
            Q5KGateUp => "flambeau_indexed_moe_mmq_q5_k_gate_up_tile8_dp4a_q8_1",
            Q5KDown => "flambeau_indexed_moe_mmq_q5_k_down_tile8_dp4a_q8_1",
            Q6KGateUp => "flambeau_indexed_moe_mmq_q6_k_gate_up_tile8_dp4a_q8_1",
            Q6KDown => "flambeau_indexed_moe_mmq_q6_k_down_tile8_dp4a_q8_1",
            Q3KGateUp => "flambeau_indexed_moe_mmq_q3_k_gate_up_tile8_dp4a_q8_1",
            Q3KDown => "flambeau_indexed_moe_mmq_q3_k_down_tile8_dp4a_q8_1",
            Iq4NlGateUp => "flambeau_indexed_moe_mmq_iq4_nl_gate_up_tile8_dp4a_q8_1",
            Iq4NlDown => "flambeau_indexed_moe_mmq_iq4_nl_down_tile8_dp4a_q8_1",
            Iq4XsGateUp => "flambeau_indexed_moe_mmq_iq4_xs_gate_up_tile8_dp4a_q8_1",
            Iq4XsDown => "flambeau_indexed_moe_mmq_iq4_xs_down_tile8_dp4a_q8_1",
            Iq2XxsGateUp => "flambeau_indexed_moe_mmq_iq2_xxs_gate_up_tile8_dp4a_q8_1",
            Iq2XxsDown => "flambeau_indexed_moe_mmq_iq2_xxs_down_tile8_dp4a_q8_1",
            Iq2XsGateUp => "flambeau_indexed_moe_mmq_iq2_xs_gate_up_tile8_dp4a_q8_1",
            Iq2XsDown => "flambeau_indexed_moe_mmq_iq2_xs_down_tile8_dp4a_q8_1",
            Iq2SGateUp => "flambeau_indexed_moe_mmq_iq2_s_gate_up_tile8_dp4a_q8_1",
            Iq2SDown => "flambeau_indexed_moe_mmq_iq2_s_down_tile8_dp4a_q8_1",
            Iq3XxsGateUp => "flambeau_indexed_moe_mmq_iq3_xxs_gate_up_tile8_dp4a_q8_1",
            Iq3XxsDown => "flambeau_indexed_moe_mmq_iq3_xxs_down_tile8_dp4a_q8_1",
            Iq3SGateUp => "flambeau_indexed_moe_mmq_iq3_s_gate_up_tile8_dp4a_q8_1",
            Iq3SDown => "flambeau_indexed_moe_mmq_iq3_s_down_tile8_dp4a_q8_1",
            Iq1SGateUp => "flambeau_indexed_moe_mmq_iq1_s_gate_up_tile8_dp4a_q8_1",
            Iq1SDown => "flambeau_indexed_moe_mmq_iq1_s_down_tile8_dp4a_q8_1",
            Iq1MGateUp => "flambeau_indexed_moe_mmq_iq1_m_gate_up_tile8_dp4a_q8_1",
            Iq1MDown => "flambeau_indexed_moe_mmq_iq1_m_down_tile8_dp4a_q8_1",
        }
    }
    /// Correctness envelope. The K-quant prefill kernels inherit the documented
    /// 5e-2 MoE envelope (Q8_1-activation quant noise on cancellation-heavy rows
    /// at the larger pair counts); the simple block quants stay at the tight 3e-2.
    fn tolerance(self) -> f32 {
        use MoeMmqTile8Dtype::*;
        match self {
            Q4KGateUp | Q4KDown | Q5KGateUp | Q5KDown | Q6KGateUp | Q6KDown
            | Q3KGateUp | Q3KDown
            | Iq4NlGateUp | Iq4NlDown | Iq4XsGateUp | Iq4XsDown
            | Iq2XxsGateUp | Iq2XxsDown | Iq2XsGateUp | Iq2XsDown
            | Iq2SGateUp | Iq2SDown | Iq3XxsGateUp | Iq3XxsDown
            | Iq3SGateUp | Iq3SDown | Iq1SGateUp | Iq1SDown
            | Iq1MGateUp | Iq1MDown => 5e-2,
            _ => 3e-2,
        }
    }
}

/// Host build of the pad-to-8 expert sort (`sorted_pair_idx_padded`,
/// `padded_offsets`): pairs grouped by expert, each expert's range padded to a
/// multiple of 8 by repeating its last real pair index. Mirrors the on-device
/// `flambeau_moe_sort_*_padded` output.
fn build_padded_sort(expert_ids: &[i32], n_experts: usize) -> (Vec<i32>, Vec<i32>) {
    let mut groups: Vec<Vec<i32>> = vec![Vec::new(); n_experts];
    for (pair, &e) in expert_ids.iter().enumerate() {
        groups[e as usize].push(pair as i32);
    }
    let mut sorted_padded = Vec::new();
    let mut padded_offsets = vec![0i32];
    for g in &groups {
        let padded_count = (g.len() + 7) & !7;
        sorted_padded.extend_from_slice(g);
        if let Some(&last) = g.last() {
            for _ in g.len()..padded_count {
                sorted_padded.push(last);
            }
        }
        padded_offsets.push(sorted_padded.len() as i32);
    }
    (sorted_padded, padded_offsets)
}

pub fn run_moe_mmq_tile8_sweep<B: BenchBackend>(
    dtype: MoeMmqTile8Dtype,
    repo_root: &Path,
    arch: &str,
) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let pmc = {
        let bytes = B::image(dtype.kernel_stem())
            .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
        let module = B::load(&dev, bytes)?;
        let kernel = B::kernel(&module, dtype.kernel_entry())?;
        B::attributes(&kernel)?
    };

    let n_experts = 4usize;
    // tile8 targets the n_tokens >= 32 prefill regime.
    let cases = [
        (32usize, 4usize, 256usize, 2048usize),
        (128, 4, 256, 2048),
        (32, 4, 64, 768),
    ];

    let mut results = Vec::new();
    for (n_tokens, top_k, n_rows, k) in cases {
        let seed = 0xDEC0DE_u64
            .wrapping_add((n_tokens as u64).wrapping_mul(53))
            .wrapping_add((top_k as u64).wrapping_mul(17))
            .wrapping_add((n_rows as u64).wrapping_mul(7));
        let (got, reference) =
            run_shape_moe_mmq_tile8::<B>(&dev, dtype, n_experts, n_rows, k, top_k, n_tokens, seed)?;
        let max_rel_err = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt());
        let tolerance = dtype.tolerance();
        results.push(ShapeResult {
            m: n_tokens,
            k,
            n: n_rows,
            seed,
            max_rel_err,
            tolerance,
            pass: max_rel_err <= tolerance,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: dtype.impl_id().to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "indexed_moe_mmq_tile8".to_string(),
        dtype_weight: "Q8_0".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))  (tile8 register-tiled MoE MMQ)"
            .to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    let written = cert.write_to_disk(repo_root)?;
    tracing::info!(
        target: "flambeau_bench::sweep_mmvq",
        cert = %written.display(), pass = cert.pass, "indexed-moe tile8 cert written"
    );
    Ok(cert)
}

/// One tile8 MoE MMQ shape. For `gate_up`: two weights, per-token activation,
/// outputs `[n_pairs][n_rows]` for gate+up (compared together). For `down`: one
/// weight, per-pair activation, output `[n_pairs][n_rows]`. Returns
/// `(got, reference)` over the flattened output(s).
#[expect(clippy::too_many_arguments, reason = "MoE shape carries 7 independent dims")]
fn run_shape_moe_mmq_tile8<B: BenchBackend>(
    dev: &B::Device,
    dtype: MoeMmqTile8Dtype,
    n_experts: usize,
    n_rows: usize,
    k: usize,
    top_k: usize,
    n_tokens: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let gate_up = dtype.gate_up();
    let n_pairs = n_tokens * top_k;
    let n_act_rows = if gate_up { n_tokens } else { n_pairs };

    let q_bytes =
        B::image("quantize_q8_1").ok_or_else(|| anyhow!("quantize_q8_1 not compiled"))?;
    let m_bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let q_module = B::load(dev, q_bytes)?;
    let m_module = B::load(dev, m_bytes)?;
    let k_quantize = B::kernel(&q_module, "flambeau_quantize_row_q8_1")?;
    let k_mmq = B::kernel(&m_module, dtype.kernel_entry())?;

    let wd = dtype.weight_dtype();
    let block_elems = wd.block_elem_count();
    let block_bytes = wd.block_size_bytes();
    assert_eq!(k % block_elems, 0);
    let n_blocks = k / block_elems;

    // Expert weights [n_experts][n_rows][k] (gate; up only for gate_up).
    let gate_raw =
        tame_scales(wd, seeded_bytes(seed, n_experts * n_rows * n_blocks * block_bytes));
    let mut gate_dq = vec![0.0f32; n_experts * n_rows * k];
    dequantize_into(wd.ggml(), &gate_raw, &mut gate_dq)?;
    let d_gate = alloc_and_upload(dev, gate_raw.as_slice());
    let (up_raw, up_dq, d_up) = if gate_up {
        let raw = tame_scales(
            wd,
            seeded_bytes(seed ^ 0xDEAD_BEEF, n_experts * n_rows * n_blocks * block_bytes),
        );
        let mut dq = vec![0.0f32; n_experts * n_rows * k];
        dequantize_into(wd.ggml(), &raw, &mut dq)?;
        let d = alloc_and_upload(dev, raw.as_slice());
        (raw, dq, Some(d))
    } else {
        (Vec::new(), Vec::new(), None)
    };

    // Activation [n_act_rows][k] → Q8_1.
    let act_f32 = seeded_f32_range(seed.wrapping_add(0xA5A5A5A5), n_act_rows * k, -1.0, 1.0);
    let d_act = alloc_and_upload(dev, act_f32.as_slice());
    let total_y_blocks = n_act_rows * k / QK8;
    let y_bytes = total_y_blocks * std::mem::size_of::<BlockQ8_1>();
    let d_y = dev.alloc(y_bytes)?;
    {
        let stream = dev.default_stream();
        let n_elems = (n_act_rows * k) as i32;
        let d_act_ptr: u64 = d_act.as_usize() as u64;
        let d_y_ptr: u64 = d_y.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_act_ptr);
        args.push_arg(&d_y_ptr);
        args.push_arg(&n_elems);
        B::launch(&k_quantize, stream, LaunchCfg::one_d(total_y_blocks as u32, QK8 as u32), args)?;
        stream.synchronize()?;
    }

    // Expert routing per pair + the pad-to-8 sort.
    let expert_ids: Vec<i32> = (0..n_pairs)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed.wrapping_mul(0x12345));
            ((h >> 32) as u32 % n_experts as u32) as i32
        })
        .collect();
    let (sorted_padded, padded_offsets) = build_padded_sort(&expert_ids, n_experts);
    let padded_total = *padded_offsets.last().unwrap() as usize;
    let d_ids = alloc_and_upload(dev, expert_ids.as_slice());
    let d_sorted = alloc_and_upload(dev, sorted_padded.as_slice());
    let d_pofs = alloc_and_upload(dev, padded_offsets.as_slice());

    let out_elems = n_pairs * n_rows;
    let d_out = dev.alloc(out_elems * 4)?;
    let d_up_out = if gate_up { Some(dev.alloc(out_elems * 4)?) } else { None };

    {
        let stream = dev.default_stream();
        // Kernel sees n_tokens (=n_act_rows for down) + top_k (=1 for down).
        let n_rows_i = n_rows as i32;
        let n_tokens_i = n_act_rows as i32;
        let top_k_i = if gate_up { top_k as i32 } else { 1 };
        let nb_i = n_blocks as i32;
        let n_experts_i = n_experts as i32;
        let g_ptr: u64 = d_gate.as_usize() as u64;
        let y_ptr: u64 = d_y.as_usize() as u64;
        let e_ptr: u64 = d_ids.as_usize() as u64;
        let s_ptr: u64 = d_sorted.as_usize() as u64;
        let po_ptr: u64 = d_pofs.as_usize() as u64;
        let o_ptr: u64 = d_out.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&g_ptr);
        let u_ptr: u64 = d_up.map_or(0, |p| p.as_usize() as u64);
        let uo_ptr: u64 = d_up_out.map_or(0, |p| p.as_usize() as u64);
        if gate_up {
            args.push_arg(&u_ptr);
        }
        args.push_arg(&y_ptr);
        args.push_arg(&e_ptr);
        args.push_arg(&s_ptr);
        args.push_arg(&po_ptr);
        args.push_arg(&o_ptr);
        if gate_up {
            args.push_arg(&uo_ptr);
        }
        args.push_arg(&n_rows_i);
        args.push_arg(&n_tokens_i);
        args.push_arg(&top_k_i);
        args.push_arg(&nb_i);
        args.push_arg(&n_experts_i);
        let grid_y = (padded_total.div_ceil(8)) as u32;
        let cfg = LaunchCfg {
            grid: ((n_rows as u32).div_ceil(64), grid_y.max(1), 1),
            block: (64, 1, 1),
            shared_bytes: 0,
        };
        B::launch(&k_mmq, stream, cfg, args)?;
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; out_elems];
    let mut got_up = vec![0.0f32; out_elems];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_out,
            out_elems * 4,
        )?;
        if let Some(p) = d_up_out {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got_up.as_mut_ptr() as usize),
                p,
                out_elems * 4,
            )?;
        }
    }
    dev.default_stream().synchronize()?;

    unsafe {
        dev.dealloc(d_gate, gate_raw.len())?;
        if let Some(p) = d_up {
            dev.dealloc(p, up_raw.len())?;
        }
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y, y_bytes)?;
        dev.dealloc(d_ids, expert_ids.len() * 4)?;
        dev.dealloc(d_sorted, sorted_padded.len() * 4)?;
        dev.dealloc(d_pofs, padded_offsets.len() * 4)?;
        dev.dealloc(d_out, out_elems * 4)?;
        if let Some(p) = d_up_out {
            dev.dealloc(p, out_elems * 4)?;
        }
    }

    // Reference: per pair, dot the selected expert's weight row with the
    // Q8_1-roundtripped activation. For gate_up the activation is the pair's
    // token row; for down it's the pair's own row. gate+up errors are folded
    // together (concatenated) so one cert covers both outputs.
    let mut reference = vec![0.0f32; out_elems];
    let mut ref_up = vec![0.0f32; out_elems];
    let mut rt_cache: Vec<Option<Vec<f32>>> = vec![None; n_act_rows];
    for p in 0..n_pairs {
        let expert = expert_ids[p] as usize;
        let act_row = if gate_up { p / top_k } else { p };
        if rt_cache[act_row].is_none() {
            rt_cache[act_row] =
                Some(quantize_q8_1_roundtrip(&act_f32[act_row * k..(act_row + 1) * k]));
        }
        let y_rt = rt_cache[act_row].as_ref().unwrap();
        let gbase = &gate_dq[expert * n_rows * k..(expert + 1) * n_rows * k];
        let g_out = reference_matmul(gbase, y_rt, n_rows, k);
        for row in 0..n_rows {
            reference[p * n_rows + row] = g_out[row];
        }
        if gate_up {
            let ubase = &up_dq[expert * n_rows * k..(expert + 1) * n_rows * k];
            let u_out = reference_matmul(ubase, y_rt, n_rows, k);
            for row in 0..n_rows {
                ref_up[p * n_rows + row] = u_out[row];
            }
        }
    }

    if gate_up {
        let mut got_all = got;
        got_all.extend_from_slice(&got_up);
        let mut ref_all = reference;
        ref_all.extend_from_slice(&ref_up);
        Ok((got_all, ref_all))
    } else {
        Ok((got, reference))
    }
}

/// DS4 4-warp_lds MMQ kernels. Consume the 144-B `block_q8_1_mmq` activation
/// (produced by the DS4 quantizer), MMQ_Y=128 × MMQ_X=64 tile, dynamic LDS,
/// 9-arg ABI with explicit Y/X strides.
#[derive(Clone, Copy)]
pub enum Mmq4WarpLdsDtype {
    Q4_0,
    Q4_1,
    Q4K,
}

impl Mmq4WarpLdsDtype {
    pub fn name(self) -> &'static str {
        match self {
            Mmq4WarpLdsDtype::Q4_0 => "Q4_0",
            Mmq4WarpLdsDtype::Q4_1 => "Q4_1",
            Mmq4WarpLdsDtype::Q4K => "Q4_K",
        }
    }
    fn weight_dtype(self) -> Dtype {
        match self {
            Mmq4WarpLdsDtype::Q4_0 => Dtype::Q4_0,
            Mmq4WarpLdsDtype::Q4_1 => Dtype::Q4_1Dp4a,
            Mmq4WarpLdsDtype::Q4K => Dtype::Q4K,
        }
    }
    fn block_bytes(self) -> usize {
        match self {
            Mmq4WarpLdsDtype::Q4_0 => 18,
            Mmq4WarpLdsDtype::Q4_1 => 20,
            Mmq4WarpLdsDtype::Q4K => 144,
        }
    }
    fn kernel_stem(self) -> &'static str {
        match self {
            Mmq4WarpLdsDtype::Q4_0 => "mmq_q4_0_4warp_lds",
            Mmq4WarpLdsDtype::Q4_1 => "mmq_q4_1_4warp_lds",
            Mmq4WarpLdsDtype::Q4K => "mmq_q4_K_lds_tiled",
        }
    }
    fn kernel_entry(self) -> &'static str {
        match self {
            Mmq4WarpLdsDtype::Q4_0 => "flambeau_mmq_q4_0_4warp_lds_q8_1",
            Mmq4WarpLdsDtype::Q4_1 => "flambeau_mmq_q4_1_4warp_lds_q8_1",
            Mmq4WarpLdsDtype::Q4K => "flambeau_mmq_q4_K_lds_tiled_q8_1",
        }
    }
    fn impl_id(self) -> &'static str {
        match self {
            Mmq4WarpLdsDtype::Q4_0 => "qmatmul_q4_0_mmq_4warp_lds",
            Mmq4WarpLdsDtype::Q4_1 => "qmatmul_q4_1_mmq_4warp_lds",
            Mmq4WarpLdsDtype::Q4K => "qmatmul_q4_k_mmq_lds_tiled",
        }
    }
    /// (MMQ_X cols/block, dynamic-LDS bytes). MMQ_Y is 128 for all.
    fn launch(self) -> (u32, u32) {
        match self {
            // tile_y 64*36 + x_qs 128*33 + x_dm 128*8 (half2) ints = 7584 → 30336 B.
            Mmq4WarpLdsDtype::Q4_0 | Mmq4WarpLdsDtype::Q4_1 => (64, 30336),
            // tile_y 16*36 + x_qs 4224 + x_dm 128 + x_sc 528 = 5456 ints → 21824 B.
            Mmq4WarpLdsDtype::Q4K => (16, 21824),
        }
    }
}

/// Q4_K 6-bit packed min for sub-block `j` from the 12-byte `scales` (mirrors
/// `flambeau_q4k_scale_min`'s `m` output; the kernel's vec_dot multiplies it by
/// `dmin`).
fn q4k_min(j: usize, q: &[u8]) -> u8 {
    if j < 4 {
        q[j + 4] & 63
    } else {
        (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    }
}

pub fn run_mmq_4warp_lds_sweep<B: BenchBackend>(
    dtype: Mmq4WarpLdsDtype,
    repo_root: &Path,
    arch: &str,
) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let pmc = {
        let bytes = B::image(dtype.kernel_stem())
            .ok_or_else(|| anyhow!("{} not compiled", dtype.kernel_stem()))?;
        let module = B::load(&dev, bytes)?;
        let kernel = B::kernel(&module, dtype.kernel_entry())?;
        B::attributes(&kernel)?
    };

    // k must be a multiple of ITER_K (256). n_cols must let the DS4 quantizer
    // tile (k % 128 == 0 → guaranteed by k % 256 == 0).
    let rows_grid = vec![128usize, 256, 4096];
    let cols_grid = vec![64usize, 128];
    let k_grid = vec![2048usize, 4096];
    let seed = 0xC0FFEE_u64;

    let mut results = Vec::new();
    for &n_rows in &rows_grid {
        for &n_cols in &cols_grid {
            for &k in &k_grid {
                let s = seed
                    .wrapping_add((n_rows as u64).wrapping_mul(0x1234567))
                    .wrapping_add((n_cols as u64).wrapping_mul(0x85EBCA6B))
                    .wrapping_add((k as u64).wrapping_mul(0x9E3779B97F4A7C15));
                let (got, reference) = run_shape_mmq_4warp_lds::<B>(&dev, dtype, n_rows, n_cols, k, s)?;
                let max_rel_err = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt());
                let tolerance = cert_tol(k);
                results.push(ShapeResult {
                    m: n_rows,
                    k,
                    n: n_cols,
                    seed: s,
                    max_rel_err,
                    tolerance,
                    pass: max_rel_err <= tolerance,
                });
            }
        }
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: dtype.impl_id().to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "qmatmul_mmq".to_string(),
        dtype_weight: dtype.name().to_string(),
        dtype_activation: "Q8_1_MMQ".to_string(),
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

#[allow(clippy::undocumented_unsafe_blocks)]
fn run_shape_mmq_4warp_lds<B: BenchBackend>(
    dev: &B::Device,
    dtype: Mmq4WarpLdsDtype,
    n_rows: usize,
    n_cols: usize,
    k: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    assert_eq!(k % 256, 0, "k {k} must be a multiple of ITER_K (256)");
    let q_bytes = B::image("quantize_q8_1_mmq")
        .ok_or_else(|| anyhow!("quantize_q8_1_mmq not compiled"))?;
    let m_bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} not compiled", dtype.kernel_stem()))?;
    let q_module = B::load(dev, q_bytes)?;
    let m_module = B::load(dev, m_bytes)?;
    let k_quantize = B::kernel(&q_module, "flambeau_quantize_q8_1_mmq")?;
    let k_mmq = B::kernel(&m_module, dtype.kernel_entry())?;

    let wd = dtype.weight_dtype();
    let block_elems = wd.block_elem_count();
    let block_bytes = wd.block_size_bytes();
    let bb_stride = dtype.block_bytes();
    let n_blocks = k / block_elems;
    let weights_raw = tame_scales(wd, seeded_bytes(seed, n_rows * n_blocks * block_bytes));
    let mut w_dequant = vec![0.0f32; n_rows * k];
    dequantize_into(wd.ggml(), &weights_raw, &mut w_dequant)?;
    let d_x = alloc_and_upload(dev, weights_raw.as_slice());

    // Activation: [n_cols][k] F32 → DS4 quantize to [k/128][n_cols] block_q8_1_mmq.
    let act_f32 = seeded_f32_range(seed.wrapping_add(0xA5A5A5A5), n_cols * k, -1.0, 1.0);
    let d_act = alloc_and_upload(dev, act_f32.as_slice());
    const DS4_BYTES: usize = 144;
    let n_big_blocks = k / 128;
    let y_bytes = n_big_blocks * n_cols * DS4_BYTES;
    let d_y_ds4 = dev.alloc(y_bytes)?;
    {
        let stream = dev.default_stream();
        let ncols_i = k as i32;
        let total_b_i = n_cols as i32;
        let d_act_ptr: u64 = d_act.as_usize() as u64;
        let d_y_ptr: u64 = d_y_ds4.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_act_ptr);
        args.push_arg(&d_y_ptr);
        args.push_arg(&ncols_i);
        args.push_arg(&total_b_i);
        let cfg = LaunchCfg {
            grid: ((k / 128) as u32, n_cols as u32, 1),
            block: (128, 1, 1),
            shared_bytes: 0,
        };
        B::launch(&k_quantize, stream, cfg, args)?;
        stream.synchronize()?;
    }

    let d_dst = dev.alloc(n_cols * n_rows * 4)?;
    {
        let stream = dev.default_stream();
        let ncols_x = k as i32;
        let nrows_x = n_rows as i32;
        let ncols_y = n_cols as i32;
        let stride_col_y = n_cols as i32;     // total_b: big-block stride in DS4 vy
        let stride_row_x = n_blocks as i32;   // Q4_0 blocks per weight row
        let nrows_dst = n_rows as i32;
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_y_ptr: u64 = d_y_ds4.as_usize() as u64;
        let d_dst_ptr: u64 = d_dst.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_x_ptr);
        args.push_arg(&d_y_ptr);
        args.push_arg(&d_dst_ptr);
        args.push_arg(&ncols_x);
        args.push_arg(&nrows_x);
        args.push_arg(&ncols_y);
        args.push_arg(&stride_col_y);
        args.push_arg(&stride_row_x);
        args.push_arg(&nrows_dst);
        let (mmq_x, shared_bytes) = dtype.launch();
        let cfg = LaunchCfg {
            grid: ((n_rows as u32).div_ceil(128), (n_cols as u32).div_ceil(mmq_x), 1),
            block: (64, 4, 1),
            shared_bytes,
        };
        B::launch(&k_mmq, stream, cfg, args)?;
        stream.synchronize()?;
    }

    let mut got = vec![0.0f32; n_cols * n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            n_cols * n_rows * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_x, weights_raw.len())?;
        dev.dealloc(d_act, act_f32.len() * 4)?;
        dev.dealloc(d_y_ds4, y_bytes)?;
        dev.dealloc(d_dst, n_cols * n_rows * 4)?;
    }

    // Reference: kernel-faithful. Per 32-elem block, sumi = Σ q_raw·q8 (raw
    // nibble 0..15, NOT q−8), d_y = amax/127 over the activation sub-block,
    // q8 = round(x/d_y), ssum = Σx (raw activation sum). Q4_0 block (bb_stride 18:
    // fp16 d, qs) → d_x·(d_y·sumi − 8·ssum); Q4_1 block (20: fp16 d, fp16 m, qs)
    // → d_x·d_y·sumi + m_x·ssum. Weight d_x/m_x + nibbles decoded from the raw
    // bytes (GGML element order — low nibble e<16, high nibble e+16).
    if matches!(dtype, Mmq4WarpLdsDtype::Q4K) {
        // Q4_K super-block affine over DS4 activation. Reuse w_dequant (correct
        // GGML dequant) for the dot term; per 32-elem sub-block the kernel adds
        // d_y·Σ(w·q8) + dmin·m·(d_y·Σq8 − ssum). dmin + the 6-bit m[sub] decoded
        // from the raw block (144 B: d, dmin fp16, scales[12], qs[128]).
        let mut reference = vec![0.0f32; n_cols * n_rows];
        for c in 0..n_cols {
            let col = &act_f32[c * k..(c + 1) * k];
            for row in 0..n_rows {
                let mut acc = 0.0f64;
                for sup in 0..n_blocks {
                    let bb = (row * n_blocks + sup) * 144;
                    let dmin = f16::from_le_bytes([weights_raw[bb + 2], weights_raw[bb + 3]]).to_f32();
                    let scales = &weights_raw[bb + 4..bb + 16];
                    for sub in 0..8usize {
                        let m_sub = q4k_min(sub, scales) as f32;
                        let e0 = sup * 256 + sub * 32;
                        let xs = &col[e0..e0 + 32];
                        let amax = xs.iter().fold(0.0f32, |mx, &v| mx.max(v.abs()));
                        let d_y = amax / 127.0;
                        let ssum: f32 = xs.iter().sum();
                        let mut sum_wq8 = 0.0f64;
                        let mut sum_q8 = 0.0f64;
                        for e in 0..32usize {
                            let q8 = if d_y != 0.0 {
                                (xs[e] / d_y).round().clamp(-127.0, 127.0)
                            } else {
                                0.0
                            };
                            sum_wq8 += w_dequant[row * k + e0 + e] as f64 * q8 as f64;
                            sum_q8 += q8 as f64;
                        }
                        acc += d_y as f64 * sum_wq8
                            + dmin as f64 * m_sub as f64 * (d_y as f64 * sum_q8 - ssum as f64);
                    }
                }
                reference[c * n_rows + row] = acc as f32;
            }
        }
        return Ok((got, reference));
    }

    let _ = &w_dequant;
    let is_q4_1 = matches!(dtype, Mmq4WarpLdsDtype::Q4_1);
    let qs_off = if is_q4_1 { 4 } else { 2 };
    let mut reference = vec![0.0f32; n_cols * n_rows];
    for c in 0..n_cols {
        let col = &act_f32[c * k..(c + 1) * k];
        for row in 0..n_rows {
            let mut acc = 0.0f64;
            for blk in 0..n_blocks {
                let xs = &col[blk * 32..(blk + 1) * 32];
                let amax = xs.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let d_y = amax / 127.0;
                let ssum: f32 = xs.iter().sum();
                let bb = (row * n_blocks + blk) * bb_stride;
                let d_x = f16::from_le_bytes([weights_raw[bb], weights_raw[bb + 1]]).to_f32();
                let m_x = if is_q4_1 {
                    f16::from_le_bytes([weights_raw[bb + 2], weights_raw[bb + 3]]).to_f32()
                } else {
                    0.0
                };
                let mut sumi = 0i64;
                for e in 0..32usize {
                    let byte = weights_raw[bb + qs_off + (e & 15)] as i32;
                    let q_raw = if e < 16 { byte & 0x0F } else { byte >> 4 };
                    let q8 = if d_y != 0.0 {
                        (xs[e] / d_y).round().clamp(-127.0, 127.0) as i64
                    } else {
                        0
                    };
                    sumi += q_raw as i64 * q8;
                }
                acc += if is_q4_1 {
                    d_x as f64 * d_y as f64 * sumi as f64 + m_x as f64 * ssum as f64
                } else {
                    d_x as f64 * (d_y as f64 * sumi as f64 - 8.0 * ssum as f64)
                };
            }
            reference[c * n_rows + row] = acc as f32;
        }
    }
    Ok((got, reference))
}

// ---- autotune: rank candidate variants, emit dispatch rows ----------------

/// Median wall-clock microseconds of one mmvq launch for `dtype` at (m, k),
/// over `iters` timed launches after a short warmup. The activation is
/// quantized once up front; only the mmvq launch + sync is inside the timed
/// region. Inputs are arbitrary (these kernels are data-independent in time).
fn time_mmvq<B: BenchBackend>(
    dev: &B::Device,
    dtype: Dtype,
    m: usize,
    k: usize,
    iters: usize,
) -> Result<f64> {
    let q_bytes =
        B::image("quantize_q8_1").ok_or_else(|| anyhow!("quantize_q8_1 not compiled"))?;
    let m_bytes = B::image(dtype.kernel_stem())
        .ok_or_else(|| anyhow!("{} kernel not compiled", dtype.kernel_stem()))?;
    let q_module = B::load(dev, q_bytes)?;
    let m_module = B::load(dev, m_bytes)?;
    let k_quantize = B::kernel(&q_module, "flambeau_quantize_row_q8_1")?;
    let k_mmvq = B::kernel(&m_module, dtype.kernel_entry())?;

    let block_elems = dtype.block_elem_count();
    let weights_raw = seeded_bytes(0xA17E, m * (k / block_elems) * dtype.block_size_bytes());
    let y_f32 = seeded_f32_range(0xB22E, k, -1.0, 1.0);

    let d_x = alloc_and_upload(dev, weights_raw.as_slice());
    let d_y_f32 = alloc_and_upload(dev, y_f32.as_slice());

    let y_q8_1_blocks = k / QK8;
    let y_q8_1_bytes = y_q8_1_blocks * std::mem::size_of::<BlockQ8_1>();
    let d_y_q8_1 = dev.alloc(y_q8_1_bytes)?;
    {
        let stream = dev.default_stream();
        let n_elems = k as i32;
        let d_y_f32_ptr: u64 = d_y_f32.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_y_f32_ptr);
        args.push_arg(&d_y_q8_1_ptr);
        args.push_arg(&n_elems);
        B::launch(&k_quantize, stream, LaunchCfg::one_d(y_q8_1_blocks as u32, QK8 as u32), args)?;
        stream.synchronize()?;
    }
    let d_dst = dev.alloc(m * 4)?;

    let m_i = m as i32;
    let n_units = (k / block_elems) as i32;
    let d_x_ptr: u64 = d_x.as_usize() as u64;
    let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
    let d_dst_ptr: u64 = d_dst.as_usize() as u64;
    let cfg = LaunchCfg::one_d((m as u32).div_ceil(dtype.rows_per_block()), dtype.launch_threads());
    let stream = dev.default_stream();

    let launch = || {
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_x_ptr);
        args.push_arg(&d_y_q8_1_ptr);
        args.push_arg(&d_dst_ptr);
        args.push_arg(&m_i);
        args.push_arg(&n_units);
        B::launch(&k_mmvq, stream, cfg, args)
    };

    for _ in 0..3 {
        launch()?;
    }
    stream.synchronize()?;

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        launch()?;
        stream.synchronize()?;
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).expect("timing has no NaN"));

    // SAFETY: each ptr came from a successful alloc above; the stream is synced.
    unsafe {
        dev.dealloc(d_x, weights_raw.len())?;
        dev.dealloc(d_y_f32, y_f32.len() * 4)?;
        dev.dealloc(d_y_q8_1, y_q8_1_bytes)?;
        dev.dealloc(d_dst, m * 4)?;
    }
    Ok(samples[samples.len() / 2])
}

/// Time one MMQ (prefill, matrix×matrix) kernel of the wave64 8-arg / tile
/// family at (`m_tokens` × `n_out` × `k`). `m_tokens` is the **dispatch axis**
/// (n_tokens) — unlike `time_mmvq`, whose `m` is n_out — so MMQ rankings map onto
/// the real `[[qmatmul_mmq]]` m-bands. Operands are zeroed (throughput is
/// content-independent). Median µs/launch over `iters`.
fn time_mmq<B: BenchBackend>(
    dev: &B::Device,
    cand: (&'static str, &'static str, u32),
    m_tokens: usize,
    n_out: usize,
    k: usize,
    iters: usize,
) -> Result<f64> {
    let (stem, entry, tile_n) = cand;
    let bytes = B::image(stem).ok_or_else(|| anyhow!("{stem} not compiled"))?;
    let module = B::load(dev, bytes)?;
    let kernel = B::kernel(&module, entry)?;

    let w_bytes = n_out * (k / QK8) * std::mem::size_of::<BlockQ8_0>();
    let y_bytes = m_tokens * (k / QK8) * std::mem::size_of::<BlockQ8_1>();
    let dst_bytes = m_tokens * n_out * 4;
    let wz = vec![0u8; w_bytes];
    let yz = vec![0u8; y_bytes];
    let d_x = alloc_and_upload(dev, wz.as_slice());
    let d_y = alloc_and_upload(dev, yz.as_slice());
    let d_dst = dev.alloc(dst_bytes)?;
    let stream = dev.default_stream();

    let ncols_x = k as i32;
    let nrows_x = n_out as i32;
    let ncols_y = m_tokens as i32;
    let nrows_y = k as i32;
    let nrows_dst = n_out as i32;
    let d_x_ptr = d_x.as_usize() as u64;
    let d_y_ptr = d_y.as_usize() as u64;
    let d_dst_ptr = d_dst.as_usize() as u64;
    let cfg = LaunchCfg {
        grid: ((n_out as u32).div_ceil(64), (m_tokens as u32).div_ceil(tile_n), 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    let launch = || {
        let mut args = <B::Args<'_>>::default();
        args.push_arg(&d_x_ptr);
        args.push_arg(&d_y_ptr);
        args.push_arg(&d_dst_ptr);
        args.push_arg(&ncols_x);
        args.push_arg(&nrows_x);
        args.push_arg(&ncols_y);
        args.push_arg(&nrows_y);
        args.push_arg(&nrows_dst);
        B::launch(&kernel, stream, cfg, args)
    };
    for _ in 0..3 {
        launch()?;
    }
    stream.synchronize()?;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        launch()?;
        stream.synchronize()?;
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).expect("timing has no NaN"));
    unsafe {
        dev.dealloc(d_x, w_bytes)?;
        dev.dealloc(d_y, y_bytes)?;
        dev.dealloc(d_dst, dst_bytes)?;
    }
    Ok(samples[samples.len() / 2])
}

/// Prefill/MMQ ranking — the compute-bound regime where multiple kernels
/// genuinely compete (unlike bandwidth-bound decode). Times each wave64-family
/// MMQ candidate for `dtype` across the **n_tokens** grid at a representative
/// (n_out, k) and prints the ranking vs the committed `[[qmatmul_mmq]]` row.
/// Measurement only (no emit) — the cross-family kernels (oracle /
/// 4warp_lds) have distinct launch signatures and aren't wired here yet.
pub fn autotune_mmq<B: BenchBackend>(
    arch: &str,
    dtype: &str,
    m_grid: &[usize],
    n_out: usize,
    k: usize,
) -> Result<()> {
    let dev = B::new_device(0)?;
    let backend = dev.backend();
    let iters = 50;
    // (stem, entry, tile_n) — wave64 8-arg family. `tile_n` = activation rows per
    // block (grid.y = M / tile_n).
    let cands: &[(&str, &str, u32)] = match dtype {
        "Q8_0" => &[
            ("mmq_q8_0_wave64", "flambeau_mmq_q8_0_wave64_q8_1", 8),
            ("mmq_q8_0_wave64_tile16", "flambeau_mmq_q8_0_wave64_tile16_q8_1", 16),
        ],
        other => bail!("autotune_mmq: --dtype {other} not wired (Q8_0 wave64 family only)"),
    };
    let avail: Vec<&(&str, &str, u32)> =
        cands.iter().filter(|(s, _, _)| B::image(s).is_some()).collect();
    if avail.is_empty() {
        bail!("autotune_mmq {backend}/{arch} {dtype}: no candidate kernel compiled");
    }
    println!(
        "autotune_mmq {backend}/{arch} {dtype}: n_out={n_out} k={k}, {} candidate(s)",
        avail.len()
    );
    for &m in m_grid {
        let mut times: Vec<(&str, f64)> = Vec::new();
        for &&(stem, entry, tile_n) in &avail {
            let us = time_mmq::<B>(&dev, (stem, entry, tile_n), m, n_out, k, iters)?;
            println!("  M={m:<5} {stem:<34} {us:9.2} us");
            times.push((stem, us));
        }
        let (ws, wt) = times.iter().copied().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        println!("    -> winner M={m}: {ws} ({wt:.2} us)");
    }
    Ok(())
}

/// Time every compiled `candidate` variant across the (m, k) grid, pick the
/// fastest per shape, print the ranking, and — when `emit` — write the winning
/// rows to `dispatch/<backend>/<arch>.toml`. Candidates whose kernel image
/// isn't compiled for this backend are skipped (so CUDA ranks only the ported
/// variants).
/// What `autotune` does with its measurement.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AutotuneMode {
    /// Rank + print only.
    DryRun,
    /// Write the winners into `dispatch/<backend>/<arch>.toml`.
    Emit,
    /// CI ratchet — diff the committed table against the fresh measurement and
    /// fail (non-zero exit) on drift. Does not write.
    Check,
}

/// A challenger must beat the committed incumbent by more than this fraction to
/// count as drift — keeps measurement noise from churning the table (AUTOTUNE.md
/// hysteresis).
const AUTOTUNE_HYSTERESIS: f64 = 0.03;

/// Whether `impl_id` has a green cert on disk for this arch — the eligibility
/// filter (AUTOTUNE.md): perf only ranks among already-certified kernels, and
/// the ratchet flags an incumbent that lost its cert.
fn has_green_cert(repo_root: &Path, backend: &str, arch: &str, impl_id: &str) -> bool {
    let path = repo_root.join(Cert::cert_path(backend, arch, impl_id));
    Cert::read_from_disk(&path).map(|c| c.pass && c.impl_id == impl_id).unwrap_or(false)
}

/// Whether the row's `shape.m` predicate covers `m`. Handles the forms the
/// dispatch tables use: `"any"`, `">=N"`, and inclusive `"lo..hi"` (both
/// `"lo..any"` and bare `"N"` also accepted).
fn m_range_contains(row: &crate::dispatch::DispatchRow, m: usize) -> bool {
    let Some(s) = row.shape.get("m").and_then(|v| v.as_str()) else { return true };
    let s = s.trim();
    if s == "any" {
        return true;
    }
    if let Some(n) = s.strip_prefix(">=") {
        return n.trim().parse().map(|lo: usize| m >= lo).unwrap_or(false);
    }
    if let Some((lo, hi)) = s.split_once("..") {
        let lo: usize = lo.trim().parse().unwrap_or(1);
        let hi: usize =
            if hi.trim() == "any" { usize::MAX } else { hi.trim().parse().unwrap_or(usize::MAX) };
        return m >= lo && m <= hi;
    }
    s.parse().map(|n: usize| n == m).unwrap_or(false)
}

pub fn autotune<B: BenchBackend>(
    arch: &str,
    candidates: &[Dtype],
    m_grid: &[usize],
    k_grid: &[usize],
    repo_root: &Path,
    mode: AutotuneMode,
) -> Result<()> {
    let dev = B::new_device(0)?;
    let backend = dev.backend();
    let iters = 50;

    // Eligibility = compiled AND certified. The cert is the gate (AUTOTUNE.md):
    // an uncertified kernel never reaches the ranking, fast or not.
    let compiled: Vec<Dtype> = candidates
        .iter()
        .copied()
        .filter(|d| B::image(d.kernel_stem()).is_some())
        .collect();
    let avail: Vec<Dtype> = compiled
        .iter()
        .copied()
        .filter(|d| has_green_cert(repo_root, backend, arch, d.impl_id()))
        .collect();
    for d in compiled.iter().filter(|d| !avail.contains(d)) {
        println!("  (dropping {} — no green cert for {backend}/{arch})", d.impl_id());
    }
    if avail.is_empty() {
        bail!("autotune {backend}/{arch}: no certified candidate kernels (run `sweep` first)");
    }
    let names: Vec<&str> = avail.iter().map(|d| d.impl_id()).collect();
    println!(
        "autotune {backend}/{arch}: {} certified candidate(s): {}",
        avail.len(),
        names.join(", ")
    );

    // The runtime dispatch routes on `m` only (KernelDescriptor.m_range; there
    // is no k-range), so a candidate that wins at one k and loses at another
    // can't be expressed — emitting both would be two rows matching the same
    // shape. The winner for an `m` is therefore the candidate with the best
    // total time across the whole k-grid; per-(m,k) timings still print so a
    // k-dependent split stays visible in the diagnostic output.
    let mut winners: Vec<(usize, Dtype)> = Vec::new();
    let mut per_m: Vec<(usize, Vec<(Dtype, f64)>)> = Vec::new();
    for &m in m_grid {
        let mut totals: Vec<(Dtype, f64)> = avail.iter().map(|&d| (d, 0.0)).collect();
        for &k in k_grid {
            for t in &mut totals {
                let us = time_mmvq::<B>(&dev, t.0, m, k, iters)?;
                println!("  m={m:<5} k={k:<6} {:<34} {us:8.2} us", t.0.impl_id());
                t.1 += us;
            }
        }
        let (wd, wtot) = totals
            .iter()
            .copied()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("at least one candidate");
        println!(
            "    -> winner m={m} (best total over {} k): {} ({wtot:.2} us)",
            k_grid.len(),
            wd.impl_id()
        );
        winners.push((m, wd));
        per_m.push((m, totals));
    }

    match mode {
        AutotuneMode::Emit => {
            let path = emit_dispatch(arch, backend, &winners, repo_root)?;
            println!("emitted {}", path.display());
        }
        AutotuneMode::Check => check_dispatch(arch, backend, repo_root, &per_m)?,
        AutotuneMode::DryRun => println!(
            "(dry run — pass --emit to write / --check to ratchet dispatch/{backend}/{arch}.toml)"
        ),
    }
    Ok(())
}

/// Ratchet the committed dispatch table against a fresh measurement. For each
/// (dtype, grid-m) the committed table routes through an incumbent impl; flag
/// the row when (a) the incumbent lost its cert, or (b) a certified candidate
/// beats it by more than the hysteresis margin. `Err` (non-zero CLI exit) iff
/// any row drifted — this is the CI signal.
fn check_dispatch(
    arch: &str,
    backend: &str,
    repo_root: &Path,
    per_m: &[(usize, Vec<(Dtype, f64)>)],
) -> Result<()> {
    let path = repo_root.join(format!("dispatch/{backend}/{arch}.toml"));
    let table = crate::dispatch::DispatchTable::load(&path)?;
    let mut issues: Vec<String> = Vec::new();

    for (m, totals) in per_m {
        let dtype_name = totals[0].0.name();
        let Some(row) = table.qmatmul.iter().find(|r| r.dtype == dtype_name && m_range_contains(r, *m))
        else {
            // No committed row for this grid-m → the runtime default kernel
            // handles it; autotune doesn't override the default, so this is not
            // drift, just an untuned shape.
            println!("  (note) {dtype_name} m={m}: no committed row — default path, not autotuned");
            continue;
        };
        let inc_impl = row.r#impl.as_str();

        if !has_green_cert(repo_root, backend, arch, inc_impl) {
            issues.push(format!("{dtype_name} m={m}: incumbent {inc_impl} has no green cert"));
        }

        let (fastest_d, fastest_t) =
            totals.iter().copied().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();

        // Compare kernels, not labels: the launch layer aliases some dispatch
        // impl_ids (e.g. Q8_0 `single_row` → `t128_vdr2`), so the incumbent is
        // resolved through its recipe; a challenger is the kernel it timed.
        let inc_live = flambeau_ops::qmatmul::recipe_stem_for(inc_impl).ok();
        let inc_time = inc_live
            .and_then(|s| totals.iter().find(|(d, _)| d.kernel_stem() == s).map(|&(_, t)| t))
            .or_else(|| totals.iter().find(|(d, _)| d.impl_id() == inc_impl).map(|&(_, t)| t));
        match inc_time {
            Some(inc_t)
                if Some(fastest_d.kernel_stem()) != inc_live
                    && fastest_t < inc_t * (1.0 - AUTOTUNE_HYSTERESIS) =>
            {
                issues.push(format!(
                    "{dtype_name} m={m}: {} is {:.2}x faster than incumbent {inc_impl} (>{:.0}% margin) — table stale",
                    fastest_d.impl_id(),
                    inc_t / fastest_t,
                    AUTOTUNE_HYSTERESIS * 100.0
                ));
            }
            // Same live kernel (alias), within margin, or incumbent absent from
            // the pool → no drift; the cert check above still gates the row.
            _ => {}
        }
    }

    if issues.is_empty() {
        println!(
            "autotune --check {backend}/{arch}: OK — committed dispatch matches measurement (≤{:.0}% hysteresis)",
            AUTOTUNE_HYSTERESIS * 100.0
        );
        Ok(())
    } else {
        for i in &issues {
            println!("  DRIFT: {i}");
        }
        bail!(
            "autotune --check {backend}/{arch}: {} dispatch row(s) drifted from measurement",
            issues.len()
        )
    }
}

/// Hand-format the autotuned `[[qmatmul]]` rows (one per winning dtype, with an
/// m-range predicate) into `dispatch/<backend>/<arch>.toml`. `DispatchRow` is
/// deserialize-only, so the table is generated text, not serde output.
fn emit_dispatch(
    arch: &str,
    backend: &str,
    winners: &[(usize, Dtype)],
    repo_root: &Path,
) -> Result<PathBuf> {
    use std::collections::BTreeMap;

    // One `[[qmatmul]]` block per dtype, keyed by `dtype` name so the table is
    // a merge: re-emit only the dtypes in this run, preserving rows for any
    // other dtype already in the table (the table is built up one dtype at a
    // time and must not clobber prior rows). Within a dtype the winners are
    // turned into DISJOINT contiguous m-runs: the impl that wins at grid point
    // `m` holds for `[m, next_grid_m)`, the last run is open-ended, and
    // consecutive same-impl points merge. This stays correct even when the
    // winning impl is non-monotonic in m (wins, loses, wins again → three
    // disjoint rows, not two overlapping ones).
    let mut by_dtype: BTreeMap<&'static str, Vec<(usize, &'static str)>> = BTreeMap::new();
    for &(m, d) in winners {
        by_dtype.entry(d.name()).or_default().push((m, d.impl_id()));
    }

    let mut blocks: BTreeMap<String, String> = BTreeMap::new();
    for (dtype, pts) in &by_dtype {
        let mut pts = pts.clone();
        pts.sort_by_key(|&(m, _)| m);
        // Collapse consecutive same-impl grid points into runs of (m_start, impl).
        let mut runs: Vec<(usize, &str)> = Vec::new();
        for (m, impl_id) in pts {
            if runs.last().map(|&(_, i)| i) != Some(impl_id) {
                runs.push((m, impl_id));
            }
        }
        let mut block = String::new();
        for (i, &(m_start, impl_id)) in runs.iter().enumerate() {
            let m_pred = if runs.len() == 1 {
                "any".to_string()
            } else if i + 1 < runs.len() {
                format!("{m_start}..{}", runs[i + 1].0)
            } else {
                format!("{m_start}..any")
            };
            write!(
                block,
                "\n[[qmatmul]]\n\
                 dtype   = \"{dtype}\"\n\
                 dtype_q = \"Q8_1\"\n\
                 shape   = {{ m = \"{m_pred}\", k = \"any\", n = \"any\" }}\n\
                 impl    = \"{impl_id}\"\n\
                 cert    = \"certs/{backend}/{arch}/{impl_id}.json\"\n\
                 notes   = \"autotuned\"\n"
            )
            .expect("write to String is infallible");
        }
        blocks.insert((*dtype).to_string(), block);
    }

    let path = repo_root.join(format!("dispatch/{backend}/{arch}.toml"));
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let spliced = splice_qmatmul_blocks(&existing, &blocks, backend, arch);
    std::fs::write(&path, &spliced).with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

/// Replace, in place, the `[[qmatmul]]` rows for each re-tuned dtype in `new`
/// (dtype → block text) while preserving every other byte of the table: other
/// dtypes' `[[qmatmul]]` rows, and all non-`[[qmatmul]]` sections
/// (`[[qmatmul_mmq]]`, attention, rope, MoE, …). The legacy whole-file rewrite
/// emitted only `[[qmatmul]]` blocks and silently dropped the rest — fatal for a
/// hand-maintained multi-section table re-emitting a single dtype.
fn splice_qmatmul_blocks(
    existing: &str,
    new: &std::collections::BTreeMap<String, String>,
    backend: &str,
    arch: &str,
) -> String {
    if existing.trim().is_empty() {
        let mut s = format!(
            "# Dispatch table — {backend} / {arch}. Autotuned by `flambeau autotune`.\n"
        );
        for b in new.values() {
            s.push_str(b);
        }
        return s;
    }

    let section_dtype = |sec: &[&str]| -> Option<String> {
        sec.iter().find_map(|l| {
            l.trim()
                .strip_prefix("dtype")?
                .trim_start()
                .strip_prefix('=')
                .map(|v| v.trim().trim_matches('"').to_string())
        })
    };

    let lines: Vec<&str> = existing.lines().collect();
    let mut out: Vec<String> = Vec::new();
    let mut spliced: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut i = 0;
    while i < lines.len() && !lines[i].trim_start().starts_with("[[") {
        out.push(lines[i].to_string());
        i += 1;
    }
    while i < lines.len() {
        let header = lines[i].trim().to_string();
        let start = i;
        i += 1;
        while i < lines.len() && !lines[i].trim_start().starts_with("[[") {
            i += 1;
        }
        let section = &lines[start..i];
        let dtype = section_dtype(section);
        let is_retuned = header == "[[qmatmul]]"
            && dtype.as_deref().map(|d| new.contains_key(d)).unwrap_or(false);
        if is_retuned {
            let dt = dtype.unwrap();
            // Emit the new rows on the first hit; drop subsequent rows of the
            // same re-tuned dtype (the new block replaces all of them).
            if spliced.insert(dt.clone()) {
                for l in new[&dt].trim_matches('\n').lines() {
                    out.push(l.to_string());
                }
                out.push(String::new());
            }
        } else {
            for l in section {
                out.push(l.to_string());
            }
        }
    }

    // A re-tuned dtype with no existing row → append its block.
    for (dt, b) in new {
        if !spliced.contains(dt) {
            for l in b.trim_matches('\n').lines() {
                out.push(l.to_string());
            }
            out.push(String::new());
        }
    }

    let mut s = out.join("\n");
    s.push('\n');
    s
}

// ---- helpers (shared with the ad-hoc tests) -------------------------------

fn seeded_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 24) as u8
        })
        .collect()
}

/// Override block-level scales so the F32 accumulation stays in the noise-
/// floor envelope the `1e-2 * sqrt(K/128)` tolerance assumes. Real GGUF
/// quants have these small by construction; random bytes don't.
fn tame_scales(dtype: Dtype, raw: Vec<u8>) -> Vec<u8> {
    let mut r = raw;
    let bs = dtype.block_size_bytes();
    let nblocks = r.len() / bs;
    for i in 0..nblocks {
        let block = &mut r[i * bs..(i + 1) * bs];
        match dtype {
            Dtype::Q8_0 | Dtype::Q8_0T128 | Dtype::Q8_0T128VDR2 | Dtype::Q8_0R2 | Dtype::Q8_0R4 | Dtype::Q8_0Dp4a => {
                // BlockQ8_0: d (f16, 2 bytes) + qs[32]. Scale ≈ 0.01 to 0.11.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Q4_0 | Dtype::Q4_0T128 | Dtype::Q4_0T128VDR2 => {
                // BlockQ4_0: d (f16, 0..2) + qs[16] nibbles. Same scale envelope.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Q5_0 => {
                // BlockQ5_0: d (f16, 0..2) + qh[4] + qs[16]. Cap d only.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Q5_1 => {
                // BlockQ5_1: d (f16, 0..2) + m (f16, 2..4) + qh[4] + qs[16].
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                let m = f16::from_f32((block[1] as f32 / 255.0) * 0.05);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&m.to_bits().to_le_bytes());
            }
            Dtype::Q2K | Dtype::Q2KR2 | Dtype::Q2KDp4a | Dtype::Q2KR2Dp4a => {
                // BlockQ2K: scales[16] + qs[64] + d (f16 @ 80) + dmin (f16 @ 82).
                // Keep scale-low-nibble in 0..15 (random already); cap d & dmin small.
                let d = f16::from_f32((block[80] as f32 / 255.0) * 0.05 + 0.005);
                let dmin = f16::from_f32((block[81] as f32 / 255.0) * 0.02);
                block[80..82].copy_from_slice(&d.to_bits().to_le_bytes());
                block[82..84].copy_from_slice(&dmin.to_bits().to_le_bytes());
            }
            Dtype::Q3K | Dtype::Q3KR2 | Dtype::Q3KDp4a | Dtype::Q3KDp4aT128Vdr2 | Dtype::Q3KR2Dp4a => {
                // BlockQ3K: hmask[32] + qs[64] + scales[12] + d (f16 at offset 108..110).
                let d_off = QK_K / 8 + QK_K / 4 + 12;
                for s in &mut block[(QK_K / 8 + QK_K / 4)..(QK_K / 8 + QK_K / 4 + 12)] {
                    *s = (*s as i32 % 32) as u8;
                }
                let d = f16::from_f32((block[d_off] as f32 / 255.0) * 0.05 + 0.005);
                block[d_off..d_off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Q4K | Dtype::Q5K | Dtype::Q4KR2 | Dtype::Q4KR2Dp4a | Dtype::Q4KDp4aT128Vdr2 | Dtype::Q5KR2 | Dtype::Q5KDp4a | Dtype::Q5KDp4aT128Vdr2 | Dtype::Q5KR2Dp4a => {
                // d (0..2), dmin (2..4). Same treatment as Q4_K test.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                let dmin = f16::from_f32((block[1] as f32 / 255.0) * 0.05);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            }
            Dtype::Q6K | Dtype::Q6KR4 | Dtype::Q6KDP4A => {
                // BlockQ6K: ql[128] + qh[64] + scales[16] i8 + d (f16 at offset 208..210).
                // Scale range: map each scale byte into [-32, 32] for plausibility.
                let scales_off = QK_K / 2 + QK_K / 4; // 128+64 = 192
                for s in &mut block[scales_off..scales_off + QK_K / 16] {
                    let sv = (*s as i32 % 65) - 32;
                    *s = sv as u8;
                }
                let d_off = scales_off + QK_K / 16;
                let d = f16::from_f32((block[d_off] as f32 / 255.0) * 0.05 + 0.01);
                block[d_off..d_off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Q4_1 | Dtype::Q4_1Dp4a | Dtype::Q4_1R2 | Dtype::Q4_1R2DP4A | Dtype::Q4_1T128 => {
                // BlockQ4_1: d (f16, 0..2) + m (f16, 2..4) + qs[16] (4..20).
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.1 + 0.01);
                let m = f16::from_f32((block[1] as f32 / 255.0) * 0.05);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&m.to_bits().to_le_bytes());
            }
            Dtype::Q8K => {
                // BlockQ8K: d (f32, 0..4) + qs[256] + bsums[16] (i16).
                // Keep d small to bound element magnitudes (qs is raw i8).
                let d = (block[0] as f32 / 255.0) * 0.02 + 0.002;
                block[0..4].copy_from_slice(&d.to_le_bytes());
            }
            Dtype::Iq4Nl | Dtype::Iq4NlR2 | Dtype::Iq4NlDp4a | Dtype::Iq4NlDp4aT128Vdr2 => {
                // BlockIq4Nl: d (f16, 0..2) + qs[16]. Keep d small —
                // KVALUES_IQ4NL values reach ±127 so an unbounded d would
                // overflow F32 accumulation in long-K shapes.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.05 + 0.005);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Iq4Xs | Dtype::Iq4XsR2 | Dtype::Iq4XsDp4a | Dtype::Iq4XsDp4aT128Vdr4 => {
                // BlockIq4Xs: d (f16, 0..2) + scales_h (u16, 2..4)
                // + scales_l[4] (4..8) + qs[128] (8..136). Per-sub-block
                // signed 6-bit scale = (low4 | high2 << 4) - 32 ∈ [-32, 31].
                // Random bytes already give a uniform distribution over
                // [-32, 31] — no targeted clamping needed.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.02 + 0.002);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Iq3Xxs | Dtype::Iq3XxsR2 | Dtype::Iq3XxsDp4a => {
                // BlockIq3Xxs: d (f16, 0..2) + qs[96]. Grid magnitudes
                // ≤ 0x3e (62); scale factor `(0.5 + (aux32>>28))*0.5` ≤ 7.75.
                // Cap d very tight to bound F32 accumulation at k=15360.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.005 + 0.0005);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Iq3S | Dtype::Iq3SR2 | Dtype::Iq3SDp4a | Dtype::Iq3SDp4aT128Vdr2 => {
                // BlockIq3S: d (f16, 0..2) + qs[64] + qh[8] + signs[32]
                // + scales[4]. Per-sub-block scale = 1+2*nibble ∈ [1, 31].
                // Even tighter d cap due to the higher scale range.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.001 + 0.0001);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Iq2Xxs | Dtype::Iq2XxsR2 | Dtype::Iq2XxsDp4a => {
                // Per-block 4-bit scale (top of aux32) and grid magnitudes
                // up to ~0x3e (62); scale chain (0.5+s) * 0.25 max ~3.875.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.005 + 0.0005);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Iq2Xs | Dtype::Iq2XsR2 | Dtype::Iq2XsDp4a => {
                // Same (0.5 + nibble) * 0.25 scaling as IQ2_XXS — cap d
                // similarly. Random scale-nibbles are fine.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.005 + 0.0005);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Iq2S | Dtype::Iq2SR2 | Dtype::Iq2SDp4a => {
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.005 + 0.0005);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Iq1S | Dtype::Iq1SR2 | Dtype::Iq1SDp4a => {
                // IQ1_S scale = (2 * (qh>>12)&7 + 1), max 15. Cap d tighter.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.001 + 0.0001);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Iq1M | Dtype::Iq1MR2 | Dtype::Iq1MDp4a => {
                // IQ1_M has no per-block d — it's reassembled from the top
                // nibble of each of 4 u16 scale-words at scales[0..8].
                // Pin those nibbles so the reassembled d_bits resolves to
                // a small positive fp16 (~1e-3), keeping F32 dot precision
                // at k=15360.
                //   d_bits = (sc[0]>>12) | ((sc[1]>>8)&0xF0)
                //          | ((sc[2]>>4)&0xF00) | (sc[3] & 0xF000)
                // Target d_bits = 0x0850. Distribute nibbles 0x0, 0x5, 0x8, 0x0.
                let pin_nib = [0x0u16, 0x5, 0x8, 0x0];
                let sc_off = QK_K / 8 + QK_K / 16; // scales array offset in BlockIq1M (no d)
                for (i, &nib) in pin_nib.iter().enumerate() {
                    let off = sc_off + 2 * i;
                    let lo = u16::from_le_bytes([block[off], block[off + 1]]);
                    let new = (lo & 0x0FFF) | (nib << 12);
                    block[off..off + 2].copy_from_slice(&new.to_le_bytes());
                }
            }
        }
    }
    r
}

fn quantize_q8_1_roundtrip(xs: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; xs.len()];
    for i in 0..(xs.len() / QK8) {
        let block = &xs[i * QK8..(i + 1) * QK8];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for (j, &v) in block.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i32;
            out[i * QK8 + j] = (q as f32) * d;
        }
    }
    out
}

fn reference_matmul(weights_f32: &[f32], y_f32: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m];
    for r in 0..m {
        let row = &weights_f32[r * k..(r + 1) * k];
        let mut acc = 0.0f64;
        for j in 0..k {
            acc += (row[j] * y_f32[j]) as f64;
        }
        out[r] = acc as f32;
    }
    out
}

fn cert_tol(_k: usize) -> f32 {
    // 3e-2 is the measured worst-case across the grid on gfx906 for the
    // single-row on-the-fly-dequant kernels — dominated by Q8_1 activation
    // quant noise on cancellation-heavy output rows. The `abs_floor` inside
    // `max_rel_err` absorbs the sqrt(K) scaling, so the bar stays flat.
    // Tighter bounds follow once MMVQ moves to the dp4a / multi-row DPP path.
    3e-2
}

#[cfg(test)]
mod autotune_tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn splice_replaces_only_retuned_qmatmul_rows() {
        let existing = "# header\n\n\
            [[qmatmul]]\ndtype = \"Q8_0\"\nimpl  = \"old_single_row\"\n\n\
            [[qmatmul]]\ndtype = \"Q4_K\"\nimpl  = \"q4k_keep\"\n\n\
            [[qmatmul_mmq]]\ndtype = \"Q8_0\"\nimpl  = \"q8_mmq_keep\"\n\n\
            [[rope]]\nimpl = \"rope_keep\"\n";
        let mut new = BTreeMap::new();
        new.insert(
            "Q8_0".to_string(),
            "\n[[qmatmul]]\ndtype   = \"Q8_0\"\nimpl    = \"new_dp4a\"\n".to_string(),
        );
        let out = splice_qmatmul_blocks(existing, &new, "hip", "gfx906");
        assert!(out.contains("new_dp4a"));
        assert!(!out.contains("old_single_row"), "old Q8_0 decode row replaced");
        // Other dtype's decode row + the non-[[qmatmul]] sections survive.
        assert!(out.contains("q4k_keep"), "Q4_K decode row preserved");
        assert!(out.contains("[[qmatmul_mmq]]") && out.contains("q8_mmq_keep"), "mmq preserved");
        assert!(out.contains("[[rope]]") && out.contains("rope_keep"), "rope preserved");
    }

    #[test]
    fn m_range_contains_handles_table_forms() {
        let row = |m: &str| crate::dispatch::DispatchRow {
            dtype: "Q8_0".into(),
            dtype_q: None,
            shape: [("m".to_string(), toml::Value::String(m.into()))].into_iter().collect(),
            r#impl: "x".into(),
            cert: "c".into(),
            notes: None,
        };
        assert!(m_range_contains(&row("any"), 4096));
        assert!(m_range_contains(&row("1..3"), 3)); // inclusive upper
        assert!(!m_range_contains(&row("1..3"), 4));
        assert!(m_range_contains(&row(">=32"), 32));
        assert!(!m_range_contains(&row(">=32"), 31));
    }
}
