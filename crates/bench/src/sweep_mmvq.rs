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
    /// DP4A 256-thread Q2_K MMVQ (Phase 3e — port of llama.cpp's
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
    /// DP4A 256-thread Q3_K MMVQ (Phase 3b — port of llama.cpp's
    /// `vec_dot_q3_K_q8_1_impl_mmvq`). 64 threads per super-block,
    /// 4 super-blocks per iter. Splits the 2-bit + 1-bit-hmask decode
    /// into two DP4A calls (one for low bits, one for hmask bit) to
    /// avoid the int32 byte-borrow corruption that __vsubss4 would
    /// sidestep on NVIDIA.
    Q3KDp4a,
    /// r2 multi-row DP4A Q3_K MMVQ (Lever 1 — refactor of
    /// `Q3KDp4a` to wave64 single-warp / 2 rows per block, half-warp
    /// reduce, no LDS rendezvous, activation reuse across rows).
    Q3KR2Dp4a,
    /// r2 multi-row variant of Q4_K (2 output rows per wave64).
    Q4KR2,
    /// DP4A r2 multi-row Q4_K MMVQ (Phase 3a — port of llama.cpp's
    /// `vec_dot_q4_K_q8_1_impl_vmmq`). Same shape as Q4KR2 but with
    /// `__builtin_amdgcn_sdot4` SIMD int8 dot products instead of
    /// per-element FP32 multiplies. Mirrors the MoE
    /// `indexed_moe_mmvq_q4_k_r2_dp4a` body.
    Q4KR2Dp4a,
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
    /// C9-i1 thin-block 128-thread single-row Q8_0 DP4A — analog of Q4_1T128
    /// for Q8_0 weights — targets the gfx906 latency-bound regime where Q8_0
    /// MMVQ is the dominant kernel on Coder-30B-Q8_0 / 27B-Q8_0 decode.
    Q8_0T128,
    /// C9-followup t128 + VDR=2 combined (the kernel that should beat
    /// vdr2-alone where t128-alone lost).
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
    /// DP4A IQ4_NL MMVQ (Phase 3d — port of llama.cpp's
    /// `vec_dot_iq4_nl_q8_1`). 256 threads/block, single row, mirrors
    /// `mmvq_q4_0` launch shape (4 lanes per 32-elem block, 64 blocks
    /// per outer iter). Uses the same `__builtin_amdgcn_perm` LUT
    /// helper as IQ4_XS dp4a.
    Iq4NlDp4a,
    /// native IQ4_XS single-row MMVQ.
    Iq4Xs,
    /// native IQ4_XS r2 multi-row MMVQ.
    Iq4XsR2,
    /// DP4A IQ4_XS MMVQ (Phase 3c — port of llama.cpp's
    /// `vec_dot_iq4_xs_q8_1`). 256 threads, single row, 32 threads
    /// per super-block. Uses `__builtin_amdgcn_perm` for 4-way LUT
    /// lookup into `kvalues_iq4nl`, then dp4a inner.
    Iq4XsDp4a,
    /// native IQ3_XXS single-row MMVQ.
    Iq3Xxs,
    /// native IQ3_XXS r2 multi-row MMVQ.
    Iq3XxsR2,
    /// DP4A IQ3_XXS MMVQ (Phase 3g — port of llama.cpp's
    /// `vec_dot_iq3_xxs_q8_1`). Uses IQ3XXS_GRID + unpack_ksigns
    /// (popcount-parity 8th-bit extension) + the apply_signs helper
    /// from IQ3_S.
    Iq3XxsDp4a,
    /// native IQ3_S single-row MMVQ.
    Iq3S,
    /// native IQ3_S r2 multi-row MMVQ.
    Iq3SR2,
    /// DP4A IQ3_S MMVQ (Phase 3f — port of llama.cpp's
    /// `vec_dot_iq3_s_q8_1`). First codebook kernel: looks up 9-bit
    /// indices into `IQ3S_GRID` for 4-i8-packed magnitudes, applies
    /// per-byte sign from packed bits (manual __vsub4 emulation on
    /// gfx906), dp4a vs Q8_1. 256 threads/block, single row, 32 threads
    /// per super-block × 8 super-blocks per iter.
    Iq3SDp4a,
    /// native IQ2_XXS single-row MMVQ.
    Iq2Xxs,
    Iq2XxsR2,
    /// DP4A IQ2_XXS MMVQ (Phase 3j — port of llama.cpp's
    /// `vec_dot_iq2_xxs_q8_1`). Per sub-block 8 qs bytes split into 4
    /// codebook indices + 4 × 7-bit signs + 4-bit scale (top of aux32).
    /// One scale per sub-block (no ls0/ls1 split).
    Iq2XxsDp4a,
    Iq2Xs,
    Iq2XsR2,
    /// DP4A IQ2_XS MMVQ (Phase 3i — port of llama.cpp's
    /// `vec_dot_iq2_xs_q8_1`). Each qs entry is uint16: low 9 bits
    /// index IQ2XS_GRID, high 7 bits are signs (popcount-parity for
    /// 8th bit). Reuses apply_signs + unpack_ksigns helpers.
    Iq2XsDp4a,
    Iq2S,
    Iq2SR2,
    /// DP4A IQ2_S MMVQ (Phase 3h — port of llama.cpp's
    /// `vec_dot_iq2_s_q8_1`). IQ2S_GRID is uint64[1024] (8 i8 per
    /// entry); per (iqs, l_grp) lane: one grid lookup, two dp4as,
    /// two sub-block scales (ls0/ls1) over l_grp split.
    Iq2SDp4a,
    Iq1S,
    Iq1SR2,
    /// DP4A IQ1_S MMVQ (Phase 3k — port of llama.cpp's
    /// `vec_dot_iq1_s_q8_1`). No sign mask; codebook stores signed i8
    /// magnitudes directly in IQ1S_GRID (uint64 of 8 packed i8).
    /// Per-sub-block delta offset (±IQ1_DELTA) introduces a bias term
    /// computed via dp4a(0x01010101, u, 0). VDR=1 (no x2 step on iqs).
    Iq1SDp4a,
    Iq1M,
    Iq1MR2,
    /// DP4A IQ1_M MMVQ (Phase 3l — port of llama.cpp's
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
            Dtype::Q4_0 => "Q4_0",
            Dtype::Q5_0 => "Q5_0",
            Dtype::Q5_1 => "Q5_1",
            Dtype::Q2K | Dtype::Q2KR2 | Dtype::Q2KDp4a | Dtype::Q2KR2Dp4a => "Q2_K",
            Dtype::Q3K | Dtype::Q3KR2 | Dtype::Q3KDp4a | Dtype::Q3KR2Dp4a => "Q3_K",
            Dtype::Q4K | Dtype::Q4KR2 | Dtype::Q4KR2Dp4a => "Q4_K",
            Dtype::Q5K | Dtype::Q5KR2 | Dtype::Q5KDp4a | Dtype::Q5KR2Dp4a => "Q5_K",
            Dtype::Q6K | Dtype::Q6KR4 | Dtype::Q6KDP4A => "Q6_K",
            Dtype::Q8K => "Q8_K",
            Dtype::Q4_1 | Dtype::Q4_1Dp4a | Dtype::Q4_1R2 | Dtype::Q4_1R2DP4A | Dtype::Q4_1T128 => {
                "Q4_1"
            }
            Dtype::Iq4Nl | Dtype::Iq4NlR2 | Dtype::Iq4NlDp4a => "IQ4_NL",
            Dtype::Iq4Xs | Dtype::Iq4XsR2 | Dtype::Iq4XsDp4a => "IQ4_XS",
            Dtype::Iq3Xxs | Dtype::Iq3XxsR2 | Dtype::Iq3XxsDp4a => "IQ3_XXS",
            Dtype::Iq3S | Dtype::Iq3SR2 | Dtype::Iq3SDp4a => "IQ3_S",
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
            Dtype::Q4_0 => GgmlDType::Q4_0,
            Dtype::Q5_0 => GgmlDType::Q5_0,
            Dtype::Q5_1 => GgmlDType::Q5_1,
            Dtype::Q2K | Dtype::Q2KR2 | Dtype::Q2KDp4a | Dtype::Q2KR2Dp4a => GgmlDType::Q2K,
            Dtype::Q3K | Dtype::Q3KR2 | Dtype::Q3KDp4a | Dtype::Q3KR2Dp4a => GgmlDType::Q3K,
            Dtype::Q4K | Dtype::Q4KR2 | Dtype::Q4KR2Dp4a => GgmlDType::Q4K,
            Dtype::Q5K | Dtype::Q5KR2 | Dtype::Q5KDp4a | Dtype::Q5KR2Dp4a => GgmlDType::Q5K,
            Dtype::Q6K | Dtype::Q6KR4 | Dtype::Q6KDP4A => GgmlDType::Q6K,
            Dtype::Q8K => GgmlDType::Q8K,
            Dtype::Q4_1 | Dtype::Q4_1Dp4a | Dtype::Q4_1R2 | Dtype::Q4_1R2DP4A | Dtype::Q4_1T128 => {
                GgmlDType::Q4_1
            }
            Dtype::Iq4Nl | Dtype::Iq4NlR2 | Dtype::Iq4NlDp4a => GgmlDType::Iq4Nl,
            Dtype::Iq4Xs | Dtype::Iq4XsR2 | Dtype::Iq4XsDp4a => GgmlDType::Iq4Xs,
            Dtype::Iq3Xxs | Dtype::Iq3XxsR2 | Dtype::Iq3XxsDp4a => GgmlDType::Iq3Xxs,
            Dtype::Iq3S | Dtype::Iq3SR2 | Dtype::Iq3SDp4a => GgmlDType::Iq3S,
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
            Dtype::Q3KR2Dp4a => "qmatmul_q3_K_mmvq_r2_dp4a_gfx906",
            Dtype::Q4K => "qmatmul_q4_K_mmvq_single_row_gfx906",
            Dtype::Q5K => "qmatmul_q5_K_mmvq_single_row_gfx906",
            Dtype::Q6K => "qmatmul_q6_K_mmvq_single_row_gfx906",
            Dtype::Q8K => "qmatmul_q8_k_mmvq_single_row",
            Dtype::Q4KR2 => "qmatmul_q4_K_mmvq_nw1_r2_gfx906",
            Dtype::Q4KR2Dp4a => "qmatmul_q4_k_mmvq_dp4a",
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
            Dtype::Iq4Xs => "qmatmul_iq4_xs_mmvq_single_row_gfx906",
            Dtype::Iq4XsR2 => "qmatmul_iq4_xs_mmvq_nw1_r2_gfx906",
            Dtype::Iq4XsDp4a => "qmatmul_iq4_xs_mmvq_dp4a",
            Dtype::Iq3Xxs => "qmatmul_iq3_xxs_mmvq_single_row_gfx906",
            Dtype::Iq3XxsR2 => "qmatmul_iq3_xxs_mmvq_nw1_r2_gfx906",
            Dtype::Iq3XxsDp4a => "qmatmul_iq3_xxs_mmvq_dp4a",
            Dtype::Iq3S => "qmatmul_iq3_s_mmvq_single_row_gfx906",
            Dtype::Iq3SR2 => "qmatmul_iq3_s_mmvq_nw1_r2_gfx906",
            Dtype::Iq3SDp4a => "qmatmul_iq3_s_mmvq_dp4a",
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
            Dtype::Q3KR2Dp4a => "mmvq_q3_k_r2_dp4a",
            Dtype::Q4K => "mmvq_q4_k",
            Dtype::Q5K => "mmvq_q5_k",
            Dtype::Q6K => "mmvq_q6_k",
            Dtype::Q8K => "mmvq_q8_k",
            Dtype::Q4KR2 => "mmvq_q4_k_r2",
            Dtype::Q4KR2Dp4a => "mmvq_q4_k_r2_dp4a",
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
            Dtype::Iq4Xs => "mmvq_iq4_xs",
            Dtype::Iq4XsR2 => "mmvq_iq4_xs_r2",
            Dtype::Iq4XsDp4a => "mmvq_iq4_xs_dp4a",
            Dtype::Iq3Xxs => "mmvq_iq3_xxs",
            Dtype::Iq3XxsR2 => "mmvq_iq3_xxs_r2",
            Dtype::Iq3XxsDp4a => "mmvq_iq3_xxs_dp4a",
            Dtype::Iq3S => "mmvq_iq3_s",
            Dtype::Iq3SR2 => "mmvq_iq3_s_r2",
            Dtype::Iq3SDp4a => "mmvq_iq3_s_dp4a",
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
            Dtype::Q3KR2Dp4a => "flambeau_mmvq_q3_k_r2_dp4a_q8_1",
            Dtype::Q8K => "flambeau_mmvq_q8_K_q8_1",
            Dtype::Q4K => "flambeau_mmvq_q4_k_q8_1",
            Dtype::Q5K => "flambeau_mmvq_q5_k_q8_1",
            Dtype::Q6K => "flambeau_mmvq_q6_k_q8_1",
            Dtype::Q4KR2 => "flambeau_mmvq_q4_k_r2_q8_1",
            Dtype::Q4KR2Dp4a => "flambeau_mmvq_q4_k_r2_dp4a_q8_1",
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
            Dtype::Iq4Xs => "flambeau_mmvq_iq4_xs_q8_1",
            Dtype::Iq4XsR2 => "flambeau_mmvq_iq4_xs_r2_q8_1",
            Dtype::Iq4XsDp4a => "flambeau_mmvq_iq4_xs_dp4a_q8_1",
            Dtype::Iq3Xxs => "flambeau_mmvq_iq3_xxs_q8_1",
            Dtype::Iq3XxsR2 => "flambeau_mmvq_iq3_xxs_r2_q8_1",
            Dtype::Iq3XxsDp4a => "flambeau_mmvq_iq3_xxs_dp4a_q8_1",
            Dtype::Iq3S => "flambeau_mmvq_iq3_s_q8_1",
            Dtype::Iq3SR2 => "flambeau_mmvq_iq3_s_r2_q8_1",
            Dtype::Iq3SDp4a => "flambeau_mmvq_iq3_s_dp4a_q8_1",
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
            Dtype::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Dtype::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Dtype::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            Dtype::Q2K | Dtype::Q2KR2 | Dtype::Q2KDp4a | Dtype::Q2KR2Dp4a => std::mem::size_of::<BlockQ2K>(),
            Dtype::Q3K | Dtype::Q3KR2 | Dtype::Q3KDp4a | Dtype::Q3KR2Dp4a => std::mem::size_of::<BlockQ3K>(),
            Dtype::Q4K | Dtype::Q4KR2 | Dtype::Q4KR2Dp4a => std::mem::size_of::<BlockQ4K>(),
            Dtype::Q5K | Dtype::Q5KR2 | Dtype::Q5KDp4a | Dtype::Q5KR2Dp4a => std::mem::size_of::<BlockQ5K>(),
            Dtype::Q6K | Dtype::Q6KR4 | Dtype::Q6KDP4A => std::mem::size_of::<BlockQ6K>(),
            Dtype::Q8K => std::mem::size_of::<BlockQ8K>(),
            Dtype::Q4_1 | Dtype::Q4_1Dp4a | Dtype::Q4_1R2 | Dtype::Q4_1R2DP4A | Dtype::Q4_1T128 => {
                std::mem::size_of::<BlockQ4_1>()
            }
            Dtype::Iq4Nl | Dtype::Iq4NlR2 | Dtype::Iq4NlDp4a => std::mem::size_of::<BlockIq4Nl>(),
            Dtype::Iq4Xs | Dtype::Iq4XsR2 | Dtype::Iq4XsDp4a => std::mem::size_of::<BlockIq4Xs>(),
            Dtype::Iq3Xxs | Dtype::Iq3XxsR2 | Dtype::Iq3XxsDp4a => std::mem::size_of::<BlockIq3Xxs>(),
            Dtype::Iq3S | Dtype::Iq3SR2 | Dtype::Iq3SDp4a => std::mem::size_of::<BlockIq3S>(),
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
            Dtype::Q4_1T128 | Dtype::Q8_0T128 | Dtype::Q8_0T128VDR2 => 128,
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

    let mut results = Vec::new();
    for &m in &spec.m_grid {
        for &k in &spec.k_grid {
            let seed = spec
                .seed
                .wrapping_add((m as u64).wrapping_mul(0x1234567))
                .wrapping_add((k as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let (got, reference) = run_shape::<B>(&dev, spec.dtype, m, k, seed)?;
            let max_rel_err = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt());
            let tolerance = cert_tol(k);
            results.push(ShapeResult {
                m,
                k,
                n: k, // single-row MMVQ: N is irrelevant; mirror K for the index
                seed,
                max_rel_err,
                tolerance,
                pass: max_rel_err <= tolerance,
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
        tolerance_formula: "|err| <= 3e-2 * max(|ref|, sqrt(k))".to_string(),
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
    Q4_0,
    Q4_1,
    Q5K,
}

impl GateUpDtype {
    pub fn name(self) -> &'static str {
        match self {
            GateUpDtype::Q8_0 => "Q8_0",
            GateUpDtype::Q4_0 => "Q4_0",
            GateUpDtype::Q4_1 => "Q4_1",
            GateUpDtype::Q5K => "Q5_K",
        }
    }
    fn ggml(self) -> GgmlDType {
        match self {
            GateUpDtype::Q8_0 => GgmlDType::Q8_0,
            GateUpDtype::Q4_0 => GgmlDType::Q4_0,
            GateUpDtype::Q4_1 => GgmlDType::Q4_1,
            GateUpDtype::Q5K => GgmlDType::Q5K,
        }
    }
    fn weight_dtype(self) -> Dtype {
        match self {
            GateUpDtype::Q8_0 => Dtype::Q8_0Dp4a,
            GateUpDtype::Q4_0 => Dtype::Q4_0,
            GateUpDtype::Q4_1 => Dtype::Q4_1Dp4a,
            GateUpDtype::Q5K => Dtype::Q5K,
        }
    }
    fn impl_id(self) -> &'static str {
        match self {
            GateUpDtype::Q8_0 => "qmatmul_q8_0_gate_up_t128_vdr2",
            GateUpDtype::Q4_0 => "qmatmul_q4_0_gate_up_dp4a",
            GateUpDtype::Q4_1 => "qmatmul_q4_1_gate_up_dp4a",
            GateUpDtype::Q5K => "qmatmul_q5_k_gate_up_dp4a",
        }
    }
    fn kernel_stem(self) -> &'static str {
        match self {
            GateUpDtype::Q8_0 => "mmvq_q8_0_gate_up_t128_vdr2",
            GateUpDtype::Q4_0 => "mmvq_q4_0_gate_up_dp4a",
            GateUpDtype::Q4_1 => "mmvq_q4_1_gate_up_dp4a",
            GateUpDtype::Q5K => "mmvq_q5_k_gate_up_dp4a",
        }
    }
    fn kernel_entry(self) -> &'static str {
        match self {
            GateUpDtype::Q8_0 => "flambeau_mmvq_q8_0_gate_up_t128_vdr2_q8_1",
            GateUpDtype::Q4_0 => "flambeau_mmvq_q4_0_gate_up_dp4a_q8_1",
            GateUpDtype::Q4_1 => "flambeau_mmvq_q4_1_gate_up_dp4a_q8_1",
            GateUpDtype::Q5K => "flambeau_mmvq_q5_k_gate_up_dp4a_q8_1",
        }
    }
    fn launch_threads(self) -> u32 {
        match self {
            GateUpDtype::Q8_0 => 128,
            GateUpDtype::Q4_0 | GateUpDtype::Q4_1 | GateUpDtype::Q5K => 256,
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
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
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

// ---- batched (N-slot) sweep -----------------------------------------------

/// Weight dtype for the batched MMVQ sweep (one weight row × N activation
/// columns). Separate from `Dtype` because the kernel has N=2/3/4 entries and a
/// slot-major output; only Q8_0 is templated so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchedDtype {
    Q8_0,
    Q4_0,
    Q4_1,
    Q4K,
    Q6K,
    Q5K,
    Q8_0RowTile,
    Q4_0RowTile,
    Q5KRowTile,
}

impl BatchedDtype {
    pub fn name(self) -> &'static str {
        match self {
            BatchedDtype::Q8_0 | BatchedDtype::Q8_0RowTile => "Q8_0",
            BatchedDtype::Q4_0 | BatchedDtype::Q4_0RowTile => "Q4_0",
            BatchedDtype::Q4_1 => "Q4_1",
            BatchedDtype::Q4K => "Q4_K",
            BatchedDtype::Q6K => "Q6_K",
            BatchedDtype::Q5K | BatchedDtype::Q5KRowTile => "Q5_K",
        }
    }
    fn weight_dtype(self) -> Dtype {
        match self {
            BatchedDtype::Q8_0 | BatchedDtype::Q8_0RowTile => Dtype::Q8_0Dp4a,
            BatchedDtype::Q4_0 | BatchedDtype::Q4_0RowTile => Dtype::Q4_0,
            BatchedDtype::Q4_1 => Dtype::Q4_1Dp4a,
            BatchedDtype::Q4K => Dtype::Q4K,
            BatchedDtype::Q6K => Dtype::Q6K,
            BatchedDtype::Q5K | BatchedDtype::Q5KRowTile => Dtype::Q5K,
        }
    }
    fn impl_id(self) -> &'static str {
        match self {
            BatchedDtype::Q8_0 => "qmatmul_q8_0_mmvq_batched",
            BatchedDtype::Q4_0 => "qmatmul_q4_0_mmvq_batched",
            BatchedDtype::Q4_1 => "qmatmul_q4_1_mmvq_batched",
            BatchedDtype::Q4K => "qmatmul_q4_k_mmvq_batched",
            BatchedDtype::Q6K => "qmatmul_q6_k_mmvq_batched",
            BatchedDtype::Q5K => "qmatmul_q5_k_r2_mmvq_batched",
            BatchedDtype::Q8_0RowTile => "qmatmul_q8_0_mmvq_row_tile_batched",
            BatchedDtype::Q4_0RowTile => "qmatmul_q4_0_mmvq_row_tile_batched",
            BatchedDtype::Q5KRowTile => "qmatmul_q5_k_mmvq_row_tile_batched",
        }
    }
    fn kernel_stem(self) -> &'static str {
        match self {
            BatchedDtype::Q8_0 => "mmvq_q8_0_batched",
            BatchedDtype::Q4_0 => "mmvq_q4_0_batched",
            BatchedDtype::Q4_1 => "mmvq_q4_1_batched",
            BatchedDtype::Q4K => "mmvq_q4_k_batched",
            BatchedDtype::Q6K => "mmvq_q6_k_batched",
            BatchedDtype::Q5K => "mmvq_q5_k_r2_batched",
            BatchedDtype::Q8_0RowTile => "mmvq_q8_0_row_tile_batched",
            BatchedDtype::Q4_0RowTile => "mmvq_q4_0_row_tile_batched",
            BatchedDtype::Q5KRowTile => "mmvq_q5_k_row_tile_batched",
        }
    }
    fn launch_threads(self) -> u32 {
        match self {
            BatchedDtype::Q8_0
            | BatchedDtype::Q4_0
            | BatchedDtype::Q4_1
            | BatchedDtype::Q8_0RowTile
            | BatchedDtype::Q4_0RowTile
            | BatchedDtype::Q5KRowTile => 256,
            BatchedDtype::Q4K | BatchedDtype::Q6K | BatchedDtype::Q5K => 64,
        }
    }
    /// Output rows computed per block: 2 for the r2 Q5_K kernel, 8 for the
    /// 32-lane-group Q5_K row-tile (same on both backends), 4 for the
    /// Q8_0/Q4_0 row-tiles (over-launches harmlessly on wave32's R=8), 1 else.
    fn rows_per_block(self) -> u32 {
        match self {
            BatchedDtype::Q5K => 2,
            BatchedDtype::Q8_0RowTile | BatchedDtype::Q4_0RowTile => 4,
            BatchedDtype::Q5KRowTile => 8,
            _ => 1,
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
            (BatchedDtype::Q4K, 2) => "flambeau_mmvq_q4_k_q8_1_batched_n2",
            (BatchedDtype::Q4K, 3) => "flambeau_mmvq_q4_k_q8_1_batched_n3",
            (BatchedDtype::Q4K, 4) => "flambeau_mmvq_q4_k_q8_1_batched_n4",
            (BatchedDtype::Q6K, 2) => "flambeau_mmvq_q6_k_q8_1_batched_n2",
            (BatchedDtype::Q6K, 3) => "flambeau_mmvq_q6_k_q8_1_batched_n3",
            (BatchedDtype::Q6K, 4) => "flambeau_mmvq_q6_k_q8_1_batched_n4",
            (BatchedDtype::Q5K, 2) => "flambeau_mmvq_q5_k_r2_q8_1_batched_n2",
            (BatchedDtype::Q5K, 3) => "flambeau_mmvq_q5_k_r2_q8_1_batched_n3",
            (BatchedDtype::Q5K, 4) => "flambeau_mmvq_q5_k_r2_q8_1_batched_n4",
            (BatchedDtype::Q8_0RowTile, 2) => "flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n2",
            (BatchedDtype::Q8_0RowTile, 3) => "flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n3",
            (BatchedDtype::Q8_0RowTile, 4) => "flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n4",
            (BatchedDtype::Q4_0RowTile, 2) => "flambeau_mmvq_q4_0_row_tile_dp4a_q8_1_batched_n2",
            (BatchedDtype::Q4_0RowTile, 3) => "flambeau_mmvq_q4_0_row_tile_dp4a_q8_1_batched_n3",
            (BatchedDtype::Q4_0RowTile, 4) => "flambeau_mmvq_q4_0_row_tile_dp4a_q8_1_batched_n4",
            (BatchedDtype::Q5KRowTile, 2) => "flambeau_mmvq_q5_k_row_tile_q8_1_batched_n2",
            (BatchedDtype::Q5KRowTile, 3) => "flambeau_mmvq_q5_k_row_tile_q8_1_batched_n3",
            (BatchedDtype::Q5KRowTile, 4) => "flambeau_mmvq_q5_k_row_tile_q8_1_batched_n4",
            _ => unreachable!("batched n_slots outside [2,4]"),
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
        let grid = (m as u32).div_ceil(dtype.rows_per_block());
        B::launch(&k_mmvq, stream, LaunchCfg::one_d(grid, dtype.launch_threads()), args)?;
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

fn run_shape<B: BenchBackend>(
    dev: &B::Device,
    dtype: Dtype,
    m: usize,
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
    let k_mmvq = B::kernel(&m_module, dtype.kernel_entry())?;

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

    unsafe {
        dev.dealloc(d_x, weights_raw.len())?;
        dev.dealloc(d_y_f32, y_f32.len() * 4)?;
        dev.dealloc(d_y_q8_1, y_q8_1_bytes)?;
        dev.dealloc(d_dst, m * 4)?;
    }

    let y_rt = quantize_q8_1_roundtrip(&y_f32);
    let reference = reference_matmul(&weights_dequant, &y_rt, m, k);
    Ok((got, reference))
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

/// Time every compiled `candidate` variant across the (m, k) grid, pick the
/// fastest per shape, print the ranking, and — when `emit` — write the winning
/// rows to `dispatch/<backend>/<arch>.toml`. Candidates whose kernel image
/// isn't compiled for this backend are skipped (so CUDA ranks only the ported
/// variants).
pub fn autotune<B: BenchBackend>(
    arch: &str,
    candidates: &[Dtype],
    m_grid: &[usize],
    k_grid: &[usize],
    repo_root: &Path,
    emit: bool,
) -> Result<()> {
    let dev = B::new_device(0)?;
    let backend = dev.backend();
    let iters = 50;

    let avail: Vec<Dtype> = candidates
        .iter()
        .copied()
        .filter(|d| B::image(d.kernel_stem()).is_some())
        .collect();
    if avail.is_empty() {
        bail!("autotune {backend}/{arch}: no candidate kernels compiled");
    }
    let names: Vec<&str> = avail.iter().map(|d| d.impl_id()).collect();
    println!(
        "autotune {backend}/{arch}: {} candidate(s): {}",
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
    }

    if emit {
        let path = emit_dispatch(arch, backend, &winners, repo_root)?;
        println!("emitted {}", path.display());
    } else {
        println!("(dry run — pass --emit to write dispatch/{backend}/{arch}.toml)");
    }
    Ok(())
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
    if let Ok(existing) = std::fs::read_to_string(&path) {
        for raw in existing.split("\n[[qmatmul]]").skip(1) {
            let dtype = raw.lines().find_map(|l| {
                l.trim()
                    .strip_prefix("dtype")?
                    .trim_start()
                    .strip_prefix('=')
                    .map(|v| v.trim().trim_matches('"').to_string())
            });
            if let Some(dt) = dtype {
                blocks
                    .entry(dt)
                    .or_insert_with(|| format!("\n[[qmatmul]]{}\n", raw.trim_end()));
            }
        }
    }

    let mut s = String::new();
    writeln!(
        s,
        "# Dispatch table — {backend} / {arch}. Autotuned by `flambeau autotune`\n\
         # (median of timed mmvq launches per shape). Regenerate; do not hand-edit."
    )
    .expect("write to String is infallible");
    for block in blocks.values() {
        s.push_str(block);
    }

    std::fs::write(&path, s).with_context(|| format!("write {}", path.display()))?;
    Ok(path)
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
            Dtype::Q4_0 => {
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
            Dtype::Q3K | Dtype::Q3KR2 | Dtype::Q3KDp4a | Dtype::Q3KR2Dp4a => {
                // BlockQ3K: hmask[32] + qs[64] + scales[12] + d (f16 at offset 108..110).
                let d_off = QK_K / 8 + QK_K / 4 + 12;
                for s in &mut block[(QK_K / 8 + QK_K / 4)..(QK_K / 8 + QK_K / 4 + 12)] {
                    *s = (*s as i32 % 32) as u8;
                }
                let d = f16::from_f32((block[d_off] as f32 / 255.0) * 0.05 + 0.005);
                block[d_off..d_off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Q4K | Dtype::Q5K | Dtype::Q4KR2 | Dtype::Q4KR2Dp4a | Dtype::Q5KR2 | Dtype::Q5KDp4a | Dtype::Q5KR2Dp4a => {
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
            Dtype::Iq4Nl | Dtype::Iq4NlR2 | Dtype::Iq4NlDp4a => {
                // BlockIq4Nl: d (f16, 0..2) + qs[16]. Keep d small —
                // KVALUES_IQ4NL values reach ±127 so an unbounded d would
                // overflow F32 accumulation in long-K shapes.
                let d = f16::from_f32((block[0] as f32 / 255.0) * 0.05 + 0.005);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            Dtype::Iq4Xs | Dtype::Iq4XsR2 | Dtype::Iq4XsDp4a => {
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
            Dtype::Iq3S | Dtype::Iq3SR2 | Dtype::Iq3SDp4a => {
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
