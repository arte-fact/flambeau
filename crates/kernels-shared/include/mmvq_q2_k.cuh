#pragma once
// Templated single-row Q2_K MMVQ (decode GEMV). Q2_K is a 256-element
// super-block: sixteen packed (4-bit scale, 4-bit min) pairs in `scales[16]`,
// 2 bits per element in `qs[64]`, fp16 `d`/`dmin`. Ports llama.cpp's
// `vec_dot_q2_K_q8_1_impl_mmvq`: 256 threads, 64 per super-block, 4 super-blocks
// per iter. Per element y = d·sc·q − dmin·m; one dp4a for the value term and
// one for the broadcast-min term (m·Σu via dp4a(m_packed, u, 0)). Backend-
// neutral: `dp4a` / the reduce from the arch header + shared headers; int loads
// go through `load_int_b2`. `OutT` is the destination dtype (F32 or F16-dst).
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<class OutT>
static __device__ __forceinline__ void mmvq_q2_k_dp4a_row(
    const flambeau_block_q2_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_superblocks_per_row)
{
    constexpr int THREADS        = 256;
    constexpr int THREADS_PER_SB = 64;
    constexpr int SB_PER_ITER    = THREADS / THREADS_PER_SB;
    constexpr int QI2_K = 16;
    constexpr int QR2_K = 4;
    constexpr int QI8_1 = 8;

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid        = threadIdx.x;
    const int sb_in_iter = tid / THREADS_PER_SB;
    const int lane_in_sb = tid & (THREADS_PER_SB - 1);
    const int iqs        = lane_in_sb >> 2;          // 0..15
    const int i          = lane_in_sb & 3;           // 0..3

    const int bq8_offset   = QR2_K * (iqs / QI8_1);  // 0 or 4
    const int scale_offset = iqs - (iqs & (QI8_1 - 1)) + ((iqs & (QI8_1 - 1)) / (QI8_1 / 2));

    const flambeau_block_q2_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;
    for (int sb = sb_in_iter; sb < n_superblocks_per_row; sb += SB_PER_ITER) {
        const flambeau_block_q2_K* bk = xrow + sb;

        const float d_sb    = (float) bk->d;
        const float dmin_sb = (float) bk->dmin;

        const int sc_byte  = bk->scales[scale_offset + 2 * i];
        const int sc_lo    = sc_byte & 0x0F;
        const int m_4bit   = (sc_byte >> 4) & 0x0F;
        const int m_packed = m_4bit * 0x01010101;

        const int v  = load_int_b2((const int8_t*) bk->qs, iqs);
        const int vi = (v >> (2 * i)) & 0x03030303;

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + bq8_offset + i;
        const int u_i = load_int_b2(ya->qs, iqs & (QI8_1 - 1));
        const float d_y = (float) ya->d;

        const int sumi_d = dp4a(vi, u_i, 0);
        const int sumi_m = dp4a(m_packed, u_i, 0);

        acc += d_y * (d_sb * (float) sc_lo * (float) sumi_d - dmin_sb * (float) sumi_m);
    }

    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}
