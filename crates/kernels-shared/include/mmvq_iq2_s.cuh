#pragma once
// Templated single-row IQ2_S MMVQ (decode GEMV). IQ2_S (2.5 bpw) stores 8-bit
// codebook indices in `qs` extended by 2 bits from `qh` (→ `IQ2S_GRID`, 1024
// entries, uint64), explicit per-byte sign masks embedded at `qs[32+...]`, and
// 4-bit sub-block scales. Ports llama.cpp's `vec_dot_iq2_s_q8_1`: 256 threads,
// 32 per super-block. Reuses IQ2_XXS's identical sign helper; weight reads are
// single bytes, only Q8_1 int loads go through `load_int_b2`.
#include "mmvq_iq2_xxs.cuh"

template<class OutT>
static __device__ __forceinline__ void mmvq_iq2_s_dp4a_row(
    const flambeau_block_iq2_s* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_superblocks_per_row)
{
    constexpr int THREADS        = 256;
    constexpr int THREADS_PER_SB = 32;
    constexpr int SB_PER_ITER    = THREADS / THREADS_PER_SB;

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid        = threadIdx.x;
    const int sb_in_iter = tid >> 5;
    const int lane_in_sb = tid & 31;
    const int iqs_idx    = lane_in_sb >> 2;      // 0..7
    const int l_grp      = lane_in_sb & 3;       // 0..3
    const int iqs        = 2 * iqs_idx;
    const int l0         = 2 * l_grp;

    const flambeau_block_iq2_s* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;
    for (int sb = sb_in_iter; sb < n_superblocks_per_row; sb += SB_PER_ITER) {
        const flambeau_block_iq2_s* bk = xrow + sb;

        const float d_sb = (float) bk->d;
        const unsigned int qs_byte = bk->qs[2 * iqs + l_grp];
        const int qh = (int) bk->qh[iqs >> 1];
        const int grid_idx = (int) qs_byte | ((qh << (8 - l0)) & 0x300);

        const unsigned int signs_byte = bk->qs[32 + 2 * iqs + l_grp];
        unsigned int signs0_pack = 0, signs1_pack = 0;
        signs0_pack |= (signs_byte & 0x01) ? 0x000000FFu : 0;
        signs0_pack |= (signs_byte & 0x02) ? 0x0000FF00u : 0;
        signs0_pack |= (signs_byte & 0x04) ? 0x00FF0000u : 0;
        signs0_pack |= (signs_byte & 0x08) ? 0xFF000000u : 0;
        signs1_pack |= (signs_byte & 0x10) ? 0x000000FFu : 0;
        signs1_pack |= (signs_byte & 0x20) ? 0x0000FF00u : 0;
        signs1_pack |= (signs_byte & 0x40) ? 0x00FF0000u : 0;
        signs1_pack |= (signs_byte & 0x80) ? 0xFF000000u : 0;

        const uint64_t grid_u64 = IQ2S_GRID[grid_idx];
        const unsigned int grid0 = (unsigned int) grid_u64;
        const unsigned int grid1 = (unsigned int) (grid_u64 >> 32);
        const int grid_signed0 = iq2xxs_apply_signs(grid0, signs0_pack);
        const int grid_signed1 = iq2xxs_apply_signs(grid1, signs1_pack);

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + (iqs >> 1);
        const int u0 = load_int_b2(ya->qs, l0 + 0);
        const int u1 = load_int_b2(ya->qs, l0 + 1);
        const float d_y = (float) ya->d;

        int sumi = dp4a(grid_signed0, u0, 0);
        sumi     = dp4a(grid_signed1, u1, sumi);

        const int sub     = iqs >> 1;
        const int sc_byte = (int) bk->scales[sub >> 1];
        const int ls      = (sub & 1) ? (sc_byte >> 4) : (sc_byte & 0x0F);

        const float scale_factor = ((float) ls + 0.5f) * 0.25f;
        acc += d_sb * d_y * (float) sumi * scale_factor;
    }

    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}
