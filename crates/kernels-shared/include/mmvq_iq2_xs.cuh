#pragma once
// Templated single-row IQ2_XS MMVQ (decode GEMV). IQ2_XS (2.3125 bpw) stores
// per element a uint16: low 9 bits index `IQ2XS_GRID` (512 entries, uint64), top
// 7 bits a sign-LUT index; plus 4-bit sub-block scales. Ports llama.cpp's
// `vec_dot_iq2_xs_q8_1`: 256 threads, 32 per super-block. Reuses IQ2_XXS's
// identical sign helpers; the uint16 `qs` reads are naturally 2-byte aligned, so
// only the Q8_1 int loads go through `load_int_b2`.
#include "mmvq_iq2_xxs.cuh"

template<class OutT>
static __device__ __forceinline__ void mmvq_iq2_xs_dp4a_row(
    const flambeau_block_iq2_xs* __restrict__ x,
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

    const flambeau_block_iq2_xs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;
    for (int sb = sb_in_iter; sb < n_superblocks_per_row; sb += SB_PER_ITER) {
        const flambeau_block_iq2_xs* bk = xrow + sb;

        const float d_sb = (float) bk->d;
        const unsigned int q2_entry = (unsigned int) ((const uint16_t*) bk->qs)[2 * iqs + l_grp];

        const int grid_idx = (int) (q2_entry & 0x1FFu);
        const uint64_t grid_u64 = IQ2XS_GRID[grid_idx];
        const unsigned int grid0 = (unsigned int) grid_u64;
        const unsigned int grid1 = (unsigned int) (grid_u64 >> 32);

        const unsigned int signs_full = iq2xxs_unpack_ksigns(q2_entry >> 9);
        const unsigned int sel_lo = signs_full & 0x08040201u;
        const unsigned int sel_hi = signs_full & 0x80402010u;
        unsigned int signs0_pack = 0, signs1_pack = 0;
        signs0_pack |= ((sel_lo >> 0) & 0xFFu) ? 0x000000FFu : 0;
        signs0_pack |= ((sel_lo >> 8) & 0xFFu) ? 0x0000FF00u : 0;
        signs0_pack |= ((sel_lo >> 16) & 0xFFu) ? 0x00FF0000u : 0;
        signs0_pack |= ((sel_lo >> 24) & 0xFFu) ? 0xFF000000u : 0;
        signs1_pack |= ((sel_hi >> 0) & 0xFFu) ? 0x000000FFu : 0;
        signs1_pack |= ((sel_hi >> 8) & 0xFFu) ? 0x0000FF00u : 0;
        signs1_pack |= ((sel_hi >> 16) & 0xFFu) ? 0x00FF0000u : 0;
        signs1_pack |= ((sel_hi >> 24) & 0xFFu) ? 0xFF000000u : 0;

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
