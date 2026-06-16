#pragma once
// Templated single-row IQ3_S MMVQ (decode GEMV). IQ3_S (3.4375 bpw) uses 8-bit
// codebook indices in `qs` extended by a 9th bit from `qh`, explicit per-byte
// sign masks in `signs[32]`, and 4-bit sub-block scales. Ports llama.cpp's
// `vec_dot_iq3_s_q8_1`: 256 threads, 32 per super-block (4 lanes × 8
// sub-blocks); each lane gathers two `IQ3S_GRID` entries, applies the explicit
// sign bytes, dp4a's, and scales by (1 + 2·nibble). Reuses IQ2_XXS's identical
// `iq2xxs_apply_signs`; backend-neutral; only the Q8_1 int loads are misaligned
// (via `load_int_b2`), the weight reads are single bytes.
#include "mmvq_iq2_xxs.cuh"

template<class OutT>
static __device__ __forceinline__ void mmvq_iq3_s_dp4a_row(
    const flambeau_block_iq3_s* __restrict__ x,
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

    const flambeau_block_iq3_s* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;
    for (int sb = sb_in_iter; sb < n_superblocks_per_row; sb += SB_PER_ITER) {
        const flambeau_block_iq3_s* bk = xrow + sb;

        const float d_sb = (float) bk->d;
        const uint8_t* qs_base = bk->qs + 4 * iqs;
        const unsigned int qs0 = qs_base[l0 + 0];
        const unsigned int qs1 = qs_base[l0 + 1];

        const unsigned int qh = bk->qh[iqs >> 1];
        const int g_idx0 = (int) qs0 | (((int) qh << (8 - l0)) & 0x100);
        const int g_idx1 = (int) qs1 | (((int) qh << (7 - l0)) & 0x100);
        const unsigned int grid0 = IQ3S_GRID[g_idx0];
        const unsigned int grid1 = IQ3S_GRID[g_idx1];

        const unsigned int signs_byte = bk->signs[4 * (iqs >> 1) + l_grp];
        unsigned int signs0_pack = 0, signs1_pack = 0;
        signs0_pack |= (signs_byte & 0x01) ? 0x000000FFu : 0;
        signs0_pack |= (signs_byte & 0x02) ? 0x0000FF00u : 0;
        signs0_pack |= (signs_byte & 0x04) ? 0x00FF0000u : 0;
        signs0_pack |= (signs_byte & 0x08) ? 0xFF000000u : 0;
        signs1_pack |= (signs_byte & 0x10) ? 0x000000FFu : 0;
        signs1_pack |= (signs_byte & 0x20) ? 0x0000FF00u : 0;
        signs1_pack |= (signs_byte & 0x40) ? 0x00FF0000u : 0;
        signs1_pack |= (signs_byte & 0x80) ? 0xFF000000u : 0;

        const int grid_signed0 = iq2xxs_apply_signs(grid0, signs0_pack);
        const int grid_signed1 = iq2xxs_apply_signs(grid1, signs1_pack);

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + (iqs >> 1);
        const int u0 = load_int_b2(ya->qs, l0 + 0);
        const int u1 = load_int_b2(ya->qs, l0 + 1);
        const float d_y = (float) ya->d;

        int sumi = dp4a(grid_signed0, u0, 0);
        sumi     = dp4a(grid_signed1, u1, sumi);

        const int sc_byte   = bk->scales[iqs >> 2];
        const int sc_nibble = (sc_byte >> ((iqs << 1) & 0x04)) & 0x0F;
        const int scale     = 1 + 2 * sc_nibble;

        acc += d_sb * d_y * (float) (sumi * scale);
    }

    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}
