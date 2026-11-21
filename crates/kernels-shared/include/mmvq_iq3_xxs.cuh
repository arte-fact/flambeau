#pragma once
// Templated single-row IQ3_XXS MMVQ (decode GEMV). IQ3_XXS (3.0625 bpw) packs,
// per 8-byte sub-block: 8 codebook-index bytes into the u32 `IQ3XXS_GRID` + an
// aux32 of four 7-bit sign-LUT indices and a 4-bit scale. Ports llama.cpp's
// `vec_dot_iq3_xxs_q8_1`: 256 threads, 32 per super-block (4 lanes × 8
// sub-blocks); each lane gathers two grid entries, applies the sign LUT, and
// dp4a's against Q8_1, with the reference `(ls*sumi + sumi/2)/2` rounding.
// Reuses IQ2_XXS's identical sign helpers; backend-neutral via the arch +
// shared headers; misaligned int loads go through `load_int_b2`.
#include "mmvq_iq2_xxs.cuh"

template<class OutT>
static __device__ __forceinline__ void mmvq_iq3_xxs_dp4a_row(
    const flambeau_block_iq3_xxs* __restrict__ x,
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

    const flambeau_block_iq3_xxs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;
    for (int sb = sb_in_iter; sb < n_superblocks_per_row; sb += SB_PER_ITER) {
        const flambeau_block_iq3_xxs* bk = xrow + sb;

        const float d_sb = (float) bk->d;
        const uint8_t* q3_base = bk->qs + 4 * iqs;
        const unsigned int q3_0 = q3_base[l0 + 0];
        const unsigned int q3_1 = q3_base[l0 + 1];
        const unsigned int aux32 =
            (unsigned int) load_int_b2((const int8_t*) (bk->qs + 64 + 2 * iqs), 0);

        const int ls = (int) (aux32 >> 28);
        const unsigned int signs_full = iq2xxs_unpack_ksigns(aux32 >> (7 * l_grp));

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

        const unsigned int grid0 = IQ3XXS_GRID[q3_0];
        const unsigned int grid1 = IQ3XXS_GRID[q3_1];
        const int grid_signed0 = iq2xxs_apply_signs(grid0, signs0_pack);
        const int grid_signed1 = iq2xxs_apply_signs(grid1, signs1_pack);

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + (iqs >> 1);
        const int u0 = load_int_b2(ya->qs, l0 + 0);
        const int u1 = load_int_b2(ya->qs, l0 + 1);
        const float d_y = (float) ya->d;

        int sumi = dp4a(grid_signed0, u0, 0);
        sumi     = dp4a(grid_signed1, u1, sumi);
        sumi     = (ls * sumi + sumi / 2) / 2;

        acc += d_sb * d_y * (float) sumi;
    }

    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}
