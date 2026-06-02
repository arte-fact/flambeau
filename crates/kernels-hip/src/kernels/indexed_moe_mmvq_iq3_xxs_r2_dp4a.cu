// indexed_moe_mmvq_iq3_xxs_r2_dp4a — IQ3_XXS MoE MMVQ with DP4A inner loop.
//
// Launch: blockDim = { 64 } (wave64), gridDim = { n_row_pairs,
// n_tokens * top_k, 1 }. 2 output rows per block; 32 lanes per row,
// each covering one (iqs, l_grp) position over (iqs_idx ∈ 0..7) ×
// (l_grp ∈ 0..3) = 32 codebook lookups per super-block.
//
// Per (iqs, l_grp): read 8 codebook bytes from qs[4*iqs..4*iqs+8) and
// the aux32 word from qs[64 + 2*iqs] (4-bit scale ls in bits [28..32),
// four 7-bit ksigns in low 28 bits). Two grid lookups (q3_0, q3_1) +
// popcount-parity sign expansion + two DP4As + reference rounding
// `(ls * sumi + sumi/2) / 2`.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "iq_grid.cuh"

static __device__ __forceinline__ int flambeau_indexed_moe_iq3_xxs_dp4a(
    int a, int b, int c
) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ int flambeau_indexed_moe_iq3_xxs_apply_signs(
    uint32_t grid, uint32_t signs
) {
    const int8_t g0 = (int8_t)(uint8_t)(grid & 0xFF);
    const int8_t g1 = (int8_t)(uint8_t)((grid >> 8) & 0xFF);
    const int8_t g2 = (int8_t)(uint8_t)((grid >> 16) & 0xFF);
    const int8_t g3 = (int8_t)(uint8_t)((grid >> 24) & 0xFF);
    const int8_t r0 = (signs & 0xFF) ? (int8_t)(-g0) : g0;
    const int8_t r1 = ((signs >> 8) & 0xFF) ? (int8_t)(-g1) : g1;
    const int8_t r2 = ((signs >> 16) & 0xFF) ? (int8_t)(-g2) : g2;
    const int8_t r3 = ((signs >> 24) & 0xFF) ? (int8_t)(-g3) : g3;
    return (int)((uint8_t)r0)
         | ((int)((uint8_t)r1) << 8)
         | ((int)((uint8_t)r2) << 16)
         | ((int)((uint8_t)r3) << 24);
}

static __device__ __forceinline__ uint32_t
flambeau_indexed_moe_iq3_xxs_unpack_ksigns(uint32_t v_7bit)
{
    const uint32_t v = v_7bit & 0x7F;
    const uint32_t p = (uint32_t) __builtin_popcount(v) & 1u;
    const uint32_t s = v ^ (p << 7);
    return s * 0x01010101u;
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_iq3_xxs_r2_dp4a_q8_1(
    const flambeau_block_iq3_xxs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    float* __restrict__ dst,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row
) {
    const int row_pair = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;

    if (token >= n_tokens) return;

    const int lane    = threadIdx.x;
    const int row_hi  = lane >> 5;
    const int lane_lo = lane & 31;

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const flambeau_block_iq3_xxs* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    const int iqs_idx = lane_lo >> 2;     // 0..7
    const int l_grp   = lane_lo & 3;      // 0..3
    const int iqs     = 2 * iqs_idx;
    const int l0      = 2 * l_grp;

    float acc = 0.0f;

    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        const flambeau_block_iq3_xxs* bk = xrow + sb;
        const float d_sb = (float) bk->d;

        const uint8_t* q3_base = bk->qs + 4 * iqs;
        const uint8_t q3_0 = q3_base[l0 + 0];
        const uint8_t q3_1 = q3_base[l0 + 1];

        const uint32_t aux32 =
            *((const uint32_t*) (bk->qs + 64 + 2 * iqs));

        const int ls = (int)(aux32 >> 28);

        const uint32_t signs_full =
            flambeau_indexed_moe_iq3_xxs_unpack_ksigns(aux32 >> (7 * l_grp));

        const uint32_t sel_lo = signs_full & 0x08040201u;
        const uint32_t sel_hi = signs_full & 0x80402010u;

        uint32_t signs0_pack = 0;
        signs0_pack |= ((sel_lo >>  0) & 0xFFu) ? 0x000000FFu : 0;
        signs0_pack |= ((sel_lo >>  8) & 0xFFu) ? 0x0000FF00u : 0;
        signs0_pack |= ((sel_lo >> 16) & 0xFFu) ? 0x00FF0000u : 0;
        signs0_pack |= ((sel_lo >> 24) & 0xFFu) ? 0xFF000000u : 0;

        uint32_t signs1_pack = 0;
        signs1_pack |= ((sel_hi >>  0) & 0xFFu) ? 0x000000FFu : 0;
        signs1_pack |= ((sel_hi >>  8) & 0xFFu) ? 0x0000FF00u : 0;
        signs1_pack |= ((sel_hi >> 16) & 0xFFu) ? 0x00FF0000u : 0;
        signs1_pack |= ((sel_hi >> 24) & 0xFFu) ? 0xFF000000u : 0;

        const uint32_t grid0 = IQ3XXS_GRID[q3_0];
        const uint32_t grid1 = IQ3XXS_GRID[q3_1];

        const int grid_signed0 =
            flambeau_indexed_moe_iq3_xxs_apply_signs(grid0, signs0_pack);
        const int grid_signed1 =
            flambeau_indexed_moe_iq3_xxs_apply_signs(grid1, signs1_pack);

        const flambeau_block_q8_1* ya = y_row + sb * 8 + (iqs >> 1);
        const int u0 = ((const int*) ya->qs)[l0 + 0];
        const int u1 = ((const int*) ya->qs)[l0 + 1];
        const float d_y = (float) ya->d;

        int sumi = flambeau_indexed_moe_iq3_xxs_dp4a(grid_signed0, u0, 0);
        sumi     = flambeau_indexed_moe_iq3_xxs_dp4a(grid_signed1, u1, sumi);

        sumi = (ls * sumi + sumi / 2) / 2;

        acc += d_sb * d_y * (float) sumi;
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
