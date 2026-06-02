// indexed_moe_mmvq_iq2_s_r2_dp4a — IQ2_S MoE MMVQ with DP4A inner loop.
// Drop-in replacement for `indexed_moe_mmvq_iq2_s.cu` (scalar FP32).
// Same per-element math as dense `mmvq_iq2_s_dp4a.cu` plus the standard
// MoE wiring. Note flambeau scale convention: `scales[ib32 >> 1]`
// (4 bytes) per super-block, NOT llama.cpp's `scales[iqs/2]` (8 bytes)
// — see feedback_iq2_s_scale_convention.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "iq_grid.cuh"

static __device__ __forceinline__ int flambeau_indexed_moe_iq2_s_dp4a(
    int a, int b, int c
) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ int flambeau_indexed_moe_iq2_s_apply_signs(
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

extern "C" __global__ void flambeau_indexed_moe_mmvq_iq2_s_r2_dp4a_q8_1(
    const flambeau_block_iq2_s* __restrict__ x,
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

    const flambeau_block_iq2_s* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    const int iqs_idx = lane_lo >> 2;     // 0..7
    const int l_grp   = lane_lo & 3;      // 0..3
    const int iqs     = 2 * iqs_idx;
    const int l0      = 2 * l_grp;

    float acc = 0.0f;

    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        const flambeau_block_iq2_s* bk = xrow + sb;
        const float d_sb = (float) bk->d;

        const uint8_t* qs_pair_base = bk->qs + 2 * iqs;
        const uint8_t qs_byte = qs_pair_base[l_grp];

        const int qh = (int) bk->qh[iqs >> 1];
        const int grid_idx = (int) qs_byte | ((qh << (8 - l0)) & 0x300);

        const uint8_t signs_byte = bk->qs[32 + 2 * iqs + l_grp];

        const int sub = iqs >> 1;
        const int sc_byte = (int) bk->scales[sub >> 1];
        const int ls = (sub & 1) ? (sc_byte >> 4) : (sc_byte & 0x0F);

        const uint64_t grid_u64 = IQ2S_GRID[grid_idx];
        const uint32_t grid0 = (uint32_t) grid_u64;
        const uint32_t grid1 = (uint32_t)(grid_u64 >> 32);

        uint32_t signs0_pack = 0;
        signs0_pack |= (signs_byte & 0x01) ? 0x000000FFu : 0;
        signs0_pack |= (signs_byte & 0x02) ? 0x0000FF00u : 0;
        signs0_pack |= (signs_byte & 0x04) ? 0x00FF0000u : 0;
        signs0_pack |= (signs_byte & 0x08) ? 0xFF000000u : 0;

        uint32_t signs1_pack = 0;
        signs1_pack |= (signs_byte & 0x10) ? 0x000000FFu : 0;
        signs1_pack |= (signs_byte & 0x20) ? 0x0000FF00u : 0;
        signs1_pack |= (signs_byte & 0x40) ? 0x00FF0000u : 0;
        signs1_pack |= (signs_byte & 0x80) ? 0xFF000000u : 0;

        const int grid_signed0 =
            flambeau_indexed_moe_iq2_s_apply_signs(grid0, signs0_pack);
        const int grid_signed1 =
            flambeau_indexed_moe_iq2_s_apply_signs(grid1, signs1_pack);

        const flambeau_block_q8_1* ya = y_row + sb * 8 + (iqs >> 1);
        const int u0 = ((const int*) ya->qs)[l0 + 0];
        const int u1 = ((const int*) ya->qs)[l0 + 1];
        const float d_y = (float) ya->d;

        int sumi = flambeau_indexed_moe_iq2_s_dp4a(grid_signed0, u0, 0);
        sumi     = flambeau_indexed_moe_iq2_s_dp4a(grid_signed1, u1, sumi);

        const float scale_factor = ((float) ls + 0.5f) * 0.25f;
        acc += d_sb * d_y * (float) sumi * scale_factor;
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
