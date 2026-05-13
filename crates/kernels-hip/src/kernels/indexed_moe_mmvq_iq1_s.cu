// indexed_moe_mmvq_iq1_s — IQ1_S MMVQ with per-token expert routing.

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "iq_grid.cuh"

extern "C" __global__ void flambeau_indexed_moe_mmvq_iq1_s_q8_1(
    const flambeau_block_iq1_s* __restrict__ x,
    const flambeau_block_q8_1*  __restrict__ y,
    const int* __restrict__ expert_ids,
    float* __restrict__ dst,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row
) {
    const int row       = blockIdx.x;
    const int slot      = blockIdx.y;
    const int token     = slot / top_k;
    const int slot_idx  = slot - token * top_k;
    if (row >= n_rows || token >= n_tokens) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const int lane    = threadIdx.x;
    const int sub_hi  = lane >> 5;
    const int lane_lo = lane & 31;
    const int l       = lane_lo >> 3;
    const int j_in_8  = lane_lo & 7;

    const flambeau_block_iq1_s* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    float acc = 0.0f;

    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_iq1_s* bk = xrow + b;
        const float d = (float) bk->d;
        const flambeau_block_q8_1* y_sb = y_row + (size_t) b * 8;

        #pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const int ib32 = 2 * grp + sub_hi;
            const int qh_u16 = (int) bk->qh[2 * ib32]
                             | ((int) bk->qh[2 * ib32 + 1] << 8);
            const float dl    = d * (2.0f * (float)((qh_u16 >> 12) & 7) + 1.0f);
            const float delta = (qh_u16 & 0x8000) ? -IQ1_DELTA : IQ1_DELTA;

            const int idx_lo = (int) bk->qs[4 * ib32 + l];
            const int high3  = (qh_u16 >> (3 * l)) & 7;
            const int idx    = idx_lo | (high3 << 8);
            const uint64_t g_u64 = IQ1S_GRID[idx];
            const int8_t g_i8 = (int8_t)((g_u64 >> (8 * j_in_8)) & 0xFF);
            const float x_val = dl * ((float) g_i8 + delta);

            const flambeau_block_q8_1* ya = y_sb + ib32;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[8 * l + j_in_8];
            acc += x_val * d_y * (float) qi;
        }
    }

    acc = gfx906_warp_reduce_sum(acc);
    if (lane == 0) {
        dst[(size_t)(token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
