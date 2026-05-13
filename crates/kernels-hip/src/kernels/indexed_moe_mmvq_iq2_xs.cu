// indexed_moe_mmvq_iq2_xs — IQ2_XS MMVQ with per-token expert routing.
// Phase 4 Slice B.

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "iq_grid.cuh"

extern "C" __global__ void flambeau_indexed_moe_mmvq_iq2_xs_q8_1(
    const flambeau_block_iq2_xs* __restrict__ x,
    const flambeau_block_q8_1*   __restrict__ y,
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

    const flambeau_block_iq2_xs* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    float acc = 0.0f;

    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_iq2_xs* bk = xrow + b;
        const float d = (float) bk->d;
        const flambeau_block_q8_1* y_sb = y_row + (size_t) b * 8;

        #pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const int ib32 = 2 * grp + sub_hi;
            const uint8_t sc_byte = bk->scales[ib32 >> 1];
            const int sc_nibble = (ib32 & 1) ? (sc_byte >> 4) : (sc_byte & 0x0F);
            const float db = d * (0.5f + (float) sc_nibble) * 0.25f;

            const uint8_t* qp = bk->qs + 8 * ib32 + 2 * l;
            const int q_u16 = (int) qp[0] | ((int) qp[1] << 8);
            const uint64_t g_u64 = IQ2XS_GRID[q_u16 & 0x1FF];
            const int lane_val = (int)((g_u64 >> (8 * j_in_8)) & 0xFF);
            const uint8_t sign_byte = KSIGNS_IQ2XS[(q_u16 >> 9) & 0x7F];
            const float sign = (sign_byte & (1u << j_in_8)) ? -1.0f : 1.0f;
            const float x_val = db * (float) lane_val * sign;

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
