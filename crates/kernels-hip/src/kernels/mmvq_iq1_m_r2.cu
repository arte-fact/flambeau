// mmvq_iq1_m_r2 — IQ1_M MMVQ, multi-row r2 (DPP half-warp reduce).

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "iq_grid.cuh"

__device__ __forceinline__ float iq1m_reassemble_d_r2(const uint8_t* __restrict__ scales) {
    const int sc0 = (int) scales[0] | ((int) scales[1] << 8);
    const int sc1 = (int) scales[2] | ((int) scales[3] << 8);
    const int sc2 = (int) scales[4] | ((int) scales[5] << 8);
    const int sc3 = (int) scales[6] | ((int) scales[7] << 8);
    const int d_bits = (sc0 >> 12)
                     | ((sc1 >> 8) & 0x00F0)
                     | ((sc2 >> 4) & 0x0F00)
                     | (sc3 & 0xF000);
    fb_fp16_t d_fp16 = *reinterpret_cast<const fb_fp16_t*>(&d_bits);
    return (float) d_fp16;
}

extern "C" __global__ void flambeau_mmvq_iq1_m_r2_q8_1(
    const flambeau_block_iq1_m*  __restrict__ x,
    const flambeau_block_q8_1*   __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row_pair = blockIdx.x;
    const int lane     = threadIdx.x;
    const int row_hi   = lane >> 5;
    const int lane_lo  = lane & 31;
    const int l        = lane_lo >> 3;
    const int j_in_8   = lane_lo & 7;

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const flambeau_block_iq1_m* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_iq1_m* bk = xrow + b;
        const float d = iq1m_reassemble_d_r2(bk->scales);
        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const int sc_word = (int) bk->scales[2 * (ib32 >> 1)]
                              | ((int) bk->scales[2 * (ib32 >> 1) + 1] << 8);
            const int shift1 = 6 * (ib32 & 1);
            const int shift2 = shift1 + 3;
            const float dl1 = d * (2.0f * (float)((sc_word >> shift1) & 7) + 1.0f);
            const float dl2 = d * (2.0f * (float)((sc_word >> shift2) & 7) + 1.0f);

            const int qh_pick = (l < 2) ? (2 * ib32) : (2 * ib32 + 1);
            const uint8_t qh_byte = bk->qh[qh_pick];
            const int shift_idx  = 8 - 4 * (l & 1);
            const int idx = (int) bk->qs[4 * ib32 + l]
                          | (((int) qh_byte << shift_idx) & 0x700);
            const int delta_bit = (l & 1) == 0 ? 0x08 : 0x80;
            const float delta = (qh_byte & delta_bit) ? -IQ1_DELTA : IQ1_DELTA;
            const float dl = (l < 2) ? dl1 : dl2;

            const uint64_t g_u64 = IQ1S_GRID[idx];
            const int8_t g_i8 = (int8_t)((g_u64 >> (8 * j_in_8)) & 0xFF);
            const float x_val = dl * ((float) g_i8 + delta);

            const flambeau_block_q8_1* ya = y_sb + ib32;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[8 * l + j_in_8];
            const float y_val = d_y * (float) qi;

            acc += x_val * y_val;
        }
    }

    acc = gfx906_half_warp_reduce_sum(acc);
    if (lane_lo == 0) dst[row] = acc;
}
