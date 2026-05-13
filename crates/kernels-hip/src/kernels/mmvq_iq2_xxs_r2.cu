// mmvq_iq2_xxs_r2 — IQ2_XXS MMVQ, multi-row r2 (DPP half-warp reduce).
// Same body as mmvq_iq2_xxs.cu lifted into the r2 loop pattern.

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "iq_grid.cuh"

extern "C" __global__ void flambeau_mmvq_iq2_xxs_r2_q8_1(
    const flambeau_block_iq2_xxs* __restrict__ x,
    const flambeau_block_q8_1*    __restrict__ y,
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

    const flambeau_block_iq2_xxs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_iq2_xxs* bk = xrow + b;
        const float d = (float) bk->d;
        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t* sp = bk->qs + 8 * ib32;
            const uint32_t aux0 =
                  (uint32_t) sp[0] | ((uint32_t) sp[1] << 8)
                | ((uint32_t) sp[2] << 16) | ((uint32_t) sp[3] << 24);
            const uint32_t aux1 =
                  (uint32_t) sp[4] | ((uint32_t) sp[5] << 8)
                | ((uint32_t) sp[6] << 16) | ((uint32_t) sp[7] << 24);

            const float    db        = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            const int      idx       = (int)((aux0 >> (8 * l)) & 0xFF);
            const uint64_t g_u64     = IQ2XXS_GRID[idx];
            const int      lane_val  = (int)((g_u64 >> (8 * j_in_8)) & 0xFF);
            const uint8_t  sign_byte = KSIGNS_IQ2XS[(aux1 >> (7 * l)) & 0x7F];
            const float    sign      = (sign_byte & (1u << j_in_8)) ? -1.0f : 1.0f;
            const float    x_val     = db * (float) lane_val * sign;

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
