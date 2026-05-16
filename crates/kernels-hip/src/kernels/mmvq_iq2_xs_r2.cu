// mmvq_iq2_xs_r2 — IQ2_XS MMVQ, multi-row r2 (DPP half-warp reduce).

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "mmvq_store.cuh"
#include "iq_grid.cuh"

template<typename OutT>
__device__ void mmvq_iq2_xs_r2_q8_1_body(
    const flambeau_block_iq2_xs* __restrict__ x,
    const flambeau_block_q8_1*   __restrict__ y,
    OutT* __restrict__ dst,
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

    const flambeau_block_iq2_xs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_iq2_xs* bk = xrow + b;
        const float d = (float) bk->d;
        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ++ib32) {
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
            const float y_val = d_y * (float) qi;

            acc += x_val * y_val;
        }
    }

    acc = gfx906_half_warp_reduce_sum(acc);
    if (lane_lo == 0) mmvq_store<OutT>(dst, row, acc);
}

extern "C" __global__ void flambeau_mmvq_iq2_xs_r2_q8_1(
    const flambeau_block_iq2_xs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq2_xs_r2_q8_1_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_iq2_xs_r2_q8_1_f16(
    const flambeau_block_iq2_xs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq2_xs_r2_q8_1_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
