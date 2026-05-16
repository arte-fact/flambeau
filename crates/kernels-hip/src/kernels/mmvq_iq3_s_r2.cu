// mmvq_iq3_s_r2 — IQ3_S MMVQ, multi-row r2 (DPP half-warp reduce).
// Same structure as `mmvq_iq3_xxs_r2.cu`: 64 threads = one wave64 = one row
// pair (lanes 0..31 own row R+0; lanes 32..63 own row R+1). Half-warp DPP
// reduce; odd-row boundary guarded.
//
// Body is the IQ3_S element-compute (9-bit codebook + per-byte signs) from
// mmvq_iq3_s.cu, lifted into the r2 loop pattern.

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "mmvq_store.cuh"
#include "iq_grid.cuh"

template<typename OutT>
__device__ void mmvq_iq3_s_r2_q8_1_body(
    const flambeau_block_iq3_s*  __restrict__ x,
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
    const int j_lane   = j_in_8 & 3;
    const int g_choice = (j_in_8 >> 2);

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const flambeau_block_iq3_s* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_iq3_s* bk = xrow + b;
        const float d = (float) bk->d;
        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t sc_byte = bk->scales[ib32 >> 1];
            const int sc_nibble = (ib32 & 1) ? (sc_byte >> 4) : (sc_byte & 0x0F);
            const float db = d * (1.0f + 2.0f * (float) sc_nibble);

            const uint8_t qh_byte = bk->qh[ib32];
            const int qh_shift = g_choice ? (7 - 2 * l) : (8 - 2 * l);
            const int g_idx = (int) bk->qs[ib32 * 8 + 2 * l + g_choice]
                            | (((int) qh_byte << qh_shift) & 256);
            const uint32_t g_u32 = IQ3S_GRID[g_idx];
            const int lane_val = (int)((g_u32 >> (8 * j_lane)) & 0xFF);

            const uint8_t signs = bk->signs[ib32 * 4 + l];
            const float sign = (signs & KMASK_IQ2XS[j_in_8]) ? -1.0f : 1.0f;
            const float x_val = db * (float) lane_val * sign;

            const flambeau_block_q8_1* ya = y_sb + ib32;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[8 * l + j_in_8];
            const float y_val = d_y * (float) qi;

            acc += x_val * y_val;
        }
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        mmvq_store<OutT>(dst, row, acc);
    }
}

extern "C" __global__ void flambeau_mmvq_iq3_s_r2_q8_1(
    const flambeau_block_iq3_s* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq3_s_r2_q8_1_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_iq3_s_r2_q8_1_f16(
    const flambeau_block_iq3_s* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq3_s_r2_q8_1_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
