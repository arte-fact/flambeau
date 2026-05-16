// mmvq_iq3_xxs_r2 — IQ3_XXS MMVQ, multi-row r2 (DPP half-warp reduce).
// Compared to the single-row oracle in `mmvq_iq3_xxs.cu`:
// - 64 threads per block = one wave64 = one row pair. Lanes 0..31 own row
//   R+0; lanes 32..63 own row R+1. `blockIdx.x` indexes the row *pair*.
// - Each half-warp iterates ALL 8 sub-blocks (ib32) for its row, covering
//   one element per (ib32, lane_lo) pair. lane_lo = 8*l + j_in_8.
// - Activation (`y_sb[ib32]`) is shared across the two rows → halved
//   per-launch cache traffic on the Q8_1 side vs two single-row launches.

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "mmvq_store.cuh"
#include "iq_grid.cuh"

template<typename OutT>
__device__ void mmvq_iq3_xxs_r2_q8_1_body(
    const flambeau_block_iq3_xxs* __restrict__ x,
    const flambeau_block_q8_1*    __restrict__ y,
    OutT* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row_pair = blockIdx.x;
    const int lane     = threadIdx.x;
    const int row_hi   = lane >> 5;
    const int lane_lo  = lane & 31;
    const int l        = lane_lo >> 3;       // 0..3
    const int j_in_8   = lane_lo & 7;        // 0..7
    const int j_lane   = j_in_8 & 3;         // 0..3
    const int g_choice = (j_in_8 >> 2);      // 0 or 1

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const flambeau_block_iq3_xxs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_iq3_xxs* bk = xrow + b;
        const float d = (float) bk->d;
        const uint8_t* scs = bk->qs + QK_K / 4;
        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t* sp = scs + 4 * ib32;
            const uint32_t aux32 =
                  (uint32_t) sp[0]
                | ((uint32_t) sp[1] << 8)
                | ((uint32_t) sp[2] << 16)
                | ((uint32_t) sp[3] << 24);

            const float    db        = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            const uint8_t  sign_byte = KSIGNS_IQ2XS[(aux32 >> (7 * l)) & 0x7F];
            const int      grid_idx  = (int) bk->qs[ib32 * 8 + 2 * l + g_choice];
            const uint32_t g_u32     = IQ3XXS_GRID[grid_idx];
            const int      lane_val  = (int)((g_u32 >> (8 * j_lane)) & 0xFF);
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

    if (lane_lo == 0) {
        mmvq_store<OutT>(dst, row, acc);
    }
}

extern "C" __global__ void flambeau_mmvq_iq3_xxs_r2_q8_1(
    const flambeau_block_iq3_xxs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq3_xxs_r2_q8_1_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_iq3_xxs_r2_q8_1_f16(
    const flambeau_block_iq3_xxs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq3_xxs_r2_q8_1_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
