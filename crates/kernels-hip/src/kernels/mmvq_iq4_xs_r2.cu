// mmvq_iq4_xs_r2 — IQ4_XS MMVQ, multi-row r2 (DPP half-warp reduce).
// Compared to the single-row oracle in `mmvq_iq4_xs.cu`:
// - 64 threads per block STILL = one wave64, but the wavefront computes
//   two output rows simultaneously. Lanes 0..31 own row R+0; lanes 32..63
//   own row R+1. `blockIdx.x` indexes the row *pair*.
// - Each half-warp iterates ALL 8 sub-blocks for its row, reading one
//   element per sub-block (8 elements per lane per super-block) so the
//   single-row "sub_hi pairs lanes across two sub-blocks" trick is gone.
// - Activation (`y_sb[s]`) is shared across the two rows → halved
//   per-launch cache traffic on the Q8_1 side vs two single-row launches.
// - Final reduction uses `gfx906_half_warp_reduce_sum` (32-lane DPP).

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "mmvq_store.cuh"

template<typename OutT>
__device__ void mmvq_iq4_xs_r2_q8_1_body(
    const flambeau_block_iq4_xs* __restrict__ x,
    const flambeau_block_q8_1*   __restrict__ y,
    OutT* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row_pair = blockIdx.x;
    const int lane     = threadIdx.x;         // 0..63
    const int row_hi   = lane >> 5;           // 0 → row R+0, 1 → row R+1
    const int lane_lo  = lane & 31;           // 0..31
    const int byte_off  = lane_lo & 15;       // 0..15
    const int nibble_hi = lane_lo >> 4;       // 0 (low) or 1 (high)
    const int elem_in_sub = byte_off + nibble_hi * 16;

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const flambeau_block_iq4_xs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_iq4_xs* bk = xrow + b;
        const float    d         = (float) bk->d;
        const uint16_t scales_h  = bk->scales_h;
        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int ls = flambeau_iq4_xs_scale(s, scales_h, bk->scales_l);

            const int byte_v = (int) bk->qs[s * 16 + byte_off];
            const int code   = (byte_v >> (nibble_hi * 4)) & 0x0F;
            const float x_val = d * (float) ls * (float) flambeau_iq4nl_lut(code);

            const flambeau_block_q8_1* ya = y_sb + s;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[elem_in_sub];
            const float y_val = d_y * (float) qi;

            acc += x_val * y_val;
        }
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        mmvq_store<OutT>(dst, row, acc);
    }
}

extern "C" __global__ void flambeau_mmvq_iq4_xs_r2_q8_1(
    const flambeau_block_iq4_xs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq4_xs_r2_q8_1_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_iq4_xs_r2_q8_1_f16(
    const flambeau_block_iq4_xs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq4_xs_r2_q8_1_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
