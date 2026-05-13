// mmvq_q2_k_r2 — Q2_K MMVQ multi-row r2 DPP-reduce.
// 64 threads / block (wave64), wavefront computes two output rows. Lanes
// 0..31 own row R+0; lanes 32..63 own row R+1. Activation shared across
// rows. Final reduce uses half-warp DPP (32 lanes, stops at xor-16).
//
// Per iteration `s` ∈ 0..8 (one Q8_1 sub-block of 32 elements at a time):
//   chunk_idx  = s >> 2
//   shift_iter = s & 3       (shift = 2*shift_iter)
//   scale_idx  = lane_lo >> 4
//   l          = lane_lo & 15
//   qi         = l + 16*scale_idx
//   is         = chunk_idx*8 + 2*shift_iter + scale_idx
//   qs byte    = qs[chunk_idx*32 + qi]
// Reconstruct: y = d*(scales[is]&0xF)*q - dmin*(scales[is]>>4).

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_mmvq_q2_K_r2_q8_1(
    const flambeau_block_q2_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row_pair = blockIdx.x;
    const int lane     = threadIdx.x;
    const int row_hi   = lane >> 5;
    const int lane_lo  = lane & 31;

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const flambeau_block_q2_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q2_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        const int scale_idx = lane_lo >> 4;
        const int l         = lane_lo & 15;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int chunk_idx  = s >> 2;
            const int shift_iter = s & 3;
            const int qi         = l + 16 * scale_idx;
            const int is         = chunk_idx * 8 + 2 * shift_iter + scale_idx;
            const int shift      = 2 * shift_iter;

            const int qs_byte = bk->qs[chunk_idx * 32 + qi];
            const int q       = (qs_byte >> shift) & 3;
            const int sc_byte = bk->scales[is];
            const int sc      = sc_byte & 0xF;
            const int mn      = sc_byte >> 4;

            const float x_val = d * (float) sc * (float) q - dmin * (float) mn;

            const flambeau_block_q8_1* ya = y_sb + s;
            const float d_y = (float) ya->d;
            const int   qi8 = (int) ya->qs[lane_lo];
            acc += x_val * d_y * (float) qi8;
        }
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[row] = acc;
    }
}
