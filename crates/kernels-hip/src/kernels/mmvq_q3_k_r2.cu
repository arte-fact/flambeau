// mmvq_q3_k_r2 — Q3_K MMVQ multi-row r2 DPP-reduce.
// 64 threads / block (wave64), but the wavefront computes two output rows:
//   lanes 0..31 own row R+0; lanes 32..63 own row R+1.
// blockIdx.x indexes the row *pair*; activation is shared across the two
// rows (one Q8_1 fetch covers both). Final reduce uses the half-warp DPP
// chain (32 lanes, stops at xor-16).
//
// Each lane visits one Q8_1 sub-block at a time (8 sub-blocks / super-block,
// 32 elements / sub-block). Inside the sub-block lane_lo (0..31) picks the
// element. Decode follows mmvq_q3_k.cu with mappings:
//   blk128       = s >> 2
//   shift_iter   = s & 3
//   scale_idx    = lane_lo >> 4
//   l            = lane_lo & 15
//   qi           = l + 16 * scale_idx
//   is           = 8*blk128 + 2*shift_iter + scale_idx
//   hmask_bit_pos = shift_iter + 4*blk128
//
// Scale unpack uses the byte-wise u32 load (Q3_K block size 110 B leaves
// scales[] misaligned for every other super-block — same idiom as mmvq_q3_k).

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_mmvq_q3_k_r2_q8_1(
    const flambeau_block_q3_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row_pair = blockIdx.x;
    const int lane     = threadIdx.x;            // 0..63
    const int row_hi   = lane >> 5;              // 0 or 1
    const int lane_lo  = lane & 31;              // 0..31

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const flambeau_block_q3_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q3_K* bk = xrow + b;
        const float d_all = (float) bk->d;
        int8_t scales[16];
        flambeau_q3k_unpack_scales(bk->scales, scales);

        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        const int scale_idx = lane_lo >> 4;
        const int l         = lane_lo & 15;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int blk128        = s >> 2;
            const int shift_iter    = s & 3;
            const int qi            = l + 16 * scale_idx;
            const int is            = 8 * blk128 + 2 * shift_iter + scale_idx;
            const int shift         = 2 * shift_iter;
            const int hmask_bit_pos = shift_iter + 4 * blk128;

            const int hmask_bit = (bk->hmask[qi] >> hmask_bit_pos) & 1;
            const int sub       = (hmask_bit == 0) ? 4 : 0;
            const int qs_byte   = bk->qs[blk128 * 32 + qi];
            const int q_val     = ((qs_byte >> shift) & 3) - sub;

            const float x_val = d_all * (float) (scales[is] - 32) * (float) q_val;

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
