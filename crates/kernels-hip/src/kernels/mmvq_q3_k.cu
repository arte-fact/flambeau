// Q3_K weight × Q8_1 activation → F32 dst. 64-thread wave64, one output
// row per block. cert-grade single-row reference; per-element on-the-fly
// decode mirrors crates/quant/src/dequant.rs::dequant_q3_k.
//
// Q3_K super-block layout (110 B, 256 elements):
//   hmask[32]   — high bit (bit 2) for each of 256 3-bit values, packed
//                 8 elements / byte. Bit `shift_iter + 4*blk128` of
//                 hmask[qi] (qi = l + 16*scale_idx ∈ 0..31).
//   qs[64]      — low 2 bits, 4 elements / byte. qs[blk128*32 + qi] >>
//                 (2*shift_iter) & 3 picks the low bits.
//   scales[12]  — 16 signed 6-bit scales (packed). Decoded via the
//                 KMASK1/KMASK2 unpack from dequant_q3_k.
//   d           — fp16 super-block scale.
//
// Reconstruction:
//   sub_bit = (hmask[qi] >> (shift_iter + 4*blk128)) & 1
//   q_val   = ((qs[blk128*32+qi] >> (2*shift_iter)) & 3) - (sub_bit == 0 ? 4 : 0)
//   y       = d · (scales[is] - 32) · q_val
//   is      = 8*blk128 + 2*shift_iter + scale_idx
//
// Each lane (0..63) handles 4 elements / super-block at positions
// {lane, lane+64, lane+128, lane+192}.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q3K_THREADS 64

extern "C" __global__ void flambeau_mmvq_q3_k_q8_1(
    const flambeau_block_q3_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane = threadIdx.x;

    const flambeau_block_q3_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q3_K* bk = xrow + b;
        const float d_all = (float) bk->d;
        int8_t scales[16];
        flambeau_q3k_unpack_scales(bk->scales, scales);

        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int y_idx = lane + k * 64;
            const int blk128 = y_idx >> 7;
            const int within_128 = y_idx & 127;
            const int shift_iter = within_128 >> 5;
            const int within_32 = within_128 & 31;
            const int scale_idx = within_32 >> 4;
            const int l = within_32 & 15;
            const int qi = l + 16 * scale_idx;
            const int is = 8 * blk128 + 2 * shift_iter + scale_idx;

            const int shift = 2 * shift_iter;
            const int hmask_bit_pos = shift_iter + 4 * blk128;
            const int hmask_bit = (bk->hmask[qi] >> hmask_bit_pos) & 1;
            const int sub = (hmask_bit == 0) ? 4 : 0;
            const int qs_byte = bk->qs[blk128 * 32 + qi];
            const int q_val = ((qs_byte >> shift) & 3) - sub;

            const float x_val = d_all * (float) (scales[is] - 32) * (float) q_val;

            const int y_block = y_idx >> 5;
            const int y_off   = y_idx & 31;
            const flambeau_block_q8_1* ya = y_sb + y_block;
            const float d_y = (float) ya->d;
            const int   qi8 = (int) ya->qs[y_off];
            acc += x_val * d_y * (float) qi8;
        }
    }

    acc = gfx906_warp_reduce_sum(acc);

    if (lane == 0) {
        dst[row] = acc;
    }
}
