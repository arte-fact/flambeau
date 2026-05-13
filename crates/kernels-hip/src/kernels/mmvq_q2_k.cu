// Q2_K weight × Q8_1 activation → F32 dst. 64-thread wave64, one output row
// per block. Per-element on-the-fly decode mirrors dequant_q2_k.
//
// Q2_K super-block (84 B, 256 elements):
//   scales[16] — packed 4-bit (scale, min): lo nibble = scale, hi = min
//   qs[64]     — 2 bits per element, 4 elements / byte
//   d          — fp16 super-block scale
//   dmin       — fp16 super-block min scale
//
// Layout walk (matches dequant_q2_k chunks_exact(32) + 4 shifts × 2 groups):
//   chunk_idx  = y_idx >> 7        (0..1)        — selects qs[chunk*32..]
//   shift_iter = (y_idx & 127) >> 5 (0..3)       — shift = 2*shift_iter
//   scale_idx  = (y_idx >> 4) & 1                 — 0 = qs[0..16], 1 = qs[16..32]
//   l          = y_idx & 15
//   qi = l + 16*scale_idx
//   is = chunk_idx * 8 + 2 * shift_iter + scale_idx
//
// Reconstruction: y = d * (scales[is] & 0xF) * q - dmin * (scales[is] >> 4)
// where q = (qs[chunk*32 + qi] >> shift) & 3.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q2K_THREADS 64

extern "C" __global__ void flambeau_mmvq_q2_K_q8_1(
    const flambeau_block_q2_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane = threadIdx.x;

    const flambeau_block_q2_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q2_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int y_idx     = lane + k * 64;
            const int chunk_idx = y_idx >> 7;
            const int within_128 = y_idx & 127;
            const int shift_iter = within_128 >> 5;
            const int within_32  = within_128 & 31;
            const int scale_idx  = within_32 >> 4;
            const int l          = within_32 & 15;
            const int qi         = l + 16 * scale_idx;
            const int is         = chunk_idx * 8 + 2 * shift_iter + scale_idx;
            const int shift      = 2 * shift_iter;

            const int qs_byte = bk->qs[chunk_idx * 32 + qi];
            const int q       = (qs_byte >> shift) & 3;
            const int sc_byte = bk->scales[is];
            const int sc      = sc_byte & 0xF;
            const int mn      = sc_byte >> 4;

            const float x_val = d * (float) sc * (float) q - dmin * (float) mn;

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
