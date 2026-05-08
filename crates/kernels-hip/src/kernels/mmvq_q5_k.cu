// mmvq_q5_k — Q5_K weight × Q8_1 activation → F32 dst.
// Same 64-thread / single-row layout as `mmvq_q4_k`. Adds the 5th bit from
// `qh` (one bit per element, 32 bytes per super-block). The bit-mask walks
// `u1 = 1 << (2*grp)` / `u2 = 2 << (2*grp)` across the 4 groups, so each
// lane's qh read is `qh[byte_off] & (hi_half ? u2 : u1)`.

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_mmvq_q5_k_q8_1(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane     = threadIdx.x;           // 0..63
    const int byte_off = lane & 31;             // 0..31
    const int hi_half  = lane >> 5;             // 0 low-nibble, 1 high-nibble

    const flambeau_block_q5_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q5_K* bk = xrow + b;

        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const int sub = 2 * grp + hi_half;

            uint8_t sc = 0, m = 0;
            flambeau_q4k_scale_min(sub, bk->scales, &sc, &m);

            const int byte_v = (int) bk->qs[grp * 32 + byte_off];
            const int raw_q4 = hi_half ? (byte_v >> 4) : (byte_v & 0x0F);

            // High bit lives in qh. For group `grp` low-nibble the mask is
            // `1 << (2*grp)`, for high-nibble `2 << (2*grp)`.
            const uint8_t qh_byte = bk->qh[byte_off];
            const int mask = (hi_half ? 2 : 1) << (2 * grp);
            const int raw_q = raw_q4 + ((qh_byte & mask) ? 16 : 0);

            const float x_val = d * (float) sc * (float) raw_q - dmin * (float) m;

            const flambeau_block_q8_1* ya = y_sb + sub;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[byte_off];
            const float y_val = d_y * (float) qi;

            acc += x_val * y_val;
        }
    }

    acc = gfx906_warp_reduce_sum(acc);

    if (lane == 0) {
        dst[row] = acc;
    }
}
