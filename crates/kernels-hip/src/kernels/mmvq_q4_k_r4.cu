// mmvq_q4_k_r4 — V2.31.e Q4_K MMVQ, 4 rows per block (wave64).
//
// Quarter-wave per row, 16 lanes each. Compared to `mmvq_q4_k_r2`:
//   r2: 64 threads, 2 rows, 32 lanes/row, 1 byte/lane/sub-block
//   r4: 64 threads, 4 rows, 16 lanes/row, 2 bytes/lane/sub-block
//
// Halves block count again (n_rows/4 vs n_rows/2). Targeted at
// Qwen3-Coder-30B decode where `mmvq_q4_k_r2` is 30 % of wall
// (V2.30.b profile) on the dense attention Q/K/V/O projections.
//
// Each row's 16 lanes cover the 32 bytes of each Q4_K sub-block by
// each handling 2 positions: byte_off + {0, 16}. Otherwise the maths
// is byte-identical to r2 — same scales, same nibble decode, same
// F32 accumulation envelope.

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_mmvq_q4_k_r4_q8_1(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row_quad = blockIdx.x;
    const int lane     = threadIdx.x;            // 0..63
    const int row_idx  = lane >> 4;              // 0..3 — which row in the quad
    const int lane_lo  = lane & 15;              // 0..15 — quarter-warp position

    const int row = row_quad * 4 + row_idx;
    if (row >= n_rows) return;                   // boundary

    const flambeau_block_q4_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q4_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint8_t sc = 0, m = 0;
            flambeau_q4k_scale_min(s, bk->scales, &sc, &m);

            const flambeau_block_q8_1* ya = y_sb + s;
            const float d_y = (float) ya->d;

            // Each lane handles 2 byte positions per sub-block: lane_lo
            // and lane_lo + 16. Covers the 32 bytes of the sub-block
            // across the 16 lanes of the quarter-warp.
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int lane_pos = lane_lo + half * 16;
                const int byte_idx = (s >> 1) * 32 + lane_pos;
                const int byte_v = (int) bk->qs[byte_idx];
                const int raw_q = (s & 1) ? (byte_v >> 4) : (byte_v & 0x0F);

                const float x_val = d * (float) sc * (float) raw_q - dmin * (float) m;

                const int qi = (int) ya->qs[lane_pos];
                const float y_val = d_y * (float) qi;

                acc += x_val * y_val;
            }
        }
    }

    acc = gfx906_quarter_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[row] = acc;
    }
}
