// mmvq_q4_k_r2 — Q4_K MMVQ, multi-row r2 DPP-reduce (candle P29 pattern).
//
// Compared to the single-row oracle in `mmvq_q4_k.cu`:
// - 64 threads per block STILL = one wave64, but the wavefront computes
//   two output rows simultaneously. Lanes 0..31 own row R+0; lanes 32..63
//   own row R+1. `blockIdx.x` indexes the row *pair*.
// - Each half-warp iterates all 8 sub-blocks for its row, reading one
//   element per sub-block (8 elements per lane per super-block). The
//   single-row kernel's "hi_half picks nibble" trick is gone — each lane
//   now visits every sub-block so it covers both nibbles.
// - Activation (`y_sb[s]`) is shared across the two rows → halved
//   per-launch cache traffic on the Q8_1 side vs two single-row launches.
// - Final reduction uses `gfx906_half_warp_reduce_sum` (32-lane DPP chain,
//   stops at xor-16, skipping the cross-half swap).
//
// This is the primary decode-path dtype for Qwen3.6 Q4_K_M.

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_mmvq_q4_k_r2_q8_1(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,                 // total output rows (may be odd — boundary handled)
    const int n_superblocks_per_row
) {
    const int row_pair = blockIdx.x;
    const int lane     = threadIdx.x;           // 0..63
    const int row_hi   = lane >> 5;             // 0 → row R+0, 1 → row R+1
    const int lane_lo  = lane & 31;             // 0..31 — position within sub-block

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;                  // boundary: odd-row count

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

            const int byte_idx = (s >> 1) * 32 + lane_lo;
            const int byte_v = (int) bk->qs[byte_idx];
            const int raw_q = (s & 1) ? (byte_v >> 4) : (byte_v & 0x0F);

            const float x_val = d * (float) sc * (float) raw_q - dmin * (float) m;

            const flambeau_block_q8_1* ya = y_sb + s;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[lane_lo];
            const float y_val = d_y * (float) qi;

            acc += x_val * y_val;
        }
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[row] = acc;
    }
}
