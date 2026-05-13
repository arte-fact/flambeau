// mmvq_iq4_nl_r2 — IQ4_NL MMVQ, multi-row r2 (DPP half-warp reduce).
// Compared to the single-row oracle in `mmvq_iq4_nl.cu`:
// - 64 threads per block STILL = one wave64, but the wavefront computes
//   two output rows simultaneously. Lanes 0..31 own row R+0; lanes 32..63
//   own row R+1. `blockIdx.x` indexes the row *pair*.
// - Each half-warp iterates EVERY block for its row, covering all 32
//   elements per block (byte_off × nibble_hi). The single-row kernel's
//   "block_hi splits a wave across 2 blocks" trick is gone — each row
//   needs the full per-block sum, not a half.
// - Activation (`y[b]`) is shared across the two rows → halved per-launch
//   cache traffic on the Q8_1 side vs two single-row launches.
// - Final reduction uses `gfx906_half_warp_reduce_sum` (32-lane DPP).

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_mmvq_iq4_nl_r2_q8_1(
    const flambeau_block_iq4_nl* __restrict__ x,
    const flambeau_block_q8_1*   __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int row_pair = blockIdx.x;
    const int lane     = threadIdx.x;         // 0..63
    const int row_hi   = lane >> 5;           // 0 → row R+0, 1 → row R+1
    const int lane_lo  = lane & 31;           // 0..31
    const int byte_off  = lane_lo & 15;       // 0..15
    const int nibble_hi = lane_lo >> 4;       // 0 (low) or 1 (high)
    const int elem_in_block = byte_off + nibble_hi * 16;

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const flambeau_block_iq4_nl* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_blocks_per_row; ++b) {
        const flambeau_block_iq4_nl* bk = xrow + b;
        const flambeau_block_q8_1*   by = y + b;

        const float d_x    = (float) bk->d;
        const int   byte_v = (int) bk->qs[byte_off];
        const int   code   = (byte_v >> (nibble_hi * 4)) & 0x0F;
        const float x_val  = d_x * (float) flambeau_iq4nl_lut(code);

        const float d_y = (float) by->d;
        const int   qi  = (int) by->qs[elem_in_block];
        const float y_val = d_y * (float) qi;

        acc += x_val * y_val;
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[row] = acc;
    }
}
