// mmvq_q4_1_r2 — 4.a.1 multi-row r2 DPP-reduce Q4_1 MMVQ.
// Sibling of `mmvq_q4_k_r2.cu` (candle P29 multi-row DPP pattern) for the
// legacy Q4_1 quant used by Qwen3.5-9B-Q4_1. Replaces the 256-thread,
// single-row `mmvq_q4_1.cu` () that is 45 % of Qwen3.5-9B decode
// tg=64 time on gfx906 (462 ms / 11,616 calls × 40 µs — measured
// 2026-04-24).
// Structure (matches Q4_K r2):
// 64 threads/block = 1 wave64. Lanes 0..31 compute row R+0; lanes 32..63
// compute row R+1. Block.x indexes the row *pair*. Each half-warp's 32
// lanes span the 32 elements of one Q4_1 block; lane_lo < 16 takes the
// low nibble, lane_lo >= 16 takes the high nibble. Y is shared across
// the two rows → per-launch cache traffic on the Q8_1 side halved vs
// two single-row launches.
// Gain is on launch count (N rows → ceil(N/2) blocks), not arithmetic —
// DP4A is replaced by FP mul because the per-lane "one element" pattern
// doesn't pack 4-at-a-time. Net win: per-call µs ↓ + fewer calls.

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_mmvq_q4_1_r2_q8_1(
    const flambeau_block_q4_1* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,                     // total output rows (may be odd)
    const int n_blocks_per_row            // Q4_1 blocks per row = k / 32
) {
    const int row_pair = blockIdx.x;
    const int lane     = threadIdx.x;     // 0..63
    const int row_hi   = lane >> 5;       // 0 → row R+0, 1 → row R+1
    const int lane_lo  = lane & 31;       // 0..31 — position within Q4_1 block

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;            // odd-row boundary

    const flambeau_block_q4_1* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_blocks_per_row; ++b) {
        const flambeau_block_q4_1* bx = xrow + b;
        const flambeau_block_q8_1* by = y + b;

        const float d = (float) bx->d;
        const float m = (float) bx->m;
        const float d_y = (float) by->d;

        // Lane 0..15 → element index == lane_lo → low nibble of qs[lane_lo].
        // Lane 16..31 → element index == lane_lo → high nibble of qs[lane_lo-16].
        const int byte_idx = lane_lo & 15;
        const int byte_v   = (int) (uint8_t) bx->qs[byte_idx];
        const int raw_q    = (lane_lo < 16) ? (byte_v & 0x0F) : ((byte_v >> 4) & 0x0F);

        const int qi = (int) by->qs[lane_lo];

        // x_val = raw_q * d + m (Q4_1 per-block affine quantisation)
        // y_val = qi * d_y (Q8_1 per-block linear quantisation)
        // acc += x_val * y_val
        const float x_val = (float) raw_q * d + m;
        const float y_val = (float) qi * d_y;
        acc += x_val * y_val;
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[row] = acc;
    }
}
