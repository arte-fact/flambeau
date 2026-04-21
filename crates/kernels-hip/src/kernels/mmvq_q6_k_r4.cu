// mmvq_q6_k_r4 — Q6_K MMVQ with r4 multi-row DPP reduce (candle P29 default).
//
// 64 threads per block = one wave64. Each 16-lane quarter-warp computes one
// output row; a wavefront produces 4 rows. `blockIdx.x` is the row-quadruple
// index.
//
// Per-lane work per super-block: 16 elements (4 sub-positions × 4 q_idx).
// This is 4× the single-row kernel's per-lane work, balancing the 4× row
// count per wave. Register usage stays at 22-ish because the inner layout
// is identical to the single-row path — we just unroll over `p` to reach
// all 64 candle-style "effective lanes" from 16 physical lanes.
//
// Final reduce: `gfx906_quarter_warp_reduce_sum` — DPP chain that stops at
// xor-8, keeping each 16-lane group's sum local (the DPP row_mask keeps
// lanes in the same 16-lane "row" from spilling into neighbours).

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_mmvq_q6_k_r4_q8_1(
    const flambeau_block_q6_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row_quad    = blockIdx.x;
    const int lane        = threadIdx.x;         // 0..63
    const int row_in_grp  = lane >> 4;           // 0..3
    const int lane_in_row = lane & 15;           // 0..15

    const int row = row_quad * 4 + row_in_grp;
    if (row >= n_rows) return;

    const flambeau_block_q6_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q6_K* bk = xrow + b;
        const float d = (float) bk->d;
        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        // NO `#pragma unroll` on this outer p loop — fully unrolling both
        // loops explodes VGPR to 64 (waves/SIMD = 4 on gfx906, worse than
        // the single-row's 10). With p looped, the compiler reuses temps
        // across iterations and VGPR drops back below the 26-VGPR wave-10
        // threshold.
        for (int p = 0; p < 4; ++p) {
            const int eff_lane = p * 16 + lane_in_row;   // 0..63 across p
            const int h        = eff_lane >> 5;          // 0 or 1
            const int pos      = eff_lane & 31;          // 0..31
            const int lsub     = pos >> 4;               // 0 or 1

            const uint8_t qh_byte = bk->qh[32 * h + pos];

            for (int q_idx = 0; q_idx < 4; ++q_idx) {
                const int ql_off = 64 * h + ((q_idx & 1) ? pos + 32 : pos);
                const int ql_byte = (int) bk->ql[ql_off];
                const int nibble  = (q_idx < 2) ? (ql_byte & 0x0F) : (ql_byte >> 4);
                const int qh_bits = (qh_byte >> (2 * q_idx)) & 0x3;
                const int raw_q   = (nibble | (qh_bits << 4)) - 32;

                const int scale_idx = 8 * h + 2 * q_idx + lsub;
                const int sc = (int) bk->scales[scale_idx];

                const float x_val = d * (float) sc * (float) raw_q;

                const int y_block = h * 4 + q_idx;
                const flambeau_block_q8_1* ya = y_sb + y_block;
                const float d_y = (float) ya->d;
                const int   qi  = (int) ya->qs[pos];
                const float y_val = d_y * (float) qi;

                acc += x_val * y_val;
            }
        }
    }

    acc = gfx906_quarter_warp_reduce_sum(acc);

    if (lane_in_row == 0) {
        dst[row] = acc;
    }
}
