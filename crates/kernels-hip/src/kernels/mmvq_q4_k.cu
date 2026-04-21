// mmvq_q4_k — Q4_K weight matrix × Q8_1 activation → F32 dst.
//
// V1.3 cert-grade single-row reference. Port strategy:
//   * On-the-fly dequantise: each lane computes `x = d * sc * raw_q - dmin * m`
//     on the fly, multiplies by the Q8_1 activation (already dequantised
//     via `d8 * qi`), and accumulates in F32.
//   * 64 threads per block = one wave64 = one output row per block.
//   * 1 super-block of 256 elements per outer loop iteration, 4 elements per
//     lane per super-block (partitioned into 4 groups of 64).
//   * Final reduction via `gfx906_warp_reduce_sum` (DPP fused).
//
// This kernel is the correctness oracle for the dp4a-optimised P29 multi-row
// variant (candle `indexed_moe_forward_q4k_q8_1_nw1_r2`). Port of the perf
// version follows once the cert harness is in place.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q4K_THREADS 64

extern "C" __global__ void flambeau_mmvq_q4_k_q8_1(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane     = threadIdx.x;           // 0..63
    const int byte_off = lane & 31;             // 0..31
    const int hi_half  = lane >> 5;             // 0 (low nibble) or 1 (high nibble)

    const flambeau_block_q4_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q4_K* bk = xrow + b;

        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;  // 8 Q8_1 blocks per super-block

        #pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            // Sub-block index — lanes 0..31 own sub-block 2*grp, lanes 32..63 own 2*grp+1.
            const int sub = 2 * grp + hi_half;

            uint8_t sc = 0, m = 0;
            flambeau_q4k_scale_min(sub, bk->scales, &sc, &m);

            const int byte_v = (int) bk->qs[grp * 32 + byte_off];
            const int raw_q  = hi_half ? (byte_v >> 4) : (byte_v & 0x0F);

            const float x_val = d * (float) sc * (float) raw_q - dmin * (float) m;

            // Activation: y_sb[sub].qs[byte_off] × y_sb[sub].d (dequantised).
            const flambeau_block_q8_1* ya = y_sb + sub;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[byte_off];
            const float y_val = d_y * (float) qi;

            acc += x_val * y_val;
        }
    }

    // Single wave64 → one full-warp reduce lands the sum in every lane.
    acc = gfx906_warp_reduce_sum(acc);

    if (lane == 0) {
        dst[row] = acc;
    }
}
