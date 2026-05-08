// mmvq_q8_0_r4_dp4a — 1.b Q8_0 MMVQ, 4 rows per block (wave64).
// Candle P29-style multi-row packing for the Q8_0 decode path. Targeted
// at Qwen3.6-27B-Q8_0 decode, where the existing single-row
// `mmvq_q8_0_dp4a_vdr2` (256 threads/block, 1 row/block) takes **81 %**
// of decode wall (0.b profile) — 21264 calls for tg=64, launch
// overhead dominates.
// Compared to `mmvq_q8_0_dp4a_vdr2`:
// vdr2 block = 256 threads, 1 row, 4 lane_in_grp × 64 blocks/iter
// r4 block = 64 threads, 4 rows, 4 lane_in_grp × 4 blocks/iter
// Four rows per launch means 4× fewer blocks; 16 lanes per row means the
// warp-reduce collapses via `gfx906_quarter_warp_reduce_sum` (the same
// primitive `indexed_moe_mmvq_q4_k_r4_dp4a` uses).
// Math identity to vdr2 is exact: per-block `sumi = dp4a(xi0,yi0) +
// dp4a(xi1,yi1)`, scaled by `d_x * d_y`. Accumulation order differs but
// F32 addition is associative within the same-block chain we preserve.

#include "block_quant.cuh"
#include "gfx906.cuh"

static __device__ __forceinline__ int flambeau_dp4a_q8_r4(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_mmvq_q8_0_r4_dp4a_q8_1(
    const flambeau_block_q8_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int row_quad = blockIdx.x;
    const int lane     = threadIdx.x;            // 0..63
    const int row_idx  = lane >> 4;              // 0..3: which row in the quad
    const int lane_lo  = lane & 15;              // 0..15: quarter-warp position

    const int row = row_quad * 4 + row_idx;
    if (row >= n_rows) return;

    // lane_lo layout: 4 lane_in_grp × 4 block_in_iter → 4 lanes handle
    // one block; 4 blocks per iter across the 16 lanes.
    const int lane_in_grp   = lane_lo & 3;       // 0..3: int32-pair within a block
    const int block_in_iter = lane_lo >> 2;      // 0..3: which block in the iter

    const flambeau_block_q8_0* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_in_iter; b < n_blocks_per_row; b += 4) {
        const flambeau_block_q8_0* bx = xrow + b;
        const flambeau_block_q8_1* by = y  + b;

        // VDR=2: two consecutive int32s per lane.
        const int xi0 = ((const int*) bx->qs)[lane_in_grp * 2 + 0];
        const int xi1 = ((const int*) bx->qs)[lane_in_grp * 2 + 1];
        const int yi0 = ((const int*) by->qs)[lane_in_grp * 2 + 0];
        const int yi1 = ((const int*) by->qs)[lane_in_grp * 2 + 1];

        int sumi = flambeau_dp4a_q8_r4(xi0, yi0, 0);
        sumi     = flambeau_dp4a_q8_r4(xi1, yi1, sumi);

        const float d = (float) bx->d * (float) by->d;
        acc += d * (float) sumi;
    }

    // Quarter-warp (16-lane) reduce — same primitive as Q4_K r4 MoE.
    acc = gfx906_quarter_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[row] = acc;
    }
}
