// mmvq_q4_0_warpcoop64 — Q4_0 single-warp MMVQ. C6-i1.
//
// gfx906-decode latency-bound lever. The 256-thread baseline and 128-thread
// (`mmvq_q4_0_t128`) variant pay an LDS round-trip + cross-warp shfl in the
// reduce. At decode (n_rows × 1 output, k ≤ 4096), VALU compute per block is
// trivial (~1 ns of DP4A) and the wall is dominated by launch + occupancy
// turnover — so a single-warp schedule that skips LDS entirely and uses the
// gfx906 DPP butterfly reduce in-place can pack 4× more blocks/CU than the
// 256t variant on this same kernel family.
//
// Structure: blockDim=64 (one wave64), grid={n_rows, 1, 1}. Each thread
// handles a (block_idx, lane4) pair — 16 Q4_0 blocks/iter (64t / 4-int per
// block = 16). DP4A inner identical to the 256t kernel; only the reduce
// differs (`gfx906_warp_reduce_sum` once, no LDS, lane-0 writes).
//
// Reference for the schedule shape: iacopPBK / llamacpp's
// `vec_dot_q4_0_q8_1_impl` warp-cooperative pattern (one wave per row).

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q4_0_WARPCOOP64_THREADS WARP_SIZE  // == 64 on gfx906
#define MMVQ_Q4_0_WARPCOOP64_BLOCKS_PER_ITER (MMVQ_Q4_0_WARPCOOP64_THREADS / 4)

static __device__ __forceinline__ int flambeau_q4_0_wc64_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_mmvq_q4_0_warpcoop64_q8_1(
    const flambeau_block_q4_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid       = threadIdx.x;
    const int lane4     = tid & 3;
    const int block_idx = tid >> 2;

    const flambeau_block_q4_0* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_Q4_0_WARPCOOP64_BLOCKS_PER_ITER) {
        const flambeau_block_q4_0* bx = xrow + b;
        const flambeau_block_q8_1* by = y + b;

        const int v = ((const int*) bx->qs)[lane4];
        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];

        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;

        int sumi = 0;
        sumi = flambeau_q4_0_wc64_dp4a(vi_lo, u_lo, sumi);
        sumi = flambeau_q4_0_wc64_dp4a(vi_hi, u_hi, sumi);

        const float d_x = (float) bx->d;
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        // Q4_0 (q - 8) bias correction split across 4 lanes per block.
        acc += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
    }

    // Single warp — no LDS round-trip needed.
    acc = gfx906_warp_reduce_sum(acc);

    if (tid == 0) {
        dst[row] = acc;
    }
}
