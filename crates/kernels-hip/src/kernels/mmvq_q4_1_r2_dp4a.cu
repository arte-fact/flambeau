// mmvq_q4_1_r2_dp4a — V2.24.a.2 DP4A multi-row Q4_1 MMVQ.
//
// Sibling of `mmvq_q4_1.cu` (V2.2.b) that emits TWO output rows per block
// while keeping the 256-thread DP4A inner loop. Halves grid.x (and launch
// count) + shares the Y read across the two rows (served from L1).
//
// V2.24.a.1 scalar r2 variant was NULL (2.3× slower) — the Q4_1 block
// structure is dense-packed so DP4A is essential. This DP4A r2 version
// preserves DP4A and only rearranges the block layout for multi-row
// output.
//
// Block/grid:
//   blockDim = 256, gridDim = ceil(n_rows / 2)
//   Same 4-threads-per-Q4_1-block layout as the single-row kernel; each
//   thread now computes partial products for 2 weight rows per iteration.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q4_1_R2_THREADS 256
#define MMVQ_Q4_1_R2_WARPS (MMVQ_Q4_1_R2_THREADS / WARP_SIZE)
#define MMVQ_Q4_1_R2_BLOCKS_PER_ITER (MMVQ_Q4_1_R2_THREADS / 4)

static __device__ __forceinline__ int flambeau_q4_1_r2_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_mmvq_q4_1_r2_dp4a_q8_1(
    const flambeau_block_q4_1* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int row_pair = blockIdx.x;
    const int row0 = row_pair * 2;
    const int row1 = row_pair * 2 + 1;
    if (row0 >= n_rows) return;
    const bool row1_ok = (row1 < n_rows);

    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid & (WARP_SIZE - 1);
    const int lane4     = tid & 3;
    const int block_idx = tid >> 2;

    const flambeau_block_q4_1* xrow0 = x + (size_t) row0 * n_blocks_per_row;
    const flambeau_block_q4_1* xrow1 = x + (size_t) row1 * n_blocks_per_row;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_Q4_1_R2_BLOCKS_PER_ITER) {
        const flambeau_block_q8_1* by = y + b;
        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        // Row 0.
        const flambeau_block_q4_1* bx0 = xrow0 + b;
        const int v0 = ((const int*) bx0->qs)[lane4];
        const int vi0_lo = (v0 >> 0) & 0x0F0F0F0F;
        const int vi0_hi = (v0 >> 4) & 0x0F0F0F0F;
        int sumi0 = flambeau_q4_1_r2_dp4a(vi0_lo, u_lo, 0);
        sumi0     = flambeau_q4_1_r2_dp4a(vi0_hi, u_hi, sumi0);
        const float d0 = (float) bx0->d;
        const float m0 = (float) bx0->m;
        acc0 += sumi0 * (d0 * d_y) + (m0 * s_y) * 0.25f;

        // Row 1 (guarded — odd n_rows tail).
        if (row1_ok) {
            const flambeau_block_q4_1* bx1 = xrow1 + b;
            const int v1 = ((const int*) bx1->qs)[lane4];
            const int vi1_lo = (v1 >> 0) & 0x0F0F0F0F;
            const int vi1_hi = (v1 >> 4) & 0x0F0F0F0F;
            int sumi1 = flambeau_q4_1_r2_dp4a(vi1_lo, u_lo, 0);
            sumi1     = flambeau_q4_1_r2_dp4a(vi1_hi, u_hi, sumi1);
            const float d1 = (float) bx1->d;
            const float m1 = (float) bx1->m;
            acc1 += sumi1 * (d1 * d_y) + (m1 * s_y) * 0.25f;
        }
    }

    acc0 = gfx906_warp_reduce_sum(acc0);
    if (row1_ok) acc1 = gfx906_warp_reduce_sum(acc1);

    __shared__ float s_warp0[MMVQ_Q4_1_R2_WARPS];
    __shared__ float s_warp1[MMVQ_Q4_1_R2_WARPS];
    if (lane == 0) {
        s_warp0[warp] = acc0;
        if (row1_ok) s_warp1[warp] = acc1;
    }
    __syncthreads();

    if (warp == 0) {
        float v0 = (lane < MMVQ_Q4_1_R2_WARPS) ? s_warp0[lane] : 0.0f;
        float v1 = (row1_ok && lane < MMVQ_Q4_1_R2_WARPS) ? s_warp1[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_Q4_1_R2_WARPS / 2; off > 0; off >>= 1) {
            v0 += __shfl_xor(v0, off, WARP_SIZE);
            v1 += __shfl_xor(v1, off, WARP_SIZE);
        }
        if (lane == 0) {
            dst[row0] = v0;
            if (row1_ok) dst[row1] = v1;
        }
    }
}
