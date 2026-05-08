// mmvq_q8_0_dp4a_vdr2 — Q8_0 MMVQ with VDR=2 DP4A inner loop.
// Refinement of mmvq_q8_0_dp4a.cu. Matches llama.cpp's
// vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ=2> pattern
// (vecdotq.cuh:243-255):
// int sumi = 0;
// for (int i = 0; i < 2; i++) sumi = dp4a(v[i], u[i], sumi);
// return (float)(d_x * d_y) * (float)sumi;
// vs our VDR=1 which did 1 dp4a + 1 float multiply PER BLOCK. This
// version does 2 dp4a + 1 float multiply per "pair slot" (half a block),
// so per-chunk float multiplies are halved and the two dp4a share a
// register chain, improving instruction-level parallelism.
// Thread layout (256 threads / block, 1 row / block):
// lane_in_grp = tid & 3 — 0..3: which int32-pair within a block (4 pairs per block of 8 int32s)
// block_idx = tid >> 2 — 0..63: block index (64 blocks covered per wave iter)
// Outer stride = 64 blocks.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_VDR2_BLOCK_THREADS 256
#define MMVQ_VDR2_WARPS_PER_BLOCK (MMVQ_VDR2_BLOCK_THREADS / WARP_SIZE)
#define MMVQ_VDR2_THREADS_PER_QBLK 4              // QI8_0 / VDR = 8/2 = 4 threads per quant block
#define MMVQ_VDR2_BLOCKS_PER_ITER (MMVQ_VDR2_BLOCK_THREADS / MMVQ_VDR2_THREADS_PER_QBLK)

static __device__ __forceinline__ int flambeau_dp4a_v2(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_mmvq_q8_0_dp4a_vdr2_q8_1(
    const flambeau_block_q8_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid         = threadIdx.x;
    const int warp        = tid / WARP_SIZE;
    const int lane        = tid & (WARP_SIZE - 1);
    const int lane_in_grp = tid & 3;                    // 0..3: int32-pair slot
    const int block_idx   = tid >> 2;                   // 0..63: which block

    const flambeau_block_q8_0* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_VDR2_BLOCKS_PER_ITER) {
        const flambeau_block_q8_0* bx = xrow + b;
        const flambeau_block_q8_1* by = y + b;

        // VDR=2: load 2 consecutive int32s (8 Q8 bytes) per thread.
        const int xi0 = ((const int*) bx->qs)[lane_in_grp * 2 + 0];
        const int xi1 = ((const int*) bx->qs)[lane_in_grp * 2 + 1];
        const int yi0 = ((const int*) by->qs)[lane_in_grp * 2 + 0];
        const int yi1 = ((const int*) by->qs)[lane_in_grp * 2 + 1];

        // Accumulate 2 dp4a into one int32 — scheduler can overlap the 2nd
        // dp4a's source-register fetch with the 1st's issue.
        int sumi = flambeau_dp4a_v2(xi0, yi0, 0);
        sumi     = flambeau_dp4a_v2(xi1, yi1, sumi);

        // One float multiply for the block's scale pair, then accumulate.
        const float d = (float) bx->d * (float) by->d;
        acc += d * (float) sumi;
    }

    // Warp-wide sum: sums across 4 lane-groups × 16 blocks-per-warp
    // → each lane holds full warp's partial row sum.
    acc = gfx906_warp_reduce_sum(acc);

    // Inter-warp reduce via LDS.
    __shared__ float s_warp[MMVQ_VDR2_WARPS_PER_BLOCK];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_VDR2_WARPS_PER_BLOCK) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_VDR2_WARPS_PER_BLOCK / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            dst[row] = v;
        }
    }
}
