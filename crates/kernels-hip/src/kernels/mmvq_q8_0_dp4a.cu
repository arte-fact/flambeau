// mmvq_q8_0_dp4a — Q8_0 weight × Q8_1 activation → F32 dst, DP4A inner loop.
// Same block/grid shape as `mmvq_q8_0`:
// blockDim = { 256 }
// gridDim = { N } (one block per output row)
// shared = 0
// The inner loop swaps per-element scalar `xi * yi * d_x * d_y` for the
// gfx906 `v_dot4_i32_i8` intrinsic: each thread owns ONE int32 (4 packed
// quants) of a block instead of one byte, runs one dp4a to get 4 int8×int8
// MACs in a single VALU, then applies the block scale once.
// Math equivalence to the scalar version is exact — dp4a with clamp=false
// does the same int32 accumulation the manual (xi*yi)-sum would.
// Thread layout per block-of-256:
// lane8 = tid & 7 — which int32 (0..7) within a quant block
// block_idx = tid >> 3 — which logical block stride (0..31)
// 32 blocks processed per iteration → outer loop strides `n_blocks_per_row`
// in 32-block steps from `block_idx`.
// First-class reference: llama.cpp `vec_dot_q8_0_q8_1_impl` +
// `vec_dot_q8_0_q8_1` in `ggml-cuda/vecdotq.cuh`, specialised for the
// MMVQ_BLOCK_THREADS=256, one-row-per-block shape of our `mmvq_q8_0`.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_DP4A_BLOCK_THREADS 256
#define MMVQ_DP4A_WARPS_PER_BLOCK (MMVQ_DP4A_BLOCK_THREADS / WARP_SIZE)
#define MMVQ_DP4A_INT32_PER_BLOCK 8           // QK8_0 / 4 = 8 int32 per quant block
#define MMVQ_DP4A_BLOCKS_PER_ITER (MMVQ_DP4A_BLOCK_THREADS / MMVQ_DP4A_INT32_PER_BLOCK)

static __device__ __forceinline__ int flambeau_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_mmvq_q8_0_dp4a_q8_1(
    const flambeau_block_q8_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid & (WARP_SIZE - 1);
    const int lane8     = tid & 7;            // which int32 (0..7) in a block
    const int block_idx = tid >> 3;           // 0..31 across the 256 threads

    const flambeau_block_q8_0* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_DP4A_BLOCKS_PER_ITER) {
        const flambeau_block_q8_0* bx = xrow + b;
        const flambeau_block_q8_1* by = y + b;

        const int xi = ((const int*) bx->qs)[lane8];
        const int yi = ((const int*) by->qs)[lane8];
        const int sumi = flambeau_dp4a(xi, yi, 0);

        const float d = (float) bx->d * (float) by->d;
        acc += d * (float) sumi;
    }

    // Warp-wide sum (64 lanes): sums across all 8 lane8 slots and all
    // (block_idx % 8) rows inside the warp.
    acc = gfx906_warp_reduce_sum(acc);

    // Inter-warp reduce via shared memory.
    __shared__ float s_warp[MMVQ_DP4A_WARPS_PER_BLOCK];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_DP4A_WARPS_PER_BLOCK) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_DP4A_WARPS_PER_BLOCK / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            dst[row] = v;
        }
    }
}
