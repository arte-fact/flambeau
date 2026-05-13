// Q8_0 batched per-N MMVQ. Compile-time N=2/3/4 specialisations like
// mmvq_q4_0_batched, but Q8_0's symmetric signed-8-bit layout drops the
// nibble unpack and the -8·s_y correction — per block this is just
// `dp4a(xi, yi, 0) · d_x · d_y` summed across slots.
//
// Thread layout: lane8 = tid & 7 (8 int32 per Q8_0 block), block_idx =
// tid >> 3 (256 / 8 = 32 blocks/iter). Output `dst[N, n_rows]` slot-major.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q8_0_BATCHED_THREADS 256
#define MMVQ_Q8_0_BATCHED_WARPS (MMVQ_Q8_0_BATCHED_THREADS / WARP_SIZE)
#define MMVQ_Q8_0_BATCHED_INT32_PER_BLOCK 8
#define MMVQ_Q8_0_BATCHED_BLOCKS_PER_ITER \
    (MMVQ_Q8_0_BATCHED_THREADS / MMVQ_Q8_0_BATCHED_INT32_PER_BLOCK)

static __device__ __forceinline__ int flambeau_q8_0_batched_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template <int N>
__device__ __forceinline__ void flambeau_mmvq_q8_0_batched_body(
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
    const int lane8     = tid & 7;
    const int block_idx = tid >> 3;

    const flambeau_block_q8_0* xrow = x + (size_t) row * n_blocks_per_row;

    float acc[N];
    #pragma unroll
    for (int s = 0; s < N; ++s) {
        acc[s] = 0.0f;
    }

    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_Q8_0_BATCHED_BLOCKS_PER_ITER) {
        const flambeau_block_q8_0* bx = xrow + b;
        const int xi = ((const int*) bx->qs)[lane8];
        const float d_x = (float) bx->d;

        #pragma unroll
        for (int s = 0; s < N; ++s) {
            const flambeau_block_q8_1* by =
                y + (size_t) s * n_blocks_per_row + b;
            const int yi = ((const int*) by->qs)[lane8];
            const int sumi = flambeau_q8_0_batched_dp4a(xi, yi, 0);
            const float d_y = (float) by->d;
            acc[s] += (float) sumi * d_x * d_y;
        }
    }

    __shared__ float s_warp[MMVQ_Q8_0_BATCHED_WARPS * N];

    #pragma unroll
    for (int s = 0; s < N; ++s) {
        float v_red = gfx906_warp_reduce_sum(acc[s]);
        if (lane == 0) {
            s_warp[s * MMVQ_Q8_0_BATCHED_WARPS + warp] = v_red;
        }
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int s = 0; s < N; ++s) {
            float v_red = (lane < MMVQ_Q8_0_BATCHED_WARPS)
                ? s_warp[s * MMVQ_Q8_0_BATCHED_WARPS + lane]
                : 0.0f;
            #pragma unroll
            for (int off = MMVQ_Q8_0_BATCHED_WARPS / 2; off > 0; off >>= 1) {
                v_red += __shfl_xor(v_red, off, WARP_SIZE);
            }
            if (lane == 0) {
                dst[(size_t) s * n_rows + row] = v_red;
            }
        }
    }
}

extern "C" __global__ void flambeau_mmvq_q8_0_q8_1_batched_n2(
    const flambeau_block_q8_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q8_0_batched_body<2>(x, y, dst, n_rows, n_blocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q8_0_q8_1_batched_n3(
    const flambeau_block_q8_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q8_0_batched_body<3>(x, y, dst, n_rows, n_blocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q8_0_q8_1_batched_n4(
    const flambeau_block_q8_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q8_0_batched_body<4>(x, y, dst, n_rows, n_blocks_per_row);
}
