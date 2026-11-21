// mmvq_q4_1_t128 — thin-block Q4_1 MMVQ.
// Same DP4A structure as `mmvq_q4_1.cu` but 128 threads/block instead of
// 256. Halves the total thread count per launch → less kernel dispatch
// overhead + higher CU occupancy (more blocks running concurrently on
// gfx906's 60 CUs, 10 waves/SIMD floor).
// 128 threads / 4-threads-per-Q4_1-block = 32 Q4_1 blocks processed per
// iteration. For Qwen3.5-9B hidden=5120 → n_blocks_per_row=160, each
// thread loops 160/32 = 5 iterations — ~5 DP4A pairs per thread, still
// small per-thread work but amortised across more grid.x blocks.
//
// Two output dtypes via templated __device__ body:
//   flambeau_mmvq_q4_1_t128_q8_1      → F32 dst (legacy scratch-then-cast)
//   flambeau_mmvq_q4_1_t128_q8_1_f16  → F16 dst (saturating; direct store)

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "mmvq_store.cuh"

#define MMVQ_Q4_1_T128_THREADS 128
#define MMVQ_Q4_1_T128_WARPS (MMVQ_Q4_1_T128_THREADS / WARP_SIZE)
#define MMVQ_Q4_1_T128_BLOCKS_PER_ITER (MMVQ_Q4_1_T128_THREADS / 4)

static __device__ __forceinline__ int flambeau_q4_1_t128_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template<typename OutT>
__device__ void mmvq_q4_1_t128_q8_1_body(
    const flambeau_block_q4_1* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid & (WARP_SIZE - 1);
    const int lane4     = tid & 3;
    const int block_idx = tid >> 2;

    const flambeau_block_q4_1* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_Q4_1_T128_BLOCKS_PER_ITER) {
        const flambeau_block_q4_1* bx = xrow + b;
        const flambeau_block_q8_1* by = y + b;

        const int v = ((const int*) bx->qs)[lane4];
        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];

        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;

        int sumi = 0;
        sumi = flambeau_q4_1_t128_dp4a(vi_lo, u_lo, sumi);
        sumi = flambeau_q4_1_t128_dp4a(vi_hi, u_hi, sumi);

        const float d_x = (float) bx->d;
        const float m_x = (float) bx->m;
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        acc += sumi * (d_x * d_y) + (m_x * s_y) * 0.25f;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_Q4_1_T128_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_Q4_1_T128_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_Q4_1_T128_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_q4_1_t128_q8_1(
    const flambeau_block_q4_1* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    mmvq_q4_1_t128_q8_1_body<float>(x, y, dst, n_rows, n_blocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q4_1_t128_q8_1_f16(
    const flambeau_block_q4_1* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    mmvq_q4_1_t128_q8_1_body<fb_fp16_t>(x, y, dst, n_rows, n_blocks_per_row);
}
