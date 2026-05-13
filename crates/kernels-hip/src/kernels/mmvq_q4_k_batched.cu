// Q4_K batched per-N MMVQ. Single-row template from mmvq_q4_k.cu —
// 64 threads/block, one output row per block, super-block decoded once
// per lane per (grp, hi_half). N activation slots inner-loop the per-
// sub-block dot.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q4K_BATCHED_THREADS 64

template <int N>
__device__ __forceinline__ void flambeau_mmvq_q4_k_batched_body(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane     = threadIdx.x;
    const int byte_off = lane & 31;
    const int hi_half  = lane >> 5;

    const flambeau_block_q4_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc[N];
    #pragma unroll
    for (int s = 0; s < N; ++s) acc[s] = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q4_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        #pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const int sub = 2 * grp + hi_half;

            uint8_t sc = 0, m = 0;
            flambeau_q4k_scale_min(sub, bk->scales, &sc, &m);

            const int byte_v = (int) bk->qs[grp * 32 + byte_off];
            const int raw_q  = hi_half ? (byte_v >> 4) : (byte_v & 0x0F);

            const float x_val = d * (float) sc * (float) raw_q - dmin * (float) m;

            #pragma unroll
            for (int s = 0; s < N; ++s) {
                const flambeau_block_q8_1* ya =
                    y + (size_t) s * n_superblocks_per_row * 8 + b * 8 + sub;
                const float d_y = (float) ya->d;
                const int   qi  = (int) ya->qs[byte_off];
                acc[s] += x_val * d_y * (float) qi;
            }
        }
    }

    #pragma unroll
    for (int s = 0; s < N; ++s) {
        acc[s] = gfx906_warp_reduce_sum(acc[s]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int s = 0; s < N; ++s) {
            dst[(size_t) s * n_rows + row] = acc[s];
        }
    }
}

extern "C" __global__ void flambeau_mmvq_q4_k_q8_1_batched_n2(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    flambeau_mmvq_q4_k_batched_body<2>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q4_k_q8_1_batched_n3(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    flambeau_mmvq_q4_k_batched_body<3>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q4_k_q8_1_batched_n4(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    flambeau_mmvq_q4_k_batched_body<4>(x, y, dst, n_rows, n_superblocks_per_row);
}
