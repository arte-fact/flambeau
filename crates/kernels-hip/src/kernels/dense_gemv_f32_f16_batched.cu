// dense_gemv_f32_f16_batched — V2.31.g batched dense GEMV.
//
// Extension of `dense_gemv_f32_f16.cu` with an outer token dimension.
// Same inner warp-reduce math; adds blockIdx.y as token index.
//
//   y[t, n] = Σ_k  w[n, k] · (float) x[t, k]
//
// Target: the MoE router at prefill. The non-batched version launches
// L × n_layers kernels (e.g. 512 × 40 = 20520 for 35B at L=512 =
// 9 % of prefill wall in V2.30.b profile). A batched launch collapses
// the L dimension into grid.y — 40 launches instead of 20520.
//
// Launch:
//   gridDim  = { n_rows, n_tokens, 1 }
//   blockDim = { 256, 1, 1 }
//   shared   = 4 floats

#include <hip/hip_runtime.h>
#include "block_quant.cuh"

#ifndef DENSE_GEMV_THREADS
#define DENSE_GEMV_THREADS 256
#endif
#define DENSE_GEMV_WARPS (DENSE_GEMV_THREADS / 64)

extern "C" __global__ void flambeau_dense_gemv_f32_f16_batched(
    const float*    __restrict__ w,     // [n_rows, k]
    const fb_fp16_t* __restrict__ x,    // [n_tokens, k]
    float*          __restrict__ y,     // [n_tokens, n_rows]
    const int n_rows,
    const int k,
    const int n_tokens
) {
    const int row   = blockIdx.x;
    const int token = blockIdx.y;
    if (row >= n_rows || token >= n_tokens) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const float*     w_row = w + (size_t) row   * k;
    const fb_fp16_t* x_row = x + (size_t) token * k;

    float local = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < k; i += DENSE_GEMV_THREADS) {
        local += w_row[i] * (float) x_row[i];
    }

    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        local += __shfl_xor(local, off, 64);
    }

    __shared__ float warp_sums[DENSE_GEMV_WARPS];
    if (lane == 0) {
        warp_sums[warp] = local;
    }
    __syncthreads();

    if (warp == 0) {
        float s = (lane < DENSE_GEMV_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = DENSE_GEMV_WARPS / 2; off > 0; off >>= 1) {
            s += __shfl_xor(s, off, 64);
        }
        if (lane == 0) {
            y[(size_t) token * n_rows + row] = s;
        }
    }
}
