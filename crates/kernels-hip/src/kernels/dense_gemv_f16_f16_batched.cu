// dense_gemv_f16_f16_batched — (iter-3) F16-weight
// variant of `dense_gemv_f32_f16_batched`. Halves the per-row HBM
// weight bandwidth at the cost of one F16→float conversion per FMA.
// y[t, n] = Σ_k (float) w[n, k] · (float) x[t, k]
// Used by the MoE router at prefill when the loader has converted the
// F32 source `ffn_gate_inp` to F16. Same launch shape as the F32
// variant; only the weight load + cast differs.

#include <hip/hip_runtime.h>
#include "block_quant.cuh"

#ifndef DENSE_GEMV_F16_BATCHED_THREADS
#define DENSE_GEMV_F16_BATCHED_THREADS 256
#endif
#define DENSE_GEMV_F16_BATCHED_WARPS (DENSE_GEMV_F16_BATCHED_THREADS / 64)

extern "C" __global__ void flambeau_dense_gemv_f16_f16_batched(
    const fb_fp16_t* __restrict__ w,    // [n_rows, k]
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

    const fb_fp16_t* w_row = w + (size_t) row   * k;
    const fb_fp16_t* x_row = x + (size_t) token * k;

    float local = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < k; i += DENSE_GEMV_F16_BATCHED_THREADS) {
        local += (float) w_row[i] * (float) x_row[i];
    }

    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        local += __shfl_xor(local, off, 64);
    }

    __shared__ float warp_sums[DENSE_GEMV_F16_BATCHED_WARPS];
    if (lane == 0) {
        warp_sums[warp] = local;
    }
    __syncthreads();

    if (warp == 0) {
        float s = (lane < DENSE_GEMV_F16_BATCHED_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = DENSE_GEMV_F16_BATCHED_WARPS / 2; off > 0; off >>= 1) {
            s += __shfl_xor(s, off, 64);
        }
        if (lane == 0) {
            y[(size_t) token * n_rows + row] = s;
        }
    }
}
