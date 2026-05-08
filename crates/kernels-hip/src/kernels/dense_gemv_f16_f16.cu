// dense_gemv_f16_f16 — per-row dense GEMV with F16 weight + F16 activation,
// F32 output.
// 6 (iter-3) sibling of `dense_gemv_f32_f16`. Halves the
// per-row HBM weight bandwidth (2 B/elem vs 4 B/elem) at the cost of one
// extra fp16→float conversion per FMA. On Coder-Next-Q4_0 the router
// runs in every one of 48 layers per decode token — saving HBM here
// chips a measurable slice off the 14.9 % of decode wall the router was
// taking in iter-2 (with F32 weights).
// Quality: the router projects to a 512-dim logit vector that's then
// topk-sampled; F16 vs F32 weight precision shifts logit values by
// ~1e-3 relative — well within Q8 KV's quality envelope (max KL 3.6e-3,
// PASS gate 5e-2) measured in Validated end-to-end
// against the F16-KV / Q8-KV parity tests as part of iter-3.
// Launch:
// gridDim = { n_rows, 1, 1 }
// blockDim = { 256, 1, 1 } // 4 wave64 per block
// shared = 4 floats (one per warp partial-sum)

#include <hip/hip_runtime.h>
#include "block_quant.cuh"

#ifndef DENSE_GEMV_F16_THREADS
#define DENSE_GEMV_F16_THREADS 256
#endif
#define DENSE_GEMV_F16_WARPS (DENSE_GEMV_F16_THREADS / 64)

extern "C" __global__ void flambeau_dense_gemv_f16_f16(
    const fb_fp16_t* __restrict__ w,     // [n_rows, k]
    const fb_fp16_t* __restrict__ x,     // [k]
    float* __restrict__ y,               // [n_rows]
    const int n_rows,
    const int k
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const fb_fp16_t* w_row = w + (size_t) row * k;

    float local = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < k; i += DENSE_GEMV_F16_THREADS) {
        local += (float) w_row[i] * (float) x[i];
    }

    // Warp reduce (wave64).
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        local += __shfl_xor(local, off, 64);
    }

    __shared__ float warp_sums[DENSE_GEMV_F16_WARPS];
    if (lane == 0) {
        warp_sums[warp] = local;
    }
    __syncthreads();

    if (warp == 0) {
        float s = (lane < DENSE_GEMV_F16_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = DENSE_GEMV_F16_WARPS / 2; off > 0; off >>= 1) {
            s += __shfl_xor(s, off, 64);
        }
        if (lane == 0) {
            y[row] = s;
        }
    }
}
