// dense_gemv_f32_f16 — per-row dense GEMV with F32 weight, F16 activation,
// F32 output.
// y[n] = Σ_k w[n, k] · (float) x[k]
// d3 use case: the MoE router, which has an F32 `ffn_gate_inp.weight`
// and consumes the F16 post-attention-norm activation to produce F32 logits
// fed to `topk_f32`.
// Weight layout (matches GGUF outermost-first): `[n_rows, k]` with `k`
// innermost/contiguous, so row `n` lives at offset `n * k`.
// Launch:
// gridDim = { n_rows, 1, 1 }
// blockDim = { 256, 1, 1 } // 4 wave64 per block
// shared = 4 floats (one per warp partial-sum)

#include <hip/hip_runtime.h>
#include "block_quant.cuh"

#ifndef DENSE_GEMV_THREADS
#define DENSE_GEMV_THREADS 256
#endif
#define DENSE_GEMV_WARPS (DENSE_GEMV_THREADS / 64)

extern "C" __global__ void flambeau_dense_gemv_f32_f16(
    const float* __restrict__ w,         // [n_rows, k]
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

    const float* w_row = w + (size_t) row * k;

    float local = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < k; i += DENSE_GEMV_THREADS) {
        local += w_row[i] * (float) x[i];
    }

    // Warp reduce (wave64).
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        local += __shfl_xor(local, off, 64);
    }

    // Cross-warp reduce via LDS: lane 0 of each warp writes its sum,
    // warp 0 reduces the 4 partials.
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
            y[row] = s;
        }
    }
}
