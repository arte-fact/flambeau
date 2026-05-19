// rmsnorm_f32 — per-row RMS normalisation, F32 weight + F32 in/out.
// Identical math to rmsnorm_f16 but keeps F32 precision end-to-end for
// GDN's ssm_norm step (applied per-head on the state-step F32 output).
// Launch:
// gridDim = { n_rows, 1, 1 }
// blockDim = { 256, 1, 1 }
// shared = 4 floats for cross-warp reduce.
// k should be > 0; k >= 256 is optimal.

#include <hip/hip_runtime.h>

#ifndef RMSNORM_F32_THREADS
#define RMSNORM_F32_THREADS 256
#endif
#define RMSNORM_F32_WARPS (RMSNORM_F32_THREADS / 64)

extern "C" __global__ void flambeau_rmsnorm_f32(
    const float* __restrict__ x,        // [n_rows, k]
    const float* __restrict__ weight,   // [k]
    float* __restrict__ y,              // [n_rows, k]
    const int n_rows,
    const int k,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const float* x_row = x + (size_t) row * k;
    float*       y_row = y + (size_t) row * k;

    // Vectorised path when k is a multiple of 4: each thread issues
    // float4 (16-byte) loads to cut HBM transactions and saturate the
    // 1024-bit-per-CU LDS path. k=2816 (gemma4 hidden) / k=2304 / k=5120
    // all satisfy the alignment.
    const bool vec4 = (k % 4) == 0
        && ((reinterpret_cast<uintptr_t>(x_row) & 0xF) == 0)
        && ((reinterpret_cast<uintptr_t>(y_row) & 0xF) == 0)
        && ((reinterpret_cast<uintptr_t>(weight) & 0xF) == 0);

    float sum_sq = 0.0f;
    if (vec4) {
        const int k4 = k >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(x_row);
        for (int i = tid; i < k4; i += RMSNORM_F32_THREADS) {
            const float4 v = x4[i];
            sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int i = tid; i < k; i += RMSNORM_F32_THREADS) {
            const float v = x_row[i];
            sum_sq += v * v;
        }
    }

    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }

    __shared__ float warp_sums[RMSNORM_F32_WARPS];
    if (lane == 0) {
        warp_sums[warp] = sum_sq;
    }
    __syncthreads();

    float total_sq;
    if (warp == 0) {
        float s = (lane < RMSNORM_F32_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = RMSNORM_F32_WARPS / 2; off > 0; off >>= 1) {
            s += __shfl_xor(s, off, 64);
        }
        if (lane == 0) {
            warp_sums[0] = s;
        }
    }
    __syncthreads();
    total_sq = warp_sums[0];

    const float mean_sq = total_sq / (float) k;
    const float rsqrt = 1.0f / sqrtf(mean_sq + eps);

    if (vec4) {
        const int k4 = k >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(x_row);
        const float4* w4 = reinterpret_cast<const float4*>(weight);
        float4*       y4 = reinterpret_cast<float4*>(y_row);
        for (int i = tid; i < k4; i += RMSNORM_F32_THREADS) {
            const float4 v = x4[i];
            const float4 w = w4[i];
            float4 r;
            r.x = v.x * w.x * rsqrt;
            r.y = v.y * w.y * rsqrt;
            r.z = v.z * w.z * rsqrt;
            r.w = v.w * w.w * rsqrt;
            y4[i] = r;
        }
    } else {
        for (int i = tid; i < k; i += RMSNORM_F32_THREADS) {
            y_row[i] = x_row[i] * weight[i] * rsqrt;
        }
    }
}
