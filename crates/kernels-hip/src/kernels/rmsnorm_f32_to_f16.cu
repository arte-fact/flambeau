// rmsnorm_f32_to_f16 — per-row RMSNorm reading F32 input + F16 weight,
// writing F16 output. Fuses what would otherwise be
//   cast_f32_to_f16 → rmsnorm_f16
// into a single kernel — one launch instead of two, one pass over the
// row instead of two. Used by gemma4's post-attn / post-ffn norm path
// after `ar_sum_f32` produces an F32 delta that needs to go through
// the F16 post-norm weight and into the F16 residual.
//
// Launch:
// gridDim = { n_rows, 1, 1 }
// blockDim = { 256, 1, 1 }
// shared = 4 floats for cross-warp reduce (same shape as rmsnorm_f32).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef RMSNORM_F32_TO_F16_THREADS
#define RMSNORM_F32_TO_F16_THREADS 256
#endif
#define RMSNORM_F32_TO_F16_WARPS (RMSNORM_F32_TO_F16_THREADS / 64)

extern "C" __global__ void flambeau_rmsnorm_f32_to_f16(
    const float* __restrict__ x,           // F32 [n_rows, k]
    const __half* __restrict__ weight,     // F16 [k]
    __half* __restrict__ y,                // F16 [n_rows, k]
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
    __half*      y_row = y + (size_t) row * k;

    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < k; i += RMSNORM_F32_TO_F16_THREADS) {
        const float v = x_row[i];
        sum_sq += v * v;
    }

    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }

    __shared__ float warp_sums[RMSNORM_F32_TO_F16_WARPS];
    if (lane == 0) {
        warp_sums[warp] = sum_sq;
    }
    __syncthreads();

    float total_sq;
    if (warp == 0) {
        float s = (lane < RMSNORM_F32_TO_F16_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = RMSNORM_F32_TO_F16_WARPS / 2; off > 0; off >>= 1) {
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

    #pragma unroll 4
    for (int i = tid; i < k; i += RMSNORM_F32_TO_F16_THREADS) {
        const float w = __half2float(weight[i]);
        const float v = x_row[i] * w * rsqrt;
        y_row[i] = __float2half(v);
    }
}

// Fused: rmsnorm_f32 → cast F16 → add residual. Replaces the
// (rmsnorm_f32_to_f16 → add_f16) pair for gemma4's post-attn /
// post-ffn paths: read F32 delta + F16 norm weight + F16 residual,
// write F16 (residual + (rmsnorm(delta) * weight)). One pass over
// the row, in-place on the residual.
//
// Launch identical to `rmsnorm_f32_to_f16`.
extern "C" __global__ void flambeau_rmsnorm_f32_to_f16_add_residual(
    const float*  __restrict__ x,         // F32 [n_rows, k] — delta to norm
    const __half* __restrict__ weight,    // F16 [k]
    const __half* __restrict__ resid_in,  // F16 [n_rows, k]
    __half*       __restrict__ resid_out, // F16 [n_rows, k] — may alias resid_in
    const int n_rows,
    const int k,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const float*  x_row     = x         + (size_t) row * k;
    const __half* resid_row = resid_in  + (size_t) row * k;
    __half*       out_row   = resid_out + (size_t) row * k;

    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < k; i += RMSNORM_F32_TO_F16_THREADS) {
        const float v = x_row[i];
        sum_sq += v * v;
    }

    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }

    __shared__ float warp_sums[RMSNORM_F32_TO_F16_WARPS];
    if (lane == 0) {
        warp_sums[warp] = sum_sq;
    }
    __syncthreads();

    float total_sq;
    if (warp == 0) {
        float s = (lane < RMSNORM_F32_TO_F16_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = RMSNORM_F32_TO_F16_WARPS / 2; off > 0; off >>= 1) {
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

    #pragma unroll 4
    for (int i = tid; i < k; i += RMSNORM_F32_TO_F16_THREADS) {
        const float w = __half2float(weight[i]);
        const float v = x_row[i] * w * rsqrt;
        const float r = __half2float(resid_row[i]);
        float sum = r + v;
        // Saturate at ±F16_MAX — matches `add_f16`'s envelope.
        if (sum > 65504.0f) sum = 65504.0f;
        else if (sum < -65504.0f) sum = -65504.0f;
        out_row[i] = __float2half(sum);
    }
}
