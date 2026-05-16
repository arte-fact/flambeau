// rmsnorm_f32_in_f16_out — per-row RMSNorm reading F32 input, F16
// learned weight, writing saturating F16 output.
//
// Used by gemma4 26B-A4B-Q8_0 PP path's `with_f32_qkv(true)`: the
// Q/K/V projection F32 output (from mmvq's per-block-scaled
// accumulator) is normalised in F32 and only down-cast to F16 at the
// rmsnorm output store. Matches llama.cpp's gemma4-iswa.cpp where Q,
// K, V are kept F32 across the per-head rmsnorm and only become F16
// at the KV cache write / attention input boundary. The legacy
// `rmsnorm_f16` path goes F32→F16-cast→rmsnorm_f16, which
// information-loses every F32 value > F16_MAX to the same saturated
// 65504 and produces ill-conditioned per-head magnitudes.
//
// Output is saturated at ±F16_MAX (matches mmvq_store<fb_fp16_t>,
// cast_f32_f16, gelu_f32_to_f16, swiglu_f32_to_f16, add_f16).
// Launch identical to rmsnorm_f16: blockDim={256}, gridDim={m},
// 4 floats LDS for cross-warp reduce. k should be a multiple of 256;
// 64/128/256/512 are tested values (gemma4 head_dim set).

#include <hip/hip_runtime.h>

#define RMSNORM_THREADS 256
#define RMSNORM_WARPS (RMSNORM_THREADS / 64)

typedef _Float16 fb_fp16_t;

extern "C" __global__ void flambeau_rmsnorm_f32_in_f16_out(
    const float*     __restrict__ x,      // [m, k] F32
    const fb_fp16_t* __restrict__ weight, // [k]    F16 learned norm weight
    fb_fp16_t*       __restrict__ y,      // [m, k] F16 (saturating)
    const int m,
    const int k,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= m) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const float*    xrow = x + (size_t) row * k;
    fb_fp16_t*      yrow = y + (size_t) row * k;

    // Phase 1: sum-of-squares (F32).
    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < k; i += RMSNORM_THREADS) {
        const float v = xrow[i];
        sum_sq += v * v;
    }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }
    __shared__ float s_warp[RMSNORM_WARPS];
    if (lane == 0) {
        s_warp[warp] = sum_sq;
    }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < RMSNORM_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = RMSNORM_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 64);
        }
        if (lane == 0) {
            s_warp[0] = v;
        }
    }
    __syncthreads();
    const float total_sq = s_warp[0];
    const float mean_sq  = total_sq / (float) k;
    const float rsqrt    = 1.0f / sqrtf(mean_sq + eps);

    // Phase 2: scale + saturating F16 store.
    #pragma unroll 4
    for (int i = tid; i < k; i += RMSNORM_THREADS) {
        const float w = (float) weight[i];
        float out = xrow[i] * w * rsqrt;
        if (out > 65504.0f) out = 65504.0f;
        else if (out < -65504.0f) out = -65504.0f;
        yrow[i] = (fb_fp16_t) out;
    }
}
