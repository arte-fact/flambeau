// gelu_f32_to_f16 — fused y[i] = (fp16)(gelu(a[i]) * b[i]), tanh-approx GELU
// (ggml LLM_FFN_GELU parity), saturated at ±F16_MAX. Pointwise.
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <cuda_runtime.h>
#include "block_quant.cuh"

#define GELU_SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define GELU_COEF_A         0.044715f

extern "C" __global__ void flambeau_gelu_f32_to_f16(
    const float* __restrict__ a,
    const float* __restrict__ b,
    fb_fp16_t* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = a[i];
    const float t = GELU_SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x);
    const float gelu = 0.5f * x * (1.0f + tanhf(t));
    float v = gelu * b[i];
    if (v > 65504.0f) v = 65504.0f;
    else if (v < -65504.0f) v = -65504.0f;
    y[i] = (fb_fp16_t) v;
}
