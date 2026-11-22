// gelu_mul_f32 — fused y[i] = gelu(a[i]) * b[i], pure F32, tanh-approx GELU.
// Pointwise. Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <cuda_runtime.h>

#define GELU_SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define GELU_COEF_A         0.044715f

extern "C" __global__ void flambeau_gelu_mul_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = a[i];
    const float t = GELU_SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x);
    const float gelu = 0.5f * x * (1.0f + tanhf(t));
    y[i] = gelu * b[i];
}
