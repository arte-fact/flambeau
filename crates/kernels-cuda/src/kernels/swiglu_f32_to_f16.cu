// swiglu_f32_to_f16 — fused y[i] = (fp16)(silu(a[i]) * b[i]), saturated at
// ±F16_MAX. Pointwise. Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <cuda_runtime.h>
#include "block_quant.cuh"

extern "C" __global__ void flambeau_swiglu_f32_to_f16(
    const float* __restrict__ a,
    const float* __restrict__ b,
    fb_fp16_t* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float av = a[i];
    const float silu = av / (1.0f + __expf(-av));
    float v = silu * b[i];
    if (v > 65504.0f) v = 65504.0f;
    else if (v < -65504.0f) v = -65504.0f;
    y[i] = (fb_fp16_t) v;
}
