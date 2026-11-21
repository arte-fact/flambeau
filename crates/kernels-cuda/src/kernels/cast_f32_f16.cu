// cast_f32_f16 — pointwise F32 → F16, saturating at ±F16_MAX so ±inf can't
// poison downstream F16 buffers. NaN inputs pass through.
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <cuda_runtime.h>
#include "block_quant.cuh"

extern "C" __global__ void flambeau_cast_f32_f16(
    const float* __restrict__ x,
    fb_fp16_t* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (v > 65504.0f) v = 65504.0f;
    else if (v < -65504.0f) v = -65504.0f;
    y[i] = (fb_fp16_t) v;
}
