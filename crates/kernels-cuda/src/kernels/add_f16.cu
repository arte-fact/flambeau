// add_f16 — pointwise `y[i] = a[i] + b[i]` in F16 (F32 accum, saturated at
// ±F16_MAX so a near-max sum can't overflow to ±inf and NaN through downstream
// rmsnorm)
// Launch: 1D, ceil(n/256) blocks × 256 threads. One element per thread.

#include <cuda_runtime.h>
#include "block_quant.cuh"

extern "C" __global__ void flambeau_add_f16(
    const fb_fp16_t* __restrict__ a,
    const fb_fp16_t* __restrict__ b,
    fb_fp16_t* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float av = (float) a[i];
    const float bv = (float) b[i];
    float v = av + bv;
    if (v > 65504.0f) v = 65504.0f;
    else if (v < -65504.0f) v = -65504.0f;
    y[i] = (fb_fp16_t) v;
}
