// cast_f32_f16 — pointwise F32 → F16, saturating at ±F16_MAX.
// Decode-path bridge: MMVQ accumulates in F32; attention / rmsnorm /
// swiglu consume F16. Saturating clamp prevents ±inf from poisoning
// downstream F16 buffers (NaN through rmsnorm variance). NaN inputs
// pass through (clamp comparisons against NaN are false).
// Launch: 1D, ceil(n/256) blocks × 256 threads. One element per thread.

#include <hip/hip_runtime.h>
#include "block_quant.cuh"

#ifndef CAST_F32_F16_THREADS
#define CAST_F32_F16_THREADS 256
#endif

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
