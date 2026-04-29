// cast_bf16_f32 — pointwise BF16 → F32.
//
// MTP-4-C-1: lossless lift from BF16 storage into F32 for accumulators
// or kernels that don't yet have a BF16 variant. Pure bit-shift; no
// rounding loss.
//
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <hip/hip_runtime.h>
#include "block_quant.cuh"

extern "C" __global__ void flambeau_cast_bf16_f32(
    const fb_bf16_t* __restrict__ x,
    float* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = fb_bf16_to_f32(x[i]);
}
