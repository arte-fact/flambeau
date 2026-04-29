// cast_f16_bf16 — pointwise F16 → BF16 via F32.
//
// MTP-4-C-1: BF16 has a wider exponent range than F16 (8 vs 5 bits)
// but a narrower mantissa (7 vs 10), so F16 → BF16 always fits in
// range but loses 3 mantissa bits. Routing through F32 makes the
// rounding step explicit.
//
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <hip/hip_runtime.h>
#include "block_quant.cuh"

extern "C" __global__ void flambeau_cast_f16_bf16(
    const fb_fp16_t* __restrict__ x,
    fb_bf16_t* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = fb_f32_to_bf16((float) x[i]);
}
