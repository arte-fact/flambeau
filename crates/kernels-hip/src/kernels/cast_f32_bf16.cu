// cast_f32_bf16 — pointwise F32 → BF16.
// bridge from F32 mmvq accumulator output to BF16 activation
// for the BF16-throughout MTP forward path. Round-to-nearest-even on
// the lower 16 bits of the F32 bit pattern; NaN preserved as quiet NaN.
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <hip/hip_runtime.h>
#include "block_quant.cuh"

extern "C" __global__ void flambeau_cast_f32_bf16(
    const float* __restrict__ x,
    fb_bf16_t* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = fb_f32_to_bf16(x[i]);
}
