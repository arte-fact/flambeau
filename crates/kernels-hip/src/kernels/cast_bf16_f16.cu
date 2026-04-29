// cast_bf16_f16 — pointwise BF16 → F16 via F32.
//
// MTP-4-C-1: BF16 has 8 exponent bits vs F16's 5, so values with
// |x| > 65504 saturate to F16 ±Inf; values with |x| < 6.1e-5
// underflow to F16 subnormals or zero. Routing through F32 keeps
// the rounding explicit and matches host-side `bf16::to_f32` then
// `f16::from_f32`.
//
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <hip/hip_runtime.h>
#include "block_quant.cuh"

extern "C" __global__ void flambeau_cast_bf16_f16(
    const fb_bf16_t* __restrict__ x,
    fb_fp16_t* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = (fb_fp16_t) fb_bf16_to_f32(x[i]);
}
