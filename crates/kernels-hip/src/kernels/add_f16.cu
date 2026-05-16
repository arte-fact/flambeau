// add_f16 — pointwise `y[i] = a[i] + b[i]` in F16.
// e1 residual fan-in primitive: used between layers to sum the
// pre-layer residual with each per-layer delta, and at the end of the
// MoE+shared combine to merge the shared-expert contribution into the
// routed-MoE output (which already folds in the post-attn residual).
// Launch: 1D, ceil(n/256) blocks × 256 threads. One element per thread.

#include <hip/hip_runtime.h>
#include "block_quant.cuh"

extern "C" __global__ void flambeau_add_f16(
    const fb_fp16_t* __restrict__ a,
    const fb_fp16_t* __restrict__ b,
    fb_fp16_t* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Up-cast to F32 for the sum so F16 denormals / subtracts don't bite.
    // The result casts back to F16 with the compiler's ties-to-even rule.
    // Saturate at ±F16_MAX — un-saturated sum of two near-F16-max values
    // overflows F16 → ±inf, which then NaNs through downstream rmsnorm
    // (variance = inf - inf). Matches the saturating clamp in
    // cast_f32_f16 / gelu_f32_to_f16 / swiglu_f32_to_f16 (#108).
    const float av = (float) a[i];
    const float bv = (float) b[i];
    float v = av + bv;
    if (v > 65504.0f) v = 65504.0f;
    else if (v < -65504.0f) v = -65504.0f;
    y[i] = (fb_fp16_t) v;
}
