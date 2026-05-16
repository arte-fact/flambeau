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
    // F32 sum prevents F16 denormals biting; saturate at ±F16_MAX so
    // a sum of two near-max F16 values doesn't overflow to ±inf (which
    // would NaN through downstream rmsnorm variance = inf − inf).
    const float av = (float) a[i];
    const float bv = (float) b[i];
    float v = av + bv;
    if (v > 65504.0f) v = 65504.0f;
    else if (v < -65504.0f) v = -65504.0f;
    y[i] = (fb_fp16_t) v;
}
