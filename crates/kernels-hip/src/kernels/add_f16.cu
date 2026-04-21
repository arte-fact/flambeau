// add_f16 — pointwise `y[i] = a[i] + b[i]` in F16.
//
// V1.7.3-e1 residual fan-in primitive: used between layers to sum the
// pre-layer residual with each per-layer delta, and at the end of the
// MoE+shared combine to merge the shared-expert contribution into the
// routed-MoE output (which already folds in the post-attn residual).
//
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
    const float av = (float) a[i];
    const float bv = (float) b[i];
    y[i] = (fb_fp16_t) (av + bv);
}
