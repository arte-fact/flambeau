// silu_f32 — pointwise SiLU / swish: y[i] = x[i] / (1 + exp(-x[i])).
// GDN path uses this on the `silu(conv_out)` step before QKV split.
// 8192-element tensor on Qwen3.6; trivial compute.
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <hip/hip_runtime.h>

extern "C" __global__ void flambeau_silu_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xi = x[i];
    // __expf uses gfx906's v_exp_f32 SFU path.
    y[i] = xi / (1.0f + __expf(-xi));
}
