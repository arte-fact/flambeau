// scale_f32 — pointwise `y[i] = x[i] * scale`.
//
// GDN path scales Q by `1 / sqrt(head_k_dim)` before the recurrent state
// step (matches candle's `q * scale` before `delta_net_step_vectorized`).
// Separate kernel rather than fusing into l2_norm_f32 to keep that
// kernel's cert untouched.
//
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <hip/hip_runtime.h>

extern "C" __global__ void flambeau_scale_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    const int n,
    const float scale
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] * scale;
}
