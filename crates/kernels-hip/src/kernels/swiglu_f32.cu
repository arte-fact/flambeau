// swiglu_f32 — pointwise `y[i] = silu(a[i]) * b[i]`.
// GDN post-state-step gated output: `gated = silu(z) * out_normed` where
// both z and out_normed are F32. F16 variant already lives in
// swiglu_f16.cu; this is the F32 sibling for the recurrent path which
// keeps precision in F32 through the state update.
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <hip/hip_runtime.h>

extern "C" __global__ void flambeau_swiglu_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float av = a[i];
    const float silu = av / (1.0f + __expf(-av));
    y[i] = silu * b[i];
}
