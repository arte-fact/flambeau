// swiglu_f32 — pointwise y[i] = silu(a[i]) * b[i], pure F32 (GDN gated output).
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <cuda_runtime.h>

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
