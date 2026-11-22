// add_f32 — pointwise `y[i] = a[i] + b[i]` in F32.
// Launch: 1D, ceil(n/256) blocks × 256 threads. One element per thread.

#include <cuda_runtime.h>

extern "C" __global__ void flambeau_add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] + b[i];
}
