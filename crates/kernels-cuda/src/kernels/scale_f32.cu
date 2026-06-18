// scale_f32 — pointwise `y[i] = x[i] * scale`. CUDA mirror of the HIP kernel.
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <cuda_runtime.h>

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
