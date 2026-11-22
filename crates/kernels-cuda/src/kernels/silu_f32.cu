// silu_f32 — pointwise SiLU / swish: y[i] = x[i] / (1 + exp(-x[i])).
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <cuda_runtime.h>

extern "C" __global__ void flambeau_silu_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xi = x[i];
    y[i] = xi / (1.0f + __expf(-xi));
}
