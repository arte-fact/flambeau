// add_f32 — pointwise `y[i] = a[i] + b[i]` in F32.
// A residual addition: keeps the residual stream in F32 across
// the MTP block (the previous F16 path lost ~1 pp of acceptance per
// residual to F16 round-off in the 5120-wide additions).
// Launch: 1D, ceil(n/256) blocks × 256 threads. One element per thread.

#include <hip/hip_runtime.h>

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
