// apply_softcap_f32 — elementwise final-logit softcap: y[i] = tanh(x[i]/cap)*cap.
// In-place safe (x and y may alias).
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <cuda_runtime.h>

extern "C" __global__ void flambeau_apply_softcap_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    const int n,
    const float cap
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float inv = 1.0f / cap;
    y[i] = tanhf(x[i] * inv) * cap;
}
