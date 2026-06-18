// causal_conv1d_f32 — depthwise causal 1D conv, F32. One thread per channel;
// y[t,c] = Σ_k w[c,k]·conv_input[t+k,c]. Channels independent.
// grid (ceil(conv_channels/256), n_new, 1), block 256.

#include <cuda_runtime.h>

#ifndef CONV1D_THREADS
#define CONV1D_THREADS 256
#endif

extern "C" __global__ void flambeau_causal_conv1d_f32(
    const float* __restrict__ conv_input,
    const float* __restrict__ weight,
    float* __restrict__ y,
    const int n_new,
    const int conv_channels,
    const int conv_kernel
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= conv_channels || t >= n_new) return;

    const float* w_c = weight + (size_t) c * conv_kernel;
    float acc = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < conv_kernel; ++k) {
        acc += conv_input[(t + k) * conv_channels + c] * w_c[k];
    }
    y[t * conv_channels + c] = acc;
}
