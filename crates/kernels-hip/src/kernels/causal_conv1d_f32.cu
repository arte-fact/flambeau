// causal_conv1d_f32 — depthwise causal 1D convolution, F32 in/out.
//
// The caller pre-concatenates `conv_kernel - 1` history tokens with the
// `n_new` incoming tokens into one contiguous buffer:
//   conv_input[n_total = (conv_kernel - 1) + n_new, conv_channels]
//
// For each output (t ∈ 0..n_new, c ∈ 0..conv_channels):
//   y[t, c] = Σ_{k=0..conv_kernel-1}  w[c, k] · conv_input[t + k, c]
//
// Depthwise — no cross-channel interaction. Channels are independent.
//
// Weight layout matches the GGUF on-disk order for `ssm_conv1d.weight`:
// outermost-first `[conv_channels, conv_kernel]` with the **kernel-tap axis
// innermost/contiguous** (offset = c * conv_kernel + k). This matches
// llama.cpp's `ggml-cuda/ssm-conv.cu::w[j] = w_block[tid * stride_w + j]`
// (tid = channel, j = tap) and avoids a load-time transpose that candle
// carries instead (`delta_net.rs:248 — .t()?.contiguous()?`).
//
// Launch shape:
//   gridDim  = { ceil(conv_channels / THREADS), n_new, 1 }
//   blockDim = { THREADS, 1, 1 }     // THREADS=256 fits gfx906 wave64 pairs

#include <hip/hip_runtime.h>

#ifndef CONV1D_THREADS
#define CONV1D_THREADS 256
#endif

extern "C" __global__ void flambeau_causal_conv1d_f32(
    const float* __restrict__ conv_input,   // [n_total, conv_channels]
    const float* __restrict__ weight,       // [conv_channels, conv_kernel]
    float* __restrict__ y,                  // [n_new, conv_channels]
    const int n_new,
    const int conv_channels,
    const int conv_kernel
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= conv_channels || t >= n_new) return;

    // Each thread owns one channel; load its `conv_kernel` weight taps
    // as a contiguous [c, 0..K) span — innermost contiguous matches GGUF.
    const float* w_c = weight + (size_t) c * conv_kernel;
    float acc = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < conv_kernel; ++k) {
        const float xv = conv_input[(t + k) * conv_channels + c];
        const float wv = w_c[k];
        acc += xv * wv;
    }
    y[t * conv_channels + c] = acc;
}
