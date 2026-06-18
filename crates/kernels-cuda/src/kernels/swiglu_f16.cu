// swiglu_f16 — SwiGLU activation `y[i] = silu(gate[i]) * up[i]`, F16 I/O.
// silu computed in F32 so large-negative gate doesn't underflow F16 exp range.
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <cuda_runtime.h>
#include "block_quant.cuh"

extern "C" __global__ void flambeau_swiglu_f16(
    const fb_fp16_t* __restrict__ gate,
    const fb_fp16_t* __restrict__ up,
    fb_fp16_t* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = (float) gate[i];
    const float u = (float) up[i];
    const float sig = 1.0f / (1.0f + __expf(-g));
    const float silu = g * sig;
    y[i] = (fb_fp16_t) (silu * u);
}
