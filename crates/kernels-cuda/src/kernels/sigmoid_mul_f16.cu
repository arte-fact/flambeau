// sigmoid_mul_f16 — pointwise `y[i] = sigmoid(gate[i]) * x[i]`, F16 in/out.
// Full-attention gating (qwen35moe): attn_out * sigmoid(gate). Numerically
// stable sigmoid (branch on sign of gate).
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <cuda_runtime.h>
#include "block_quant.cuh"

extern "C" __global__ void flambeau_sigmoid_mul_f16(
    const fb_fp16_t* __restrict__ gate,
    const fb_fp16_t* __restrict__ x,
    fb_fp16_t* __restrict__ y,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = (float) gate[idx];
    float sig;
    if (g >= 0.0f) {
        sig = 1.0f / (1.0f + __expf(-g));
    } else {
        const float z = __expf(g);
        sig = z / (1.0f + z);
    }
    const float xv = (float) x[idx];
    y[idx] = (fb_fp16_t) (sig * xv);
}
