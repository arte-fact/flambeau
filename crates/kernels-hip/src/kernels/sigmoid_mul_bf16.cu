// sigmoid_mul_bf16 — pointwise `y[i] = sigmoid(gate[i]) * x[i]`, BF16 in/out.
// BF16 sibling of `sigmoid_mul_f16`. Same numerically-stable
// sigmoid formula (branch on sign so the exp argument stays ≤ 0). Used
// for the MTP attention output gate `attn_gated = attn_out * sigmoid(gate)`
// (NOT silu — see post-mortem).
// Launch: blockDim = 256, gridDim = ceil(n / 256).

#include "block_quant.cuh"

extern "C" __global__ void flambeau_sigmoid_mul_bf16(
    const fb_bf16_t* __restrict__ gate,
    const fb_bf16_t* __restrict__ x,
    fb_bf16_t* __restrict__ y,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = fb_bf16_to_f32(gate[idx]);
    float sig;
    if (g >= 0.0f) {
        sig = 1.0f / (1.0f + __expf(-g));
    } else {
        const float z = __expf(g);
        sig = z / (1.0f + z);
    }
    const float xv = fb_bf16_to_f32(x[idx]);
    y[idx] = fb_f32_to_bf16(sig * xv);
}
