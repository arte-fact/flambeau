// sigmoid_mul_f16 — pointwise `y[i] = sigmoid(gate[i]) * x[i]`, F16 in/out.
// Qwen3.5/3.6 full-attention gating: after `attention_decode_f16` produces
// `attn_out[n_heads, head_dim]` and the Q-proj split extracts `gate` of
// the same shape, the gated output is `attn_out * sigmoid(gate)` — a plain
// sigmoid (logistic) element-wise multiply, **not** SiLU's `gate * sigmoid(gate)`.
// This matches llama.cpp's qwen35moe.cpp graph:
// attn_pregate = fattn output
// gate_sigmoid = sigmoid(Qcur_full view → gate)
// attn_gated = attn_pregate * gate_sigmoid
// Formerly this path called `swiglu_f16(gate, attn, out)`, which computes
// `silu(gate) * attn = gate * sigmoid(gate) * attn` — off by an extra factor
// of `gate` per element. That was first-diverging-layer bug: our
// full-attn layer 3 output diverged from llama.cpp by 10-25× per element,
// cascading through the remaining 37 layers into totally wrong logits.
// Launch:
// blockDim = { 256 }
// gridDim = { ceil(n / 256), 1, 1 }

#include <hip/hip_runtime.h>

typedef _Float16 fb_fp16_t;

extern "C" __global__ void flambeau_sigmoid_mul_f16(
    const fb_fp16_t* __restrict__ gate,
    const fb_fp16_t* __restrict__ x,
    fb_fp16_t* __restrict__ y,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = (float) gate[idx];
    // Numerically stable sigmoid: for g >= 0 use 1/(1+exp(-g)); else exp(g)/(1+exp(g)).
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
