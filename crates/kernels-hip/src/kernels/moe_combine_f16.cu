// moe_combine_f16 — MoE expert output fan-in + residual add.
// For each token (row) and each hidden dim element:
// out[token, d] = residual[token, d]
// + Σ_{k=0..top_k} weight[token, k] * expert_out[token, k, d]
// Candle C1 pattern. Pure pointwise per (token, d) slot with a small
// inner sum over `top_k`.
// Launch:
// blockDim = { 256 }
// gridDim = { ceil(n_tokens * hidden / 256), 1, 1 }
// A thread owns one `(token, d)` slot. Expert-output buffer is shaped
// [n_tokens, top_k, hidden] F16 and the expert-weight buffer is
// [n_tokens, top_k] F32 (router softmax output).

#include <hip/hip_runtime.h>

typedef _Float16 fb_fp16_t;

extern "C" __global__ void flambeau_moe_combine_f16(
    const fb_fp16_t* __restrict__ expert_outs,   // [n_tokens, top_k, hidden]
    const float*     __restrict__ weights,        // [n_tokens, top_k]
    const fb_fp16_t* __restrict__ residual,       // [n_tokens, hidden]
    fb_fp16_t*       __restrict__ out,            // [n_tokens, hidden]
    const int n_tokens,
    const int top_k,
    const int hidden
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_tokens * hidden;
    if (idx >= total) return;

    const int token = idx / hidden;
    const int d     = idx - token * hidden;

    float acc = (float) residual[idx];
    #pragma unroll 4
    for (int k = 0; k < top_k; ++k) {
        const float w = weights[(size_t) token * top_k + k];
        const float e = (float)
            expert_outs[((size_t) token * top_k + k) * hidden + d];
        acc += w * e;
    }
    out[idx] = (fb_fp16_t) acc;
}
