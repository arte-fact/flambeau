// moe_combine_no_residual_f32 — F32 expert fan-in Σ_k w_k·e_k (TP-sharded MoE
// where the partial must stay F32). Pure pointwise per (token, d).
// Launch: 1D, ceil(n_tokens*hidden/256) blocks × 256 threads.

#include <cuda_runtime.h>

extern "C" __global__ void flambeau_moe_combine_no_residual_f32(
    const float* __restrict__ expert_outs,   // [n_tokens, top_k, hidden]
    const float* __restrict__ weights,       // [n_tokens, top_k]
    float*       __restrict__ out,           // [n_tokens, hidden]
    const int n_tokens,
    const int top_k,
    const int hidden
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_tokens * hidden;
    if (idx >= total) return;

    const int token = idx / hidden;
    const int d     = idx - token * hidden;

    float acc = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < top_k; ++k) {
        const float w = weights[(size_t) token * top_k + k];
        const float e = expert_outs[((size_t) token * top_k + k) * hidden + d];
        acc += w * e;
    }
    out[idx] = acc;
}
