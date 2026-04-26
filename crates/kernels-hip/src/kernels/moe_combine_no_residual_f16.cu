// moe_combine_no_residual_f16 — TP-4b-i2 variant of `moe_combine_f16`
// without the input residual.
//
//   out[token, d] = Σ_{k=0..top_k} weight[token, k] * expert_out[token, k, d]
//
// Used by the TP-sharded MoE forward, where the residual stream is
// folded later by the AllReduce-residual kernel rather than by the
// combine. Same launch shape as `moe_combine_f16`.

#include <hip/hip_runtime.h>

typedef _Float16 fb_fp16_t;

extern "C" __global__ void flambeau_moe_combine_no_residual_f16(
    const fb_fp16_t* __restrict__ expert_outs,   // [n_tokens, top_k, hidden]
    const float*     __restrict__ weights,        // [n_tokens, top_k]
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

    float acc = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < top_k; ++k) {
        const float w = weights[(size_t) token * top_k + k];
        const float e = (float)
            expert_outs[((size_t) token * top_k + k) * hidden + d];
        acc += w * e;
    }
    out[idx] = (fb_fp16_t) acc;
}
