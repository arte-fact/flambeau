// moe_combine_f16 — MoE expert output fan-in + residual add. One thread per
// (token, d), inner sum over top_k. Pure pointwise, no cross-lane reduce.
// Launch: 1D, ceil(n_tokens*hidden/256) blocks × 256 threads.

#include <cuda_runtime.h>
#include "block_quant.cuh"

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
        const float e = (float) expert_outs[((size_t) token * top_k + k) * hidden + d];
        acc += w * e;
    }
    out[idx] = (fb_fp16_t) acc;
}
