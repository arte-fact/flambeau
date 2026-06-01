// moe_combine_no_residual_f32 — F32-throughout sibling of
// `moe_combine_no_residual_f16`. Reads F32 per-expert down outputs and
// writes an F32 weighted sum. Used by the TP-sharded MoE forward when
// the row-parallel partial must stay F32 (head_dim=512 + Q8_0 cases:
// gemma4 26B-A4B-Q8_0 — F16 saturates on V-norm spikes that propagate
// into the down output).
//
// out[token, d] = Σ_{k=0..top_k} weight[token, k] * expert_out[token, k, d]

#include <hip/hip_runtime.h>

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
