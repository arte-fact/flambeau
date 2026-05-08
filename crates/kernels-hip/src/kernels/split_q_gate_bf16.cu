// split_q_gate_bf16 — BF16 sibling of `split_q_gate_f16`. Splits the
// interleaved `(Q | gate)` output of a gated-attention query projection
// into two contiguous BF16 tensors.
// matches the F16 kernel byte-for-byte; only storage type
// differs. Pointwise strided copy (no arithmetic), so no F32 staging.

#include "block_quant.cuh"

#ifndef SPLIT_BF16_THREADS
#define SPLIT_BF16_THREADS 128
#endif

extern "C" __global__ void flambeau_split_q_gate_bf16(
    const fb_bf16_t* __restrict__ fused,   // [n_tokens, n_head, 2 * head_dim]
    fb_bf16_t* __restrict__ q,             // [n_tokens, n_head, head_dim]
    fb_bf16_t* __restrict__ gate,          // [n_tokens, n_head, head_dim]
    const int n_tokens,
    const int n_head,
    const int head_dim
) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int d     = blockIdx.z * SPLIT_BF16_THREADS + threadIdx.x;
    if (token >= n_tokens || head >= n_head || d >= head_dim) return;

    const int fused_base = (token * n_head + head) * (2 * head_dim);
    const int split_base = (token * n_head + head) * head_dim;
    q[split_base + d]    = fused[fused_base + d];
    gate[split_base + d] = fused[fused_base + head_dim + d];
}
