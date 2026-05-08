// split_q_gate_f16 — split the interleaved (Q | gate) output of a gated-
// attention query projection into two contiguous tensors.
// Qwen3.5/3.6 / Qwen3-Next full-attention layers produce `attn_q` with shape
// [n_tokens, n_head, 2 * head_dim]
// where per-head the first `head_dim` lanes are the Q used in attention and
// the second `head_dim` lanes are the output gate applied after attention
// (`attn = silu(gate) * attn_pregate`). The attention kernel wants a
// contiguous `q[n_tokens, n_head, head_dim]` instead, so the split is done
// up-front into two separate buffers.
// Pointwise strided copy: one thread per output element.
// Launch:
// gridDim = { n_tokens, n_head, ceil(head_dim / THREADS) }
// blockDim = { THREADS, 1, 1 }

#include <hip/hip_runtime.h>

typedef _Float16 fb_fp16_t;

#ifndef SPLIT_THREADS
#define SPLIT_THREADS 128
#endif

extern "C" __global__ void flambeau_split_q_gate_f16(
    const fb_fp16_t* __restrict__ fused,   // [n_tokens, n_head, 2 * head_dim]
    fb_fp16_t* __restrict__ q,             // [n_tokens, n_head, head_dim]
    fb_fp16_t* __restrict__ gate,          // [n_tokens, n_head, head_dim]
    const int n_tokens,
    const int n_head,
    const int head_dim
) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int d     = blockIdx.z * SPLIT_THREADS + threadIdx.x;
    if (token >= n_tokens || head >= n_head || d >= head_dim) return;

    const int fused_base = (token * n_head + head) * (2 * head_dim);
    const int split_base = (token * n_head + head) * head_dim;
    q[split_base + d]    = fused[fused_base + d];
    gate[split_base + d] = fused[fused_base + head_dim + d];
}
