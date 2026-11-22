// split_q_gate_f16 — split interleaved (Q | gate) attention-query output into
// two contiguous tensors. Pure strided copy, one thread per output element.
//
// Launch: gridDim = { n_tokens, n_head, ceil(head_dim/128) }, blockDim = { 128 }.

#include <cuda_runtime.h>
#include "block_quant.cuh"

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
