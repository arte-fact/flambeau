// rope_f16 — rotary positional embedding, interleaved-pair layout, F16 in-place.
// Pure per-element index math (no cross-lane reduce). Launch: gridDim =
// { n_tokens, n_heads, 1 }, blockDim = { head_dim / 2 }.

#include <cuda_runtime.h>
#include "block_quant.cuh"

extern "C" __global__ void flambeau_rope_f16(
    fb_fp16_t* __restrict__ x,                 // [n_tokens, n_heads, head_dim]
    const int* __restrict__ positions,         // [n_tokens]
    const float theta_base,
    const int n_heads,
    const int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int pair_idx  = threadIdx.x;
    if (pair_idx * 2 >= head_dim) return;

    const float exponent = 2.0f * (float) pair_idx / (float) head_dim;
    const float inv_freq = 1.0f / powf(theta_base, exponent);
    const float angle    = (float) positions[token_idx] * inv_freq;
    const float c = cosf(angle);
    const float s = sinf(angle);

    const int base = ((token_idx * n_heads + head_idx) * head_dim) + 2 * pair_idx;
    const float x0 = (float) x[base];
    const float x1 = (float) x[base + 1];
    x[base]     = (fb_fp16_t)(x0 * c - x1 * s);
    x[base + 1] = (fb_fp16_t)(x0 * s + x1 * c);
}
