// rope_neox_partial_bf16 — BF16 sibling of `rope_neox_partial_f16`.
//
// MTP-4-C-5: same partial-NeoX RoPE math (rotates the first
// `rotated_dims` of `head_dim`, split-pair layout
// `(i, i + rotated_dims/2)`). BF16 storage in-place; F32 internal
// trig + multiply.
//
// Launch shape:
//   blockDim = { rotated_dims / 2 }
//   gridDim  = { n_tokens, n_heads, 1 }

#include "block_quant.cuh"

extern "C" __global__ void flambeau_rope_neox_partial_bf16(
    fb_bf16_t* __restrict__ x,                 // [n_tokens, n_heads, head_dim]
    const int* __restrict__ positions,         // [n_tokens]
    const float theta_base,
    const int n_heads,
    const int head_dim,
    const int rotated_dims
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int pair_i    = threadIdx.x;

    if (pair_i * 2 >= rotated_dims) return;

    const float exponent = 2.0f * (float) pair_i / (float) rotated_dims;
    const float inv_freq = 1.0f / powf(theta_base, exponent);
    const float angle    = (float) positions[token_idx] * inv_freq;
    const float c = cosf(angle);
    const float s = sinf(angle);

    const int head_base = (token_idx * n_heads + head_idx) * head_dim;
    const int half      = rotated_dims >> 1;
    const int lo = head_base + pair_i;
    const int hi = head_base + pair_i + half;
    const float x0 = fb_bf16_to_f32(x[lo]);
    const float x1 = fb_bf16_to_f32(x[hi]);
    x[lo] = fb_f32_to_bf16(x0 * c - x1 * s);
    x[hi] = fb_f32_to_bf16(x0 * s + x1 * c);
}
