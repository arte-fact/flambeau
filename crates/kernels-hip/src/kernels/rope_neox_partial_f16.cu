// rope_neox_partial_f16 — NeoX-style partial RoPE, F16 in-place.
// Qwen3.5/3.6/3-Next full-attention layers use a **partial** RoPE that
// rotates only the first `rotated_dims` of `head_dim`, with the pair layout
// split (not interleaved): pair indices are `(i, i + rotated_dims/2)`.
// This mirrors llama.cpp's `rope_multi` kernel (the non-vision, non-imrope
// path) for the text-only case where all MROPE sections point at the same
// position ID — mathematically equivalent to plain partial NeoX RoPE.
// For each (token, head, pair_i ∈ 0..rotated_dims/2):
// angle = positions[token] * theta_base^(-2*pair_i/rotated_dims)
// (x0, x1) = (x[pair_i], x[pair_i + rotated_dims/2])
// x[pair_i] = x0*cos(angle) - x1*sin(angle)
// x[pair_i + rotated_dims/2] = x0*sin(angle) + x1*cos(angle)
// Dimensions `rotated_dims..head_dim` pass through unchanged (no write).
// Caller invokes separately for Q and K. Gate (from qwen3.5/3.6 gated
// attention) is never rotated.
// Launch shape:
// blockDim = { rotated_dims / 2 } (32 threads at rotated_dims=64)
// gridDim = { n_tokens, n_heads, 1 }
// shared = 0

#include <hip/hip_runtime.h>

typedef _Float16 fb_fp16_t;

extern "C" __global__ void flambeau_rope_neox_partial_f16(
    fb_fp16_t* __restrict__ x,                 // [n_tokens, n_heads, head_dim]
    const int* __restrict__ positions,         // [n_tokens]
    const float theta_base,                    // usually 10000.0 or 1e7
    const int n_heads,
    const int head_dim,                        // total per-head width (e.g. 256)
    const int rotated_dims                     // partial RoPE width (e.g. 64)
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
    const float x0 = (float) x[lo];
    const float x1 = (float) x[hi];
    x[lo] = (fb_fp16_t)(x0 * c - x1 * s);
    x[hi] = (fb_fp16_t)(x0 * s + x1 * c);
}
