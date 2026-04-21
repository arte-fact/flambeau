// rope_f16 — rotary positional embedding, interleaved-pair layout, F16 in-place.
//
// For each (token, head, pair) triple:
//   angle = positions[token] * theta_base^(-2*pair/head_dim)
//   (x0, x1) = (x[2*pair], x[2*pair+1])
//   x[2*pair]   = x0*cos(angle) - x1*sin(angle)
//   x[2*pair+1] = x0*sin(angle) + x1*cos(angle)
//
// Layout matches Qwen3.6 (and candle's gemma4-style interleaved-pair):
// consecutive F16 values in the head_dim axis form one rotating pair. The
// kernel handles a single tensor — caller invokes it separately for Q and K.
//
// Launch shape:
//   blockDim  = { head_dim / 2 }    (64 threads at head_dim=128)
//   gridDim   = { n_tokens, n_heads, 1 }
//   shared    = 0
//
// `head_dim` must be even. For head_dim=128 we fit in a single wave64 block.

#include <hip/hip_runtime.h>

typedef _Float16 fb_fp16_t;

extern "C" __global__ void flambeau_rope_f16(
    fb_fp16_t* __restrict__ x,                 // [n_tokens, n_heads, head_dim]
    const int* __restrict__ positions,         // [n_tokens]
    const float theta_base,                    // e.g. 10000.0
    const int n_heads,
    const int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int pair_idx  = threadIdx.x;

    if (pair_idx * 2 >= head_dim) return;

    // `powf(a, 2.0 * pair / head_dim)` has enough precision for RoPE; the
    // equivalent `exp2f(log2f(theta_base) * ...)` would be fewer cycles but
    // complicates the unit test. Keep powf until we're perf-tuning.
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
