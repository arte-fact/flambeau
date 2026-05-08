// attention_prefill_f16 — GQA prefill attention, F16 KV cache.
// Generalises the decode kernel to multi-token Q. Each (q_token, q_head)
// pair gets its own block; the online-softmax recurrence is identical to
// decode but the context is causally truncated to
// `min(q_offset + q_token + 1, n_k_tokens)`.
// Layout:
// Q: [n_q_tokens, n_heads_q, head_dim]
// K/V: [n_k_tokens, n_heads_kv, head_dim] (same layout KvCache<F16Contig> uses)
// Out: [n_q_tokens, n_heads_q, head_dim]
// Supported head_dim: {64, 128, 256} (matching attention_decode_f16).
// Caller launches with `block = head_dim` threads; kernel computes
// `n_warps = blockDim.x / 64` at runtime and sums `score_parts[0..n_warps]`.
// Launch shape:
// blockDim = { head_dim }
// gridDim = { n_q_tokens, n_heads_q, 1 }
// This is the correctness oracle. First-class perf variant is a tiled
// flash-attn-v2 (block-over-Q + block-over-K with cooperative LDS tiles
// and K-transposed layout in the KvCache); it ships as a separate impl_id
// behind the same cert once perf work lands.

#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

typedef _Float16 fb_fp16_t;

#define PREFILL_MAX_HEAD_DIM 256
#define PREFILL_MAX_WARPS (PREFILL_MAX_HEAD_DIM / 64)

extern "C" __global__ void flambeau_attention_prefill_f16(
    const fb_fp16_t* __restrict__ q,           // [n_q_tokens, n_heads_q, head_dim]
    const fb_fp16_t* __restrict__ k_cache,     // [n_k_tokens, n_heads_kv, head_dim]
    const fb_fp16_t* __restrict__ v_cache,     // [n_k_tokens, n_heads_kv, head_dim]
    fb_fp16_t* __restrict__ out,               // [n_q_tokens, n_heads_q, head_dim]
    const int n_q_tokens,
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,                        // must be 64, 128 or 256
    const int n_k_tokens,
    const int q_offset,                        // global position of Q[0]
    const float scale
) {
    const int q_token = blockIdx.x;
    const int q_head  = blockIdx.y;
    if (q_token >= n_q_tokens || q_head >= n_heads_q) return;
    const int group   = n_heads_q / n_heads_kv;
    const int kv_head = q_head / group;

    const int tid     = threadIdx.x;
    const int warp    = tid >> 6;
    const int lane    = tid & 63;
    const int n_warps = blockDim.x >> 6;

    // Causal mask: Q token at global position (q_offset + q_token) can
    // attend to K tokens 0..=(q_offset + q_token). Clamp to the cache.
    int limit = q_offset + q_token + 1;
    if (limit > n_k_tokens) {
        limit = n_k_tokens;
    }

    // --- Load Q for this (q_token, q_head) into LDS ---
    __shared__ float q_shared[PREFILL_MAX_HEAD_DIM];
    if (tid < head_dim) {
        q_shared[tid] = (float) q[((size_t) q_token * n_heads_q + q_head) * head_dim + tid];
    }

    // --- Online softmax running stats ---
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    __shared__ float out_shared[PREFILL_MAX_HEAD_DIM];
    if (tid < head_dim) {
        out_shared[tid] = 0.0f;
    }
    __shared__ float score_parts[PREFILL_MAX_WARPS];
    __syncthreads();

    for (int t = 0; t < limit; ++t) {
        const size_t kv_row = ((size_t) t * n_heads_kv + kv_head) * head_dim;

        // 1. Q · K[t, kv_head]
        float my_partial = 0.0f;
        if (tid < head_dim) {
            my_partial = q_shared[tid] * (float) k_cache[kv_row + tid];
        }
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            my_partial += __shfl_xor(my_partial, off, 64);
        }
        if (lane == 0) {
            score_parts[warp] = my_partial;
        }
        __syncthreads();
        float score_t = 0.0f;
        #pragma unroll
        for (int w = 0; w < PREFILL_MAX_WARPS; ++w) {
            if (w < n_warps) score_t += score_parts[w];
        }
        score_t *= scale;

        // 2. Online softmax update.
        float new_max   = fmaxf(running_max, score_t);
        float scale_old = __expf(running_max - new_max);
        float coeff_t   = __expf(score_t - new_max);

        // 3. running_out += coeff_t * V[t, kv_head]
        if (tid < head_dim) {
            const float v_t = (float) v_cache[kv_row + tid];
            out_shared[tid] = out_shared[tid] * scale_old + coeff_t * v_t;
        }
        running_sum = running_sum * scale_old + coeff_t;
        running_max = new_max;

        __syncthreads();
    }

    if (tid < head_dim) {
        const float norm = 1.0f / running_sum;
        out[((size_t) q_token * n_heads_q + q_head) * head_dim + tid] =
            (fb_fp16_t) (out_shared[tid] * norm);
    }
}
