// attention_prefill_q8_kv — GQA prefill attention with Q8_0 KV cache.
//
// Same control flow + causal mask as `attention_prefill_f16`; K and V come
// from a `KvCache<Q8Contig>` and are dequantised on the fly during the
// score and V-accumulate (mirrors the chunk pass of
// `attention_decode_q8_kv_splitk`).
//
// This is the oracle path — used for any n_q_tokens. Faster
// flash-tile-Q8 variant is V2 follow-up; even this oracle replaces the
// per-token Q8 prefill fallback (~50 ms/layer/token at n_kv=200) with
// a single batched launch per layer.
//
// Layout:
//   Q:     [n_q_tokens, n_heads_q, head_dim]    — F16
//   K/V:   [n_k_tokens, n_heads_kv, head_dim/32] block_q8_0
//   Out:   [n_q_tokens, n_heads_q, head_dim]    — F16
//
// Supported head_dim: {64, 128, 256}.
// Launch: blockDim = { head_dim }, gridDim = { n_q_tokens, n_heads_q, 1 }.

#include "block_quant.cuh"
#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

#define PREFILL_Q8_MAX_HEAD_DIM 256
#define PREFILL_Q8_MAX_WARPS (PREFILL_Q8_MAX_HEAD_DIM / 64)

extern "C" __global__ void flambeau_attention_prefill_q8_kv(
    const fb_fp16_t* __restrict__ q,                    // [n_q_tokens, n_heads_q, head_dim]
    const flambeau_block_q8_0* __restrict__ k_cache,    // [n_k_tokens, n_heads_kv, head_dim/32]
    const flambeau_block_q8_0* __restrict__ v_cache,    // [n_k_tokens, n_heads_kv, head_dim/32]
    fb_fp16_t* __restrict__ out,                        // [n_q_tokens, n_heads_q, head_dim]
    const int n_q_tokens,
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,                                 // 64, 128 or 256
    const int n_k_tokens,
    const int q_offset,                                 // global position of Q[0]
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

    const int nb_per_row   = head_dim / 32;
    const int block_idx    = tid / 32;
    const int block_offset = tid & 31;

    // Causal mask: same as F16 oracle.
    int limit = q_offset + q_token + 1;
    if (limit > n_k_tokens) limit = n_k_tokens;

    __shared__ float q_shared[PREFILL_Q8_MAX_HEAD_DIM];
    if (tid < head_dim) {
        q_shared[tid] = (float) q[((size_t) q_token * n_heads_q + q_head) * head_dim + tid];
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    __shared__ float out_shared[PREFILL_Q8_MAX_HEAD_DIM];
    if (tid < head_dim) out_shared[tid] = 0.0f;
    __shared__ float score_parts[PREFILL_Q8_MAX_WARPS];
    __syncthreads();

    for (int t = 0; t < limit; ++t) {
        const size_t kv_row_blocks = ((size_t) t * n_heads_kv + kv_head) * nb_per_row;

        float my_partial = 0.0f;
        if (tid < head_dim) {
            const flambeau_block_q8_0* k_block =
                k_cache + kv_row_blocks + block_idx;
            const float k_d = (float) k_block->d;
            const int   k_q = (int)   k_block->qs[block_offset];
            my_partial = q_shared[tid] * (k_d * (float) k_q);
        }
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            my_partial += __shfl_xor(my_partial, off, 64);
        }
        if (lane == 0) score_parts[warp] = my_partial;
        __syncthreads();
        float score_t = 0.0f;
        #pragma unroll
        for (int w = 0; w < PREFILL_Q8_MAX_WARPS; ++w) {
            if (w < n_warps) score_t += score_parts[w];
        }
        score_t *= scale;

        float new_max   = fmaxf(running_max, score_t);
        float scale_old = __expf(running_max - new_max);
        float coeff_t   = __expf(score_t - new_max);

        if (tid < head_dim) {
            const flambeau_block_q8_0* v_block =
                v_cache + kv_row_blocks + block_idx;
            const float v_d = (float) v_block->d;
            const int   v_q = (int)   v_block->qs[block_offset];
            const float v_v = v_d * (float) v_q;
            out_shared[tid] = out_shared[tid] * scale_old + coeff_t * v_v;
        }
        running_sum = running_sum * scale_old + coeff_t;
        running_max = new_max;

        __syncthreads();
    }

    if (tid < head_dim) {
        const float norm = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
        out[((size_t) q_token * n_heads_q + q_head) * head_dim + tid] =
            (fb_fp16_t) (out_shared[tid] * norm);
    }
}
