// attention_decode_q8_kv — GQA decode attention with Q8_0 KV cache.
//
// Same online-softmax shape as `attention_decode_f16`, but K and V come
// from a `KvCache<Q8Contig>`. Each (token, kv_head) row is `head_dim / 32`
// `block_q8_0`s; we dequantise on the fly during the dot product and the
// V-accumulate step.
//
// Qwen3.6 Q4_K_M with Q8 KV: 2× HBM saving on the KV-fetch side vs
// F16, with quant noise bounded (V1.6.6 correctness cert + V1.7 quality
// cert gate at delta-ppl ≤ 0.5% per the roadmap).
//
// Layout:
//   k_cache / v_cache: [n_tokens, n_heads_kv, head_dim/32] × block_q8_0
//   so at head_dim=128 there are 4 blocks per (token, kv_head) row;
//   at head_dim=256 there are 8.
//
// Supported head_dim: {64, 128, 256}, matching the F16 decode variant.
// Thread mapping (block = head_dim):
//   tid ∈ [0, head_dim)
//   block_idx    = tid / 32         — which Q8_0 block within the row
//   block_offset = tid & 31         — which int8 inside that block

#include "block_quant.cuh"
#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

#define ATTN_Q8_MAX_HEAD_DIM 256
#define ATTN_Q8_MAX_WARPS (ATTN_Q8_MAX_HEAD_DIM / 64)

extern "C" __global__ void flambeau_attention_decode_q8_kv(
    const fb_fp16_t* __restrict__ q,                    // [n_heads_q, head_dim]
    const flambeau_block_q8_0* __restrict__ k_cache,    // [n_tokens, n_heads_kv, head_dim/32]
    const flambeau_block_q8_0* __restrict__ v_cache,    // [n_tokens, n_heads_kv, head_dim/32]
    fb_fp16_t* __restrict__ out,                        // [n_heads_q, head_dim]
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,                                 // must be 64, 128 or 256
    const int n_tokens,
    const float scale
) {
    const int q_head  = blockIdx.x;
    if (q_head >= n_heads_q) return;
    const int group   = n_heads_q / n_heads_kv;
    const int kv_head = q_head / group;

    const int tid     = threadIdx.x;
    const int warp    = tid >> 6;
    const int lane    = tid & 63;
    const int n_warps = blockDim.x >> 6;

    const int nb_per_row = head_dim / 32;
    const int block_idx    = tid / 32;
    const int block_offset = tid & 31;

    // --- Load Q (F16) into LDS ---
    __shared__ float q_shared[ATTN_Q8_MAX_HEAD_DIM];
    if (tid < head_dim) {
        q_shared[tid] = (float) q[(size_t) q_head * head_dim + tid];
    }

    // --- Online softmax stats ---
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    __shared__ float out_shared[ATTN_Q8_MAX_HEAD_DIM];
    if (tid < head_dim) {
        out_shared[tid] = 0.0f;
    }
    __shared__ float score_parts[ATTN_Q8_MAX_WARPS];
    __syncthreads();

    for (int t = 0; t < n_tokens; ++t) {
        // (t, kv_head) row base for K and V, in block_q8_0 units.
        const size_t kv_row_blocks = ((size_t) t * n_heads_kv + kv_head) * nb_per_row;

        // --- Q · K[t, kv_head] ---
        //
        // Each thread reads one int8 from its block + the block's scale.
        // `d_k * qs` gives the dequantised F32 for this thread's element.
        float my_partial = 0.0f;
        if (tid < head_dim) {
            const flambeau_block_q8_0* k_block =
                k_cache + kv_row_blocks + block_idx;
            const float k_d = (float) k_block->d;
            const int   k_q = (int)   k_block->qs[block_offset];
            const float k_v = k_d * (float) k_q;
            my_partial = q_shared[tid] * k_v;
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
        for (int w = 0; w < ATTN_Q8_MAX_WARPS; ++w) {
            if (w < n_warps) score_t += score_parts[w];
        }
        score_t *= scale;

        // --- Online softmax update ---
        float new_max   = fmaxf(running_max, score_t);
        float scale_old = __expf(running_max - new_max);
        float coeff_t   = __expf(score_t - new_max);

        // --- V-accum: dequant V on the fly ---
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
        const float norm = 1.0f / running_sum;
        out[(size_t) q_head * head_dim + tid] =
            (fb_fp16_t) (out_shared[tid] * norm);
    }
}
