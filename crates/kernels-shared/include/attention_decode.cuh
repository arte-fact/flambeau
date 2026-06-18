#pragma once

// GQA decode attention, F16 KV cache. One block per Q head, blockDim = head_dim
// (one thread per output element). Flash-attn-v2 online softmax: running
// (max, sum, out) so the [n_tokens] scores are never materialized. The only
// cross-lane op is the per-position Q·K dot — a block sum over head_dim threads
// (warp_reduce_sum + a runtime-sized cross-warp LDS rendezvous), neutral over
// the WARP_SIZE seam. The online-softmax update is replicated per thread (every
// thread reconstructs score_t), so it needs no warp ops.

#ifndef ATTN_DECODE_INFINITY
#define ATTN_DECODE_INFINITY __builtin_huge_valf()
#endif
#ifndef ATTN_MAX_HEAD_DIM
#define ATTN_MAX_HEAD_DIM 512
#endif

static __device__ void attention_decode_f16_body(
    const fb_fp16_t* __restrict__ q,
    const fb_fp16_t* __restrict__ k_cache,
    const fb_fp16_t* __restrict__ v_cache,
    fb_fp16_t* __restrict__ out,
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,
    const int n_tokens,
    const float scale,
    const int window_size,
    const int ring_depth
) {
    const int q_head = blockIdx.x;
    if (q_head >= n_heads_q) return;
    const int group = n_heads_q / n_heads_kv;
    const int kv_head = q_head / group;

    const int tid = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;

    __shared__ float q_shared[ATTN_MAX_HEAD_DIM];
    __shared__ float out_shared[ATTN_MAX_HEAD_DIM];
    __shared__ float score_parts[ATTN_MAX_HEAD_DIM / WARP_SIZE];
    if (tid < head_dim) {
        q_shared[tid] = (float) q[(size_t) q_head * head_dim + tid];
        out_shared[tid] = 0.0f;
    }
    __syncthreads();

    float running_max = -ATTN_DECODE_INFINITY;
    float running_sum = 0.0f;

    int t_start = 0;
    if (window_size > 0) {
        const int qpos = n_tokens - 1;
        t_start = qpos - window_size + 1;
        if (t_start < 0) t_start = 0;
    }

    for (int t = t_start; t < n_tokens; ++t) {
        const int t_phys = (ring_depth > 0) ? (t % ring_depth) : t;
        const size_t kv_row = ((size_t) t_phys * n_heads_kv + kv_head) * head_dim;

        float my_partial = (tid < head_dim)
            ? q_shared[tid] * (float) k_cache[kv_row + tid]
            : 0.0f;
        // Plain __shfl_xor butterfly (NOT the DPP warp_reduce_sum) to stay
        // bit-identical to the hand kernel's reduce on wave64; the WARP_SIZE
        // width keeps it correct on wave32.
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            my_partial += __shfl_xor(my_partial, off, WARP_SIZE);
        }
        if (lane == 0) {
            score_parts[warp] = my_partial;
        }
        __syncthreads();
        float score_t = 0.0f;
        for (int w = 0; w < n_warps; ++w) {
            score_t += score_parts[w];
        }
        score_t *= scale;

        const float new_max = fmaxf(running_max, score_t);
        const float scale_old = __expf(running_max - new_max);
        const float coeff_t = __expf(score_t - new_max);

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
        out[(size_t) q_head * head_dim + tid] = (fb_fp16_t) (out_shared[tid] * norm);
    }
}
