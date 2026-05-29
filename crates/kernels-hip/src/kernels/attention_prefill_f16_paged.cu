// attention_prefill_f16_paged — PagedAttention sibling of
// `attention_prefill_f16`. Same flash-attn-v2 online-softmax body;
// the only difference is the per-`t` K/V row address is resolved
// through the slot's row of the block table instead of a contiguous
// base.
//
// Layout:
//   Q: [n_q_tokens, n_heads_q, head_dim] F16 (same as the contiguous
//      sibling).
//   k_pool / v_pool: [n_pages, page_size, n_heads_kv, head_dim] F16
//      page pools. The slot's row of `block_table` indexes into
//      these.
//   block_table: pointer at the slot's row of the global
//      `[max_slots, max_pages_per_slot]` table — `[max_pages_per_slot]`
//      u32. For token `t` of the slot, the page is
//      `block_table[t / page_size]` and the in-page row is
//      `t % page_size`.
//   Out: [n_q_tokens, n_heads_q, head_dim] F16.
// Supported head_dim: {64, 128, 256, 512}. `page_size` must be a
// power of two so the divide/modulo compile to shifts and AND masks.
//
// Block layout (identical to attention_prefill_f16):
//   blockDim = { head_dim }
//   gridDim  = { n_q_tokens, n_heads_q, 1 }
//   shared   = q_shared[512] + out_shared[512] + score_parts[8]
//            = 4128 bytes
//
// VGPR delta vs the contiguous kernel: ~+3 (page_idx, p_off, page)
// computed once per outer t iteration. Block-table memory pattern is
// sequential u32 reads — 4 bytes per `page_size = 16` inner
// iterations, fully amortised in L1.
//
// Correctness oracle: with an identity-mapped block table
// (`block_table[p] = p`) and `n_pages = n_k_tokens / page_size`, the
// output is bit-identical to `attention_prefill_f16` running on the
// same K/V data. The paired parity test in
// `backend-hip/tests/attention_prefill_f16_paged.rs` enforces this.

#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

typedef _Float16 fb_fp16_t;

#define PREFILL_PAGED_MAX_HEAD_DIM 512
#define PREFILL_PAGED_MAX_WARPS (PREFILL_PAGED_MAX_HEAD_DIM / 64)

extern "C" __global__ void flambeau_attention_prefill_f16_paged(
    const fb_fp16_t* __restrict__ q,                 // [n_q_tokens, n_heads_q, head_dim]
    const fb_fp16_t* __restrict__ k_pool,            // [n_pages, page_size, n_heads_kv, head_dim]
    const fb_fp16_t* __restrict__ v_pool,            // [n_pages, page_size, n_heads_kv, head_dim]
    const unsigned int* __restrict__ block_table,    // [max_pages_per_slot] for THIS slot
    fb_fp16_t* __restrict__ out,                     // [n_q_tokens, n_heads_q, head_dim]
    const int n_q_tokens,
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,                              // 64, 128, 256, or 512
    const int n_k_tokens,
    const int q_offset,                              // global position of Q[0]
    const int page_size,
    const float scale,
    const int window_size                            // SWA radius, 0 = unbounded causal
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

    const int qpos_global = q_offset + q_token;
    int limit = qpos_global + 1;
    if (limit > n_k_tokens) {
        limit = n_k_tokens;
    }
    int t_start = 0;
    if (window_size > 0) {
        t_start = qpos_global - window_size + 1;
        if (t_start < 0) t_start = 0;
    }

    __shared__ float q_shared[PREFILL_PAGED_MAX_HEAD_DIM];
    if (tid < head_dim) {
        q_shared[tid] = (float) q[((size_t) q_token * n_heads_q + q_head) * head_dim + tid];
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    __shared__ float out_shared[PREFILL_PAGED_MAX_HEAD_DIM];
    if (tid < head_dim) {
        out_shared[tid] = 0.0f;
    }
    __shared__ float score_parts[PREFILL_PAGED_MAX_WARPS];
    __syncthreads();

    const int kv_width = n_heads_kv * head_dim;
    const int page_mask = page_size - 1;
    for (int t = t_start; t < limit; ++t) {
        const int page_idx_in_slot = t / page_size;
        const int p_off            = t & page_mask;
        const unsigned int page    = block_table[page_idx_in_slot];
        const size_t kv_row = ((size_t) page * page_size + p_off) * kv_width
                            + (size_t) kv_head * head_dim;

        // 1. Q · K[t, kv_head]
        float my_partial = 0.0f;
        if (tid < head_dim) {
            my_partial = q_shared[tid] * (float) k_pool[kv_row + tid];
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
        for (int w = 0; w < PREFILL_PAGED_MAX_WARPS; ++w) {
            if (w < n_warps) score_t += score_parts[w];
        }
        score_t *= scale;

        float new_max   = fmaxf(running_max, score_t);
        float scale_old = __expf(running_max - new_max);
        float coeff_t   = __expf(score_t - new_max);

        if (tid < head_dim) {
            const float v_t = (float) v_pool[kv_row + tid];
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
