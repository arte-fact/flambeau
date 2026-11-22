#pragma once

// F16 attention variants (prefill / batched-slots / paged) — all the GQA online
// -softmax sibling of attention_decode_f16. The only cross-lane op is the
// per-position Q·K dot, reduced via a plain __shfl_xor butterfly over the
// WARP_SIZE seam (NOT the DPP warp_reduce_sum) for bit-identity with the hand
// kernels; the online softmax + out accumulation are per-thread (replicated).
#ifndef ATTN_F16_INF
#define ATTN_F16_INF __builtin_huge_valf()
#endif
#define ATTN_F16_MAX_HEAD_DIM 512

// Block sum of the per-lane Q·K partial, broadcast to all threads. score_parts
// is caller LDS of at least ATTN_F16_MAX_HEAD_DIM/WARP_SIZE floats.
static __device__ __forceinline__ float attn_qk_reduce(float my_partial, float* score_parts) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        my_partial += __shfl_xor(my_partial, off, WARP_SIZE);
    }
    if (lane == 0) score_parts[warp] = my_partial;
    __syncthreads();
    float s = 0.0f;
    for (int w = 0; w < n_warps; ++w) s += score_parts[w];
    return s;
}

static __device__ void attention_prefill_f16_body(
    const fb_fp16_t* __restrict__ q,
    const fb_fp16_t* __restrict__ k_cache,
    const fb_fp16_t* __restrict__ v_cache,
    fb_fp16_t* __restrict__ out,
    int n_q_tokens, int n_heads_q, int n_heads_kv, int head_dim,
    int n_k_tokens, int q_offset, float scale, int window_size, int ring_depth)
{
    const int q_token = blockIdx.x;
    const int q_head = blockIdx.y;
    if (q_token >= n_q_tokens || q_head >= n_heads_q) return;
    const int kv_head = q_head / (n_heads_q / n_heads_kv);
    const int tid = threadIdx.x;

    const int qpos = q_offset + q_token;
    int limit = qpos + 1;
    if (limit > n_k_tokens) limit = n_k_tokens;
    int t_start = 0;
    if (window_size > 0) { t_start = qpos - window_size + 1; if (t_start < 0) t_start = 0; }

    __shared__ float q_shared[ATTN_F16_MAX_HEAD_DIM];
    __shared__ float out_shared[ATTN_F16_MAX_HEAD_DIM];
    __shared__ float score_parts[ATTN_F16_MAX_HEAD_DIM / WARP_SIZE];
    if (tid < head_dim) {
        q_shared[tid] = (float) q[((size_t) q_token * n_heads_q + q_head) * head_dim + tid];
        out_shared[tid] = 0.0f;
    }
    __syncthreads();

    float running_max = -ATTN_F16_INF;
    float running_sum = 0.0f;
    for (int t = t_start; t < limit; ++t) {
        const int t_phys = (ring_depth > 0) ? (t % ring_depth) : t;
        const size_t kv_row = ((size_t) t_phys * n_heads_kv + kv_head) * head_dim;
        float mp = (tid < head_dim) ? q_shared[tid] * (float) k_cache[kv_row + tid] : 0.0f;
        const float score_t = attn_qk_reduce(mp, score_parts) * scale;
        const float new_max = fmaxf(running_max, score_t);
        const float scale_old = __expf(running_max - new_max);
        const float coeff_t = __expf(score_t - new_max);
        if (tid < head_dim) {
            out_shared[tid] = out_shared[tid] * scale_old + coeff_t * (float) v_cache[kv_row + tid];
        }
        running_sum = running_sum * scale_old + coeff_t;
        running_max = new_max;
        __syncthreads();
    }
    if (tid < head_dim) {
        const float norm = 1.0f / running_sum;
        out[((size_t) q_token * n_heads_q + q_head) * head_dim + tid] = (fb_fp16_t) (out_shared[tid] * norm);
    }
}

static __device__ void attention_decode_f16_batched_body(
    const fb_fp16_t* __restrict__ q_batched,
    const uint64_t* __restrict__ k_cache_ptrs,
    const uint64_t* __restrict__ v_cache_ptrs,
    fb_fp16_t* __restrict__ out_batched,
    const int* __restrict__ n_tokens_kv,
    int n_heads_q, int n_heads_kv, int head_dim, int n_slots,
    float scale, int window_size, int ring_depth)
{
    const int q_head = blockIdx.x;
    const int slot = blockIdx.y;
    if (q_head >= n_heads_q || slot >= n_slots) return;
    const int kv_head = q_head / (n_heads_q / n_heads_kv);
    const int tid = threadIdx.x;

    const fb_fp16_t* k_cache = (const fb_fp16_t*) k_cache_ptrs[slot];
    const fb_fp16_t* v_cache = (const fb_fp16_t*) v_cache_ptrs[slot];
    const int n_tokens = n_tokens_kv[slot];
    const fb_fp16_t* q_row = q_batched + ((size_t) slot * n_heads_q + q_head) * (size_t) head_dim;
    fb_fp16_t* out_row = out_batched + ((size_t) slot * n_heads_q + q_head) * (size_t) head_dim;

    __shared__ float q_shared[ATTN_F16_MAX_HEAD_DIM];
    __shared__ float out_shared[ATTN_F16_MAX_HEAD_DIM];
    __shared__ float score_parts[ATTN_F16_MAX_HEAD_DIM / WARP_SIZE];
    if (tid < head_dim) { q_shared[tid] = (float) q_row[tid]; out_shared[tid] = 0.0f; }
    __syncthreads();

    int t_start = 0;
    if (window_size > 0) { t_start = (n_tokens - 1) - window_size + 1; if (t_start < 0) t_start = 0; }
    float running_max = -ATTN_F16_INF;
    float running_sum = 0.0f;
    for (int t = t_start; t < n_tokens; ++t) {
        const int t_phys = (ring_depth > 0) ? (t % ring_depth) : t;
        const size_t kv_row = ((size_t) t_phys * n_heads_kv + kv_head) * head_dim;
        float mp = (tid < head_dim) ? q_shared[tid] * (float) k_cache[kv_row + tid] : 0.0f;
        const float score_t = attn_qk_reduce(mp, score_parts) * scale;
        const float new_max = fmaxf(running_max, score_t);
        const float scale_old = __expf(running_max - new_max);
        const float coeff_t = __expf(score_t - new_max);
        if (tid < head_dim) {
            out_shared[tid] = out_shared[tid] * scale_old + coeff_t * (float) v_cache[kv_row + tid];
        }
        running_sum = running_sum * scale_old + coeff_t;
        running_max = new_max;
        __syncthreads();
    }
    if (tid < head_dim) {
        const float norm = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        out_row[tid] = (fb_fp16_t) (out_shared[tid] * norm);
    }
}

static __device__ void attention_decode_f16_paged_body(
    const fb_fp16_t* __restrict__ q_batched,
    const fb_fp16_t* __restrict__ k_pool,
    const fb_fp16_t* __restrict__ v_pool,
    const unsigned int* __restrict__ block_tables,
    fb_fp16_t* __restrict__ out_batched,
    const int* __restrict__ n_tokens_kv,
    int n_heads_q, int n_heads_kv, int head_dim, int n_slots,
    int page_size, int max_pages_per_slot, float scale)
{
    const int q_head = blockIdx.x;
    const int slot = blockIdx.y;
    if (q_head >= n_heads_q || slot >= n_slots) return;
    const int kv_head = q_head / (n_heads_q / n_heads_kv);
    const int tid = threadIdx.x;

    const int n_tokens = n_tokens_kv[slot];
    const int kv_width = n_heads_kv * head_dim;
    const unsigned int* slot_table = block_tables + (size_t) slot * max_pages_per_slot;
    const fb_fp16_t* q_row = q_batched + ((size_t) slot * n_heads_q + q_head) * (size_t) head_dim;
    fb_fp16_t* out_row = out_batched + ((size_t) slot * n_heads_q + q_head) * (size_t) head_dim;

    __shared__ float q_shared[ATTN_F16_MAX_HEAD_DIM];
    __shared__ float out_shared[ATTN_F16_MAX_HEAD_DIM];
    __shared__ float score_parts[ATTN_F16_MAX_HEAD_DIM / WARP_SIZE];
    if (tid < head_dim) { q_shared[tid] = (float) q_row[tid]; out_shared[tid] = 0.0f; }
    __syncthreads();

    const int page_mask = page_size - 1;
    float running_max = -ATTN_F16_INF;
    float running_sum = 0.0f;
    for (int t = 0; t < n_tokens; ++t) {
        const unsigned int page = slot_table[t / page_size];
        const size_t kv_row = ((size_t) page * page_size + (t & page_mask)) * kv_width + (size_t) kv_head * head_dim;
        float mp = (tid < head_dim) ? q_shared[tid] * (float) k_pool[kv_row + tid] : 0.0f;
        const float score_t = attn_qk_reduce(mp, score_parts) * scale;
        const float new_max = fmaxf(running_max, score_t);
        const float scale_old = __expf(running_max - new_max);
        const float coeff_t = __expf(score_t - new_max);
        if (tid < head_dim) {
            out_shared[tid] = out_shared[tid] * scale_old + coeff_t * (float) v_pool[kv_row + tid];
        }
        running_sum = running_sum * scale_old + coeff_t;
        running_max = new_max;
        __syncthreads();
    }
    if (tid < head_dim) {
        const float norm = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        out_row[tid] = (fb_fp16_t) (out_shared[tid] * norm);
    }
}

static __device__ void attention_prefill_f16_paged_body(
    const fb_fp16_t* __restrict__ q,
    const fb_fp16_t* __restrict__ k_pool,
    const fb_fp16_t* __restrict__ v_pool,
    const unsigned int* __restrict__ block_table,
    fb_fp16_t* __restrict__ out,
    int n_q_tokens, int n_heads_q, int n_heads_kv, int head_dim,
    int n_k_tokens, int q_offset, int page_size, float scale, int window_size)
{
    const int q_token = blockIdx.x;
    const int q_head = blockIdx.y;
    if (q_token >= n_q_tokens || q_head >= n_heads_q) return;
    const int kv_head = q_head / (n_heads_q / n_heads_kv);
    const int tid = threadIdx.x;
    const int kv_width = n_heads_kv * head_dim;

    const int qpos = q_offset + q_token;
    int limit = qpos + 1;
    if (limit > n_k_tokens) limit = n_k_tokens;
    int t_start = 0;
    if (window_size > 0) { t_start = qpos - window_size + 1; if (t_start < 0) t_start = 0; }

    __shared__ float q_shared[ATTN_F16_MAX_HEAD_DIM];
    __shared__ float out_shared[ATTN_F16_MAX_HEAD_DIM];
    __shared__ float score_parts[ATTN_F16_MAX_HEAD_DIM / WARP_SIZE];
    if (tid < head_dim) {
        q_shared[tid] = (float) q[((size_t) q_token * n_heads_q + q_head) * head_dim + tid];
        out_shared[tid] = 0.0f;
    }
    __syncthreads();

    const int page_mask = page_size - 1;
    float running_max = -ATTN_F16_INF;
    float running_sum = 0.0f;
    for (int t = t_start; t < limit; ++t) {
        const unsigned int page = block_table[t / page_size];
        const size_t kv_row = ((size_t) page * page_size + (t & page_mask)) * kv_width + (size_t) kv_head * head_dim;
        float mp = (tid < head_dim) ? q_shared[tid] * (float) k_pool[kv_row + tid] : 0.0f;
        const float score_t = attn_qk_reduce(mp, score_parts) * scale;
        const float new_max = fmaxf(running_max, score_t);
        const float scale_old = __expf(running_max - new_max);
        const float coeff_t = __expf(score_t - new_max);
        if (tid < head_dim) {
            out_shared[tid] = out_shared[tid] * scale_old + coeff_t * (float) v_pool[kv_row + tid];
        }
        running_sum = running_sum * scale_old + coeff_t;
        running_max = new_max;
        __syncthreads();
    }
    if (tid < head_dim) {
        const float norm = 1.0f / running_sum;
        out[((size_t) q_token * n_heads_q + q_head) * head_dim + tid] = (fb_fp16_t) (out_shared[tid] * norm);
    }
}
