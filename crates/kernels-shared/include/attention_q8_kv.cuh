#pragma once

// GQA attention with a Q8_0 KV cache. block.x = head_dim/4 threads — each owns 4
// elements (one int32 of packed int8). Q is quantized to Q8_0 in LDS once; the
// score is integer dp4a(K, Q) per thread, scaled by K.d·Q.d, then summed over
// the block. The block sum rides block_tree_sum (LDS tree, wave-agnostic — works
// for the sub-warp blocks at head_dim=64) instead of the wave64-hardcoded
// multi-block shuffle, so it is correct on wave64 and wave32. dp4a comes from
// the arch seam. V dequants per element (F32). Online softmax.
#ifndef ATTN_Q8_INF
#define ATTN_Q8_INF __builtin_huge_valf()
#endif
#define ATTN_Q8_MAX_HEAD_DIM 512

// Per-thread Q8 quantize of this thread's 4 Q elements; writes q_qs / q_d_block.
static __device__ __forceinline__ void attn_q8_quantize_q(
    const fb_fp16_t* __restrict__ q, size_t q_base, int elem_base, int quad_in_block,
    int q8_block_idx, int8_t* q_qs, float* q_d_block)
{
    float qv[4];
    qv[0] = (float) q[q_base + elem_base + 0];
    qv[1] = (float) q[q_base + elem_base + 1];
    qv[2] = (float) q[q_base + elem_base + 2];
    qv[3] = (float) q[q_base + elem_base + 3];
    float amax = fmaxf(fmaxf(fabsf(qv[0]), fabsf(qv[1])), fmaxf(fabsf(qv[2]), fabsf(qv[3])));
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, off, 8));
    }
    const float qd = amax / 127.0f;
    const float qid = (qd != 0.0f) ? 1.0f / qd : 0.0f;
    q_qs[elem_base + 0] = (int8_t) min(127, max(-127, (int) rintf(qv[0] * qid)));
    q_qs[elem_base + 1] = (int8_t) min(127, max(-127, (int) rintf(qv[1] * qid)));
    q_qs[elem_base + 2] = (int8_t) min(127, max(-127, (int) rintf(qv[2] * qid)));
    q_qs[elem_base + 3] = (int8_t) min(127, max(-127, (int) rintf(qv[3] * qid)));
    if (quad_in_block == 0) q_d_block[q8_block_idx] = qd;
}

// One online-softmax step against K/V block_q8_0 rows at kv_row_blocks. Updates
// running_max / running_sum / v_out[4]; returns nothing.
static __device__ __forceinline__ void attn_q8_step(
    const flambeau_block_q8_0* __restrict__ k_cache,
    const flambeau_block_q8_0* __restrict__ v_cache,
    size_t kv_row_blocks, int q8_block_idx, int quad_in_block, int q_packed,
    const float* q_d_block, float scale, float* red,
    float& running_max, float& running_sum, float v_out[4])
{
    const flambeau_block_q8_0* k_block = k_cache + kv_row_blocks + q8_block_idx;
    // qs sits at block offset 2 → a 4-byte int read faults on NVIDIA; load_int_b2
    // reconstructs it from two aligned uint16 reads (bit-identical on AMD).
    const int k_packed = load_int_b2(k_block->qs, quad_in_block);
    const float k_d = (float) k_block->d;
    const int sumi = dp4a(k_packed, q_packed, 0);
    const float contrib = k_d * q_d_block[q8_block_idx] * (float) sumi;
    const float score_t = block_tree_sum(contrib, red) * scale;

    const float new_max = fmaxf(running_max, score_t);
    const float scale_old = __expf(running_max - new_max);
    const float coeff_t = __expf(score_t - new_max);

    const flambeau_block_q8_0* v_block = v_cache + kv_row_blocks + q8_block_idx;
    const float v_d = (float) v_block->d;
    const int v_packed = load_int_b2(v_block->qs, quad_in_block);
    v_out[0] = v_out[0] * scale_old + coeff_t * (v_d * (float) (int) (int8_t) ((v_packed >> 0) & 0xFF));
    v_out[1] = v_out[1] * scale_old + coeff_t * (v_d * (float) (int) (int8_t) ((v_packed >> 8) & 0xFF));
    v_out[2] = v_out[2] * scale_old + coeff_t * (v_d * (float) (int) (int8_t) ((v_packed >> 16) & 0xFF));
    v_out[3] = v_out[3] * scale_old + coeff_t * (v_d * (float) (int) (int8_t) ((v_packed >> 24) & 0xFF));
    running_sum = running_sum * scale_old + coeff_t;
    running_max = new_max;
}

static __device__ void attention_decode_q8_kv_body(
    const fb_fp16_t* __restrict__ q,
    const flambeau_block_q8_0* __restrict__ k_cache,
    const flambeau_block_q8_0* __restrict__ v_cache,
    fb_fp16_t* __restrict__ out,
    int n_heads_q, int n_heads_kv, int head_dim, int n_tokens,
    float scale, int window_size, int ring_depth)
{
    const int q_head = blockIdx.x;
    if (q_head >= n_heads_q) return;
    const int kv_head = q_head / (n_heads_q / n_heads_kv);
    const int tid = threadIdx.x;
    const int elem_base = tid * 4;
    const int n_blocks_per_row = head_dim / 32;
    const int q8_block_idx = elem_base / 32;
    const int quad_in_block = (elem_base % 32) / 4;

    __shared__ int8_t q_qs[ATTN_Q8_MAX_HEAD_DIM];
    __shared__ float q_d_block[ATTN_Q8_MAX_HEAD_DIM / 32];
    __shared__ float red[ATTN_Q8_MAX_HEAD_DIM / 4];
    attn_q8_quantize_q(q, (size_t) q_head * head_dim, elem_base, quad_in_block, q8_block_idx, q_qs, q_d_block);
    __syncthreads();
    const int q_packed = ((const int*) q_qs)[tid];

    int t_start = 0;
    if (window_size > 0) { t_start = (n_tokens - 1) - window_size + 1; if (t_start < 0) t_start = 0; }
    float running_max = -ATTN_Q8_INF, running_sum = 0.0f, v_out[4] = {0, 0, 0, 0};
    for (int t = t_start; t < n_tokens; ++t) {
        const int t_phys = (ring_depth > 0) ? (t % ring_depth) : t;
        const size_t kv_row_blocks = ((size_t) t_phys * n_heads_kv + kv_head) * n_blocks_per_row;
        attn_q8_step(k_cache, v_cache, kv_row_blocks, q8_block_idx, quad_in_block, q_packed, q_d_block, scale, red, running_max, running_sum, v_out);
    }
    const float norm = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    const size_t ob = (size_t) q_head * head_dim + elem_base;
    out[ob + 0] = (fb_fp16_t) (v_out[0] * norm);
    out[ob + 1] = (fb_fp16_t) (v_out[1] * norm);
    out[ob + 2] = (fb_fp16_t) (v_out[2] * norm);
    out[ob + 3] = (fb_fp16_t) (v_out[3] * norm);
}

// Split-K chunk pass: per (q_head, chunk) Q8 attention over the chunk's tokens →
// partial (m, s, o). Combine pass reuses attention_decode_f16_splitk_combine.
static __device__ void attention_decode_q8_kv_splitk_chunk_body(
    const fb_fp16_t* __restrict__ q,
    const flambeau_block_q8_0* __restrict__ k_cache,
    const flambeau_block_q8_0* __restrict__ v_cache,
    float* __restrict__ partials_m,
    float* __restrict__ partials_s,
    float* __restrict__ partials_o,
    int n_heads_q, int n_heads_kv, int head_dim, int n_tokens,
    int n_chunks, int chunk_size, float scale, int window_size, int ring_depth, int chunk_base)
{
    const int q_head = blockIdx.x;
    const int chunk = blockIdx.y;
    if (q_head >= n_heads_q) return;
    const int kv_head = q_head / (n_heads_q / n_heads_kv);
    const int tid = threadIdx.x;
    const int elem_base = tid * 4;
    const int n_blocks_per_row = head_dim / 32;
    const int q8_block_idx = elem_base / 32;
    const int quad_in_block = (elem_base % 32) / 4;

    int t_start = chunk_base + chunk * chunk_size;
    int t_end = t_start + chunk_size;
    if (t_end > n_tokens) t_end = n_tokens;
    if (window_size > 0) {
        int window_start = (n_tokens - 1) - window_size + 1;
        if (window_start < 0) window_start = 0;
        if (t_start < window_start) t_start = window_start;
        if (t_start > t_end) t_start = t_end;
    }
    const int part_idx = q_head * n_chunks + chunk;
    if (t_start >= t_end) {
        if (tid == 0) { partials_m[part_idx] = -ATTN_Q8_INF; partials_s[part_idx] = 0.0f; }
        partials_o[(size_t) part_idx * head_dim + elem_base + 0] = 0.0f;
        partials_o[(size_t) part_idx * head_dim + elem_base + 1] = 0.0f;
        partials_o[(size_t) part_idx * head_dim + elem_base + 2] = 0.0f;
        partials_o[(size_t) part_idx * head_dim + elem_base + 3] = 0.0f;
        return;
    }

    __shared__ int8_t q_qs[ATTN_Q8_MAX_HEAD_DIM];
    __shared__ float q_d_block[ATTN_Q8_MAX_HEAD_DIM / 32];
    __shared__ float red[ATTN_Q8_MAX_HEAD_DIM / 4];
    attn_q8_quantize_q(q, (size_t) q_head * head_dim, elem_base, quad_in_block, q8_block_idx, q_qs, q_d_block);
    __syncthreads();
    const int q_packed = ((const int*) q_qs)[tid];

    float running_max = -ATTN_Q8_INF, running_sum = 0.0f, v_out[4] = {0, 0, 0, 0};
    for (int t = t_start; t < t_end; ++t) {
        const int t_phys = (ring_depth > 0) ? (t % ring_depth) : t;
        const size_t kv_row_blocks = ((size_t) t_phys * n_heads_kv + kv_head) * n_blocks_per_row;
        attn_q8_step(k_cache, v_cache, kv_row_blocks, q8_block_idx, quad_in_block, q_packed, q_d_block, scale, red, running_max, running_sum, v_out);
    }
    if (tid == 0) { partials_m[part_idx] = running_max; partials_s[part_idx] = running_sum; }
    partials_o[(size_t) part_idx * head_dim + elem_base + 0] = v_out[0];
    partials_o[(size_t) part_idx * head_dim + elem_base + 1] = v_out[1];
    partials_o[(size_t) part_idx * head_dim + elem_base + 2] = v_out[2];
    partials_o[(size_t) part_idx * head_dim + elem_base + 3] = v_out[3];
}

static __device__ void attention_prefill_q8_kv_body(
    const fb_fp16_t* __restrict__ q,
    const flambeau_block_q8_0* __restrict__ k_cache,
    const flambeau_block_q8_0* __restrict__ v_cache,
    fb_fp16_t* __restrict__ out,
    int n_q_tokens, int n_heads_q, int n_heads_kv, int head_dim,
    int n_k_tokens, int q_offset, float scale, int window_size, int ring_depth)
{
    const int q_token = blockIdx.x;
    const int q_head = blockIdx.y;
    if (q_token >= n_q_tokens || q_head >= n_heads_q) return;
    const int kv_head = q_head / (n_heads_q / n_heads_kv);
    const int tid = threadIdx.x;
    const int elem_base = tid * 4;
    const int n_blocks_per_row = head_dim / 32;
    const int q8_block_idx = elem_base / 32;
    const int quad_in_block = (elem_base % 32) / 4;

    const int qpos = q_offset + q_token;
    int limit = qpos + 1;
    if (limit > n_k_tokens) limit = n_k_tokens;
    int t_start = 0;
    if (window_size > 0) { t_start = qpos - window_size + 1; if (t_start < 0) t_start = 0; }

    __shared__ int8_t q_qs[ATTN_Q8_MAX_HEAD_DIM];
    __shared__ float q_d_block[ATTN_Q8_MAX_HEAD_DIM / 32];
    __shared__ float red[ATTN_Q8_MAX_HEAD_DIM / 4];
    attn_q8_quantize_q(q, ((size_t) q_token * n_heads_q + q_head) * head_dim, elem_base, quad_in_block, q8_block_idx, q_qs, q_d_block);
    __syncthreads();
    const int q_packed = ((const int*) q_qs)[tid];

    float running_max = -ATTN_Q8_INF, running_sum = 0.0f, v_out[4] = {0, 0, 0, 0};
    for (int t = t_start; t < limit; ++t) {
        const int t_phys = (ring_depth > 0) ? (t % ring_depth) : t;
        const size_t kv_row_blocks = ((size_t) t_phys * n_heads_kv + kv_head) * n_blocks_per_row;
        attn_q8_step(k_cache, v_cache, kv_row_blocks, q8_block_idx, quad_in_block, q_packed, q_d_block, scale, red, running_max, running_sum, v_out);
    }
    const float norm = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    const size_t ob = ((size_t) q_token * n_heads_q + q_head) * head_dim + elem_base;
    out[ob + 0] = (fb_fp16_t) (v_out[0] * norm);
    out[ob + 1] = (fb_fp16_t) (v_out[1] * norm);
    out[ob + 2] = (fb_fp16_t) (v_out[2] * norm);
    out[ob + 3] = (fb_fp16_t) (v_out[3] * norm);
}
