#pragma once

// Flash-decoding (two-pass split-K) decode attention, F16 KV. Pass 1 (chunk):
// each block (q_head, chunk) reduces CHUNK tokens to a partial (m, s, o). Pass 2
// (combine): merge n_chunks partials per head, online-softmax style. The chunk
// kernel's only cross-lane op is the Q·K dot — a plain __shfl_xor butterfly over
// the WARP_SIZE seam (NOT the DPP warp_reduce_sum, for bit-identity with the
// hand kernel). The combine kernel is per-lane (no warp ops), wave-agnostic.

#ifndef ATTN_SK_INFINITY
#define ATTN_SK_INFINITY __builtin_huge_valf()
#endif
#ifndef ATTN_SK_MAX_HEAD_DIM
#define ATTN_SK_MAX_HEAD_DIM 512
#endif
#define ATTN_SK_MAX_WARPS (ATTN_SK_MAX_HEAD_DIM / WARP_SIZE)

static __device__ void attention_decode_f16_splitk_chunk_body(
    const fb_fp16_t* __restrict__ q,
    const fb_fp16_t* __restrict__ k_cache,
    const fb_fp16_t* __restrict__ v_cache,
    float* __restrict__ partials_m,
    float* __restrict__ partials_s,
    float* __restrict__ partials_o,
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,
    const int n_tokens,
    const int n_chunks,
    const int chunk_size,
    const float scale,
    const int window_size,
    const int ring_depth,
    const int chunk_base
) {
    const int q_head = blockIdx.x;
    const int chunk = blockIdx.y;
    if (q_head >= n_heads_q) return;
    const int group = n_heads_q / n_heads_kv;
    const int kv_head = q_head / group;

    const int tid = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int nwarps = blockDim.x / WARP_SIZE;

    int t_start = chunk_base + chunk * chunk_size;
    int t_end = t_start + chunk_size;
    if (t_end > n_tokens) t_end = n_tokens;
    if (window_size > 0) {
        const int qpos = n_tokens - 1;
        int swa_min = qpos - window_size + 1;
        if (swa_min < 0) swa_min = 0;
        if (t_start < swa_min) t_start = swa_min;
    }
    if (t_start > t_end) t_start = t_end;

    __shared__ float q_shared[ATTN_SK_MAX_HEAD_DIM];
    if (tid < head_dim) {
        q_shared[tid] = (float) q[(size_t) q_head * head_dim + tid];
    }

    float running_max = -ATTN_SK_INFINITY;
    float running_sum = 0.0f;
    __shared__ float out_shared[ATTN_SK_MAX_HEAD_DIM];
    if (tid < head_dim) out_shared[tid] = 0.0f;
    __shared__ float score_parts[ATTN_SK_MAX_WARPS * 4];
    __syncthreads();

    int t = t_start;
    for (; t + 3 < t_end; t += 4) {
        const int tp0 = (ring_depth > 0) ? (t % ring_depth) : t;
        const int tp1 = (ring_depth > 0) ? ((t + 1) % ring_depth) : (t + 1);
        const int tp2 = (ring_depth > 0) ? ((t + 2) % ring_depth) : (t + 2);
        const int tp3 = (ring_depth > 0) ? ((t + 3) % ring_depth) : (t + 3);
        const size_t kv_row_0 = ((size_t) tp0 * n_heads_kv + kv_head) * head_dim;
        const size_t kv_row_1 = ((size_t) tp1 * n_heads_kv + kv_head) * head_dim;
        const size_t kv_row_2 = ((size_t) tp2 * n_heads_kv + kv_head) * head_dim;
        const size_t kv_row_3 = ((size_t) tp3 * n_heads_kv + kv_head) * head_dim;

        float partial_0 = 0.0f, partial_1 = 0.0f, partial_2 = 0.0f, partial_3 = 0.0f;
        if (tid < head_dim) {
            const float qv = q_shared[tid];
            partial_0 = qv * (float) k_cache[kv_row_0 + tid];
            partial_1 = qv * (float) k_cache[kv_row_1 + tid];
            partial_2 = qv * (float) k_cache[kv_row_2 + tid];
            partial_3 = qv * (float) k_cache[kv_row_3 + tid];
        }
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            partial_0 += __shfl_xor(partial_0, off, WARP_SIZE);
            partial_1 += __shfl_xor(partial_1, off, WARP_SIZE);
            partial_2 += __shfl_xor(partial_2, off, WARP_SIZE);
            partial_3 += __shfl_xor(partial_3, off, WARP_SIZE);
        }
        if (lane == 0) {
            score_parts[warp * 4 + 0] = partial_0;
            score_parts[warp * 4 + 1] = partial_1;
            score_parts[warp * 4 + 2] = partial_2;
            score_parts[warp * 4 + 3] = partial_3;
        }
        __syncthreads();
        float score_0 = 0.0f, score_1 = 0.0f, score_2 = 0.0f, score_3 = 0.0f;
        for (int w = 0; w < nwarps; ++w) {
            score_0 += score_parts[w * 4 + 0];
            score_1 += score_parts[w * 4 + 1];
            score_2 += score_parts[w * 4 + 2];
            score_3 += score_parts[w * 4 + 3];
        }
        score_0 *= scale;
        score_1 *= scale;
        score_2 *= scale;
        score_3 *= scale;

        const float tile_max = fmaxf(fmaxf(score_0, score_1), fmaxf(score_2, score_3));
        const float new_max = fmaxf(running_max, tile_max);
        const float scale_old = __expf(running_max - new_max);
        const float coeff_0 = __expf(score_0 - new_max);
        const float coeff_1 = __expf(score_1 - new_max);
        const float coeff_2 = __expf(score_2 - new_max);
        const float coeff_3 = __expf(score_3 - new_max);

        if (tid < head_dim) {
            const float v_0 = (float) v_cache[kv_row_0 + tid];
            const float v_1 = (float) v_cache[kv_row_1 + tid];
            const float v_2 = (float) v_cache[kv_row_2 + tid];
            const float v_3 = (float) v_cache[kv_row_3 + tid];
            out_shared[tid] = out_shared[tid] * scale_old
                            + coeff_0 * v_0 + coeff_1 * v_1 + coeff_2 * v_2 + coeff_3 * v_3;
        }
        running_sum = running_sum * scale_old + coeff_0 + coeff_1 + coeff_2 + coeff_3;
        running_max = new_max;
        __syncthreads();
    }
    for (; t < t_end; ++t) {
        const int t_phys = (ring_depth > 0) ? (t % ring_depth) : t;
        const size_t kv_row = ((size_t) t_phys * n_heads_kv + kv_head) * head_dim;
        float my_partial = (tid < head_dim) ? q_shared[tid] * (float) k_cache[kv_row + tid] : 0.0f;
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            my_partial += __shfl_xor(my_partial, off, WARP_SIZE);
        }
        if (lane == 0) score_parts[warp * 4] = my_partial;
        __syncthreads();
        float score_t = 0.0f;
        for (int w = 0; w < nwarps; ++w) {
            score_t += score_parts[w * 4];
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

    const int part_idx = q_head * n_chunks + chunk;
    if (tid == 0) {
        partials_m[part_idx] = running_max;
        partials_s[part_idx] = running_sum;
    }
    if (tid < head_dim) {
        partials_o[(size_t) part_idx * head_dim + tid] = out_shared[tid];
    }
}

static __device__ void attention_decode_f16_splitk_combine_body(
    const float* __restrict__ partials_m,
    const float* __restrict__ partials_s,
    const float* __restrict__ partials_o,
    fb_fp16_t* __restrict__ out,
    const int n_heads_q,
    const int n_chunks,
    const int head_dim
) {
    const int q_head = blockIdx.x;
    if (q_head >= n_heads_q) return;
    const int tid = threadIdx.x;

    float g_max = -ATTN_SK_INFINITY;
    for (int c = 0; c < n_chunks; ++c) {
        g_max = fmaxf(g_max, partials_m[q_head * n_chunks + c]);
    }

    float g_sum = 0.0f;
    float g_out = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float mc = partials_m[q_head * n_chunks + c];
        const float sc = partials_s[q_head * n_chunks + c];
        const float w = __expf(mc - g_max);
        g_sum += sc * w;
        if (tid < head_dim) {
            g_out += partials_o[((size_t) q_head * n_chunks + c) * head_dim + tid] * w;
        }
    }

    if (tid < head_dim) {
        const float norm = (g_sum > 0.0f) ? 1.0f / g_sum : 0.0f;
        out[(size_t) q_head * head_dim + tid] = (fb_fp16_t) (g_out * norm);
    }
}
