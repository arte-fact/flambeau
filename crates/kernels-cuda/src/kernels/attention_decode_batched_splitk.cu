// Batched-slots split-K (flash-decoding) decode attention, F16 KV. CUDA-only
// (architectural rule 5 keeps this out of the shared header; gfx906 keeps the
// serial attention_decode_f16_batched and the launch falls back when this
// module is absent). Adds a chunk dimension to attention_decode_f16_batched so
// each block reduces one (q_head, slot, chunk) of CHUNK tokens to a partial
// (m, s, o); a combine pass merges the n_chunks partials per (q_head, slot).
// Empty / out-of-window chunks emit m=-inf,s=0 -> weight 0 in combine, so
// variable per-slot lengths and SWA windowing need no special-casing.
#include "sm_80.cuh"
#include "block_quant.cuh"
#include "attention_f16.cuh"

static __device__ void batched_splitk_chunk(
    const fb_fp16_t* __restrict__ q_batched,
    const uint64_t* __restrict__ k_cache_ptrs,
    const uint64_t* __restrict__ v_cache_ptrs,
    float* __restrict__ partials_m,
    float* __restrict__ partials_s,
    float* __restrict__ partials_o,
    const int* __restrict__ n_tokens_kv,
    int n_heads_q, int n_heads_kv, int head_dim, int n_slots,
    int n_chunks, int chunk_size,
    float scale, int window_size, int ring_depth)
{
    const int q_head = blockIdx.x;
    const int slot   = blockIdx.y;
    const int chunk  = blockIdx.z;
    if (q_head >= n_heads_q || slot >= n_slots || chunk >= n_chunks) return;
    const int kv_head = q_head / (n_heads_q / n_heads_kv);
    const int tid = threadIdx.x;

    const fb_fp16_t* k_cache = (const fb_fp16_t*) k_cache_ptrs[slot];
    const fb_fp16_t* v_cache = (const fb_fp16_t*) v_cache_ptrs[slot];
    const int n_tokens = n_tokens_kv[slot];
    const fb_fp16_t* q_row = q_batched + ((size_t) slot * n_heads_q + q_head) * (size_t) head_dim;

    int t_start = chunk * chunk_size;
    int t_end   = t_start + chunk_size;
    if (t_end > n_tokens) t_end = n_tokens;
    if (window_size > 0) {
        int swa_min = (n_tokens - 1) - window_size + 1;
        if (swa_min < 0) swa_min = 0;
        if (t_start < swa_min) t_start = swa_min;
    }
    if (t_start > t_end) t_start = t_end;

    __shared__ float q_shared[ATTN_F16_MAX_HEAD_DIM];
    __shared__ float out_shared[ATTN_F16_MAX_HEAD_DIM];
    __shared__ float score_parts[ATTN_F16_MAX_HEAD_DIM / WARP_SIZE];
    if (tid < head_dim) { q_shared[tid] = (float) q_row[tid]; out_shared[tid] = 0.0f; }
    __syncthreads();

    float running_max = -ATTN_F16_INF;
    float running_sum = 0.0f;
    for (int t = t_start; t < t_end; ++t) {
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

    const int part_idx = (q_head * n_slots + slot) * n_chunks + chunk;
    if (tid == 0) { partials_m[part_idx] = running_max; partials_s[part_idx] = running_sum; }
    if (tid < head_dim) partials_o[(size_t) part_idx * head_dim + tid] = out_shared[tid];
}

static __device__ void batched_splitk_combine(
    const float* __restrict__ partials_m,
    const float* __restrict__ partials_s,
    const float* __restrict__ partials_o,
    fb_fp16_t* __restrict__ out_batched,
    int n_heads_q, int n_slots, int n_chunks, int head_dim)
{
    const int q_head = blockIdx.x;
    const int slot   = blockIdx.y;
    if (q_head >= n_heads_q || slot >= n_slots) return;
    const int tid = threadIdx.x;
    const int base = (q_head * n_slots + slot) * n_chunks;

    float g_max = -ATTN_F16_INF;
    for (int c = 0; c < n_chunks; ++c) g_max = fmaxf(g_max, partials_m[base + c]);

    float g_sum = 0.0f, g_out = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float mc = partials_m[base + c];
        const float sc = partials_s[base + c];
        const float w  = __expf(mc - g_max);
        g_sum += sc * w;
        if (tid < head_dim) g_out += partials_o[(size_t) (base + c) * head_dim + tid] * w;
    }
    if (tid < head_dim) {
        const float norm = (g_sum > 0.0f) ? (1.0f / g_sum) : 0.0f;
        out_batched[((size_t) slot * n_heads_q + q_head) * head_dim + tid] = (fb_fp16_t) (g_out * norm);
    }
}

extern "C" __global__ void flambeau_attention_decode_f16_batched_splitk_chunk(
    const fb_fp16_t* q_batched, const uint64_t* k_cache_ptrs, const uint64_t* v_cache_ptrs,
    float* partials_m, float* partials_s, float* partials_o, const int* n_tokens_kv,
    int n_heads_q, int n_heads_kv, int head_dim, int n_slots, int n_chunks, int chunk_size,
    float scale, int window_size, int ring_depth)
{
    batched_splitk_chunk(q_batched, k_cache_ptrs, v_cache_ptrs, partials_m, partials_s, partials_o,
        n_tokens_kv, n_heads_q, n_heads_kv, head_dim, n_slots, n_chunks, chunk_size,
        scale, window_size, ring_depth);
}

extern "C" __global__ void flambeau_attention_decode_f16_batched_splitk_combine(
    const float* partials_m, const float* partials_s, const float* partials_o,
    fb_fp16_t* out_batched, int n_heads_q, int n_slots, int n_chunks, int head_dim)
{
    batched_splitk_combine(partials_m, partials_s, partials_o, out_batched,
        n_heads_q, n_slots, n_chunks, head_dim);
}
