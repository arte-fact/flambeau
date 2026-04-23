// attention_decode_f16_splitk — flash-decoding (two-pass split-K) variant
// of attention_decode_f16. Attacks the occupancy starvation of the single-
// pass kernel on Qwen3.6 decode (head_dim=256, n_heads_q=16, n_heads_kv=2):
//
// Single-pass: grid = {16, 1} → 16 blocks on 60 CUs = 27 % occupancy, each
//   block serially iterates n_tokens. Measured 2869 µs at n_tokens≈2048 —
//   ~700× off the 4 MiB / 1 TB/s HBM roofline. NOT bandwidth-bound; block
//   count is the bottleneck.
//
// Split-K: grid = {16, n_chunks} → each block handles `CHUNK` tokens of
//   one Q head. A second reduce kernel merges `n_chunks` partial (m, s, o)
//   triples online-softmax style. At n_chunks=8 we land 128 blocks → >2×
//   CU saturation with each block doing 1/8 the work.
//
// Layout of partials (written by the chunk kernel, read by combine):
//   partials_m[n_heads_q * n_chunks]             f32  — per-chunk local max
//   partials_s[n_heads_q * n_chunks]             f32  — per-chunk local sum
//   partials_o[n_heads_q * n_chunks * head_dim]  f32  — per-chunk local accum
//
// Per-chunk head index is blockIdx.x, chunk index is blockIdx.y.

#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

typedef _Float16 fb_fp16_t;

#define ATTN_SK_MAX_HEAD_DIM 256
#define ATTN_SK_MAX_WARPS (ATTN_SK_MAX_HEAD_DIM / 64)

// Pass 1 — each block (blockIdx.x = q_head, blockIdx.y = chunk) processes
// tokens [chunk*CHUNK, min((chunk+1)*CHUNK, n_tokens)).
extern "C" __global__ void flambeau_attention_decode_f16_splitk_chunk(
    const fb_fp16_t* __restrict__ q,       // [n_heads_q, head_dim]
    const fb_fp16_t* __restrict__ k_cache, // [n_tokens, n_heads_kv, head_dim]
    const fb_fp16_t* __restrict__ v_cache, // [n_tokens, n_heads_kv, head_dim]
    float* __restrict__ partials_m,        // [n_heads_q, n_chunks]
    float* __restrict__ partials_s,        // [n_heads_q, n_chunks]
    float* __restrict__ partials_o,        // [n_heads_q, n_chunks, head_dim]
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,
    const int n_tokens,
    const int n_chunks,
    const int chunk_size,
    const float scale
) {
    const int q_head = blockIdx.x;
    const int chunk  = blockIdx.y;
    if (q_head >= n_heads_q) return;
    const int group   = n_heads_q / n_heads_kv;
    const int kv_head = q_head / group;

    const int tid    = threadIdx.x;
    const int warp   = tid >> 6;
    const int lane   = tid & 63;
    const int nwarps = blockDim.x >> 6;

    const int t_start = chunk * chunk_size;
    int t_end         = t_start + chunk_size;
    if (t_end > n_tokens) t_end = n_tokens;

    __shared__ float q_shared[ATTN_SK_MAX_HEAD_DIM];
    if (tid < head_dim) {
        q_shared[tid] = (float) q[(size_t) q_head * head_dim + tid];
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    __shared__ float out_shared[ATTN_SK_MAX_HEAD_DIM];
    if (tid < head_dim) out_shared[tid] = 0.0f;
    __shared__ float score_parts[ATTN_SK_MAX_WARPS];
    __syncthreads();

    // If the chunk is empty (n_tokens < t_start, possible when
    // n_chunks > ceil(n_tokens/chunk_size) — not by our launch, but guard
    // anyway), still write partials_m = -INF, s = 0, o = 0 so combine sees
    // a neutral contribution.
    for (int t = t_start; t < t_end; ++t) {
        const size_t kv_row = ((size_t) t * n_heads_kv + kv_head) * head_dim;

        float my_partial = 0.0f;
        if (tid < head_dim) {
            my_partial = q_shared[tid] * (float) k_cache[kv_row + tid];
        }
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            my_partial += __shfl_xor(my_partial, off, 64);
        }
        if (lane == 0) score_parts[warp] = my_partial;
        __syncthreads();
        float score_t = 0.0f;
        #pragma unroll
        for (int w = 0; w < ATTN_SK_MAX_WARPS; ++w) {
            if (w < nwarps) score_t += score_parts[w];
        }
        score_t *= scale;

        float new_max   = fmaxf(running_max, score_t);
        float scale_old = __expf(running_max - new_max);
        float coeff_t   = __expf(score_t - new_max);

        if (tid < head_dim) {
            const float v_t = (float) v_cache[kv_row + tid];
            out_shared[tid] = out_shared[tid] * scale_old + coeff_t * v_t;
        }
        running_sum = running_sum * scale_old + coeff_t;
        running_max = new_max;

        __syncthreads();
    }

    // Write partials: one per-chunk (m, s, o[head_dim]) row per head.
    const int part_idx = q_head * n_chunks + chunk;
    if (tid == 0) {
        partials_m[part_idx] = running_max;
        partials_s[part_idx] = running_sum;
    }
    if (tid < head_dim) {
        partials_o[(size_t) part_idx * head_dim + tid] = out_shared[tid];
    }
}

// Pass 2 — combine n_chunks partials per Q head. Grid = {n_heads_q},
// block = head_dim threads. Merges online-softmax style.
extern "C" __global__ void flambeau_attention_decode_f16_splitk_combine(
    const float* __restrict__ partials_m,  // [n_heads_q, n_chunks]
    const float* __restrict__ partials_s,  // [n_heads_q, n_chunks]
    const float* __restrict__ partials_o,  // [n_heads_q, n_chunks, head_dim]
    fb_fp16_t* __restrict__ out,           // [n_heads_q, head_dim]
    const int n_heads_q,
    const int n_chunks,
    const int head_dim
) {
    const int q_head = blockIdx.x;
    if (q_head >= n_heads_q) return;
    const int tid = threadIdx.x;

    // Find global max across chunks (cheap — n_chunks small, e.g. 8).
    float g_max = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) {
        float mc = partials_m[q_head * n_chunks + c];
        g_max = fmaxf(g_max, mc);
    }

    // Accumulate global sum and per-lane global out.
    float g_sum = 0.0f;
    float g_out = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        float mc = partials_m[q_head * n_chunks + c];
        float sc = partials_s[q_head * n_chunks + c];
        float w  = __expf(mc - g_max);
        g_sum += sc * w;
        if (tid < head_dim) {
            float oc = partials_o[((size_t) q_head * n_chunks + c) * head_dim + tid];
            g_out += oc * w;
        }
    }

    if (tid < head_dim) {
        // g_sum is the same across lanes (partials are per-chunk scalars); it
        // can be zero if all chunks were empty — guard like the single-pass
        // kernel does implicitly by only writing when sum > 0.
        float norm = (g_sum > 0.0f) ? 1.0f / g_sum : 0.0f;
        out[(size_t) q_head * head_dim + tid] = (fb_fp16_t) (g_out * norm);
    }
}
