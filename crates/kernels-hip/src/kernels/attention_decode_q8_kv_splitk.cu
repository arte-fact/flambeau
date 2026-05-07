// attention_decode_q8_kv_splitk — flash-decoding (split-K) variant of
// attention_decode_q8_kv. Same occupancy fix as the F16 split-K kernel:
// partition the n_tokens dimension across grid.y so each block does
// 1/n_chunks of the work, then a combine pass merges per-chunk
// (m, s, o[head_dim]) triples online-softmax style.
//
// K and V come from `KvCache<Q8Contig>` — dequantised on the fly during
// the dot product and the V-accumulate, just like the single-pass
// `attention_decode_q8_kv` kernel. Combine pass is the SAME as the F16
// split-K combine (operates on f32 partials, layout-agnostic) — we keep
// a separate symbol so both modules stay self-contained.

#include "block_quant.cuh"
#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

#define ATTN_Q8SK_MAX_HEAD_DIM 256
#define ATTN_Q8SK_MAX_WARPS (ATTN_Q8SK_MAX_HEAD_DIM / 64)

// Pass 1 — each block (blockIdx.x = q_head, blockIdx.y = chunk) processes
// tokens [chunk*chunk_size, min((chunk+1)*chunk_size, n_tokens)).
extern "C" __global__ void flambeau_attention_decode_q8_kv_splitk_chunk(
    const fb_fp16_t* __restrict__ q,                    // [n_heads_q, head_dim]
    const flambeau_block_q8_0* __restrict__ k_cache,    // [n_tokens, n_heads_kv, head_dim/32]
    const flambeau_block_q8_0* __restrict__ v_cache,    // [n_tokens, n_heads_kv, head_dim/32]
    float* __restrict__ partials_m,                     // [n_heads_q, n_chunks]
    float* __restrict__ partials_s,                     // [n_heads_q, n_chunks]
    float* __restrict__ partials_o,                     // [n_heads_q, n_chunks, head_dim]
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,                                 // 64, 128 or 256
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

    const int nb_per_row   = head_dim / 32;
    const int block_idx    = tid / 32;
    const int block_offset = tid & 31;

    const int t_start = chunk * chunk_size;
    int t_end         = t_start + chunk_size;
    if (t_end > n_tokens) t_end = n_tokens;

    __shared__ float q_shared[ATTN_Q8SK_MAX_HEAD_DIM];
    if (tid < head_dim) {
        q_shared[tid] = (float) q[(size_t) q_head * head_dim + tid];
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    __shared__ float out_shared[ATTN_Q8SK_MAX_HEAD_DIM];
    if (tid < head_dim) out_shared[tid] = 0.0f;
    __shared__ float score_parts[ATTN_Q8SK_MAX_WARPS];
    __syncthreads();

    for (int t = t_start; t < t_end; ++t) {
        const size_t kv_row_blocks =
            ((size_t) t * n_heads_kv + kv_head) * nb_per_row;

        // Q · K[t, kv_head] — dequant K on the fly.
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
        for (int w = 0; w < ATTN_Q8SK_MAX_WARPS; ++w) {
            if (w < nwarps) score_t += score_parts[w];
        }
        score_t *= scale;

        float new_max   = fmaxf(running_max, score_t);
        float scale_old = __expf(running_max - new_max);
        float coeff_t   = __expf(score_t - new_max);

        // V-accum — dequant V on the fly.
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

    // Write per-chunk partials. If the chunk was empty (t_start >= n_tokens,
    // possible on the last chunk when n_tokens is not a multiple of
    // chunk_size), running_max stays -INF / running_sum 0 — combine treats
    // the contribution as neutral.
    const int part_idx = q_head * n_chunks + chunk;
    if (tid == 0) {
        partials_m[part_idx] = running_max;
        partials_s[part_idx] = running_sum;
    }
    if (tid < head_dim) {
        partials_o[(size_t) part_idx * head_dim + tid] = out_shared[tid];
    }
}

// Pass 2 — combine n_chunks partials per Q head. Identical math to the
// F16 split-K combine; symbol kept distinct so the Q8 module is self-
// contained and can be loaded without the F16 split-K module present.
extern "C" __global__ void flambeau_attention_decode_q8_kv_splitk_combine(
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

    float g_max = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) {
        float mc = partials_m[q_head * n_chunks + c];
        g_max = fmaxf(g_max, mc);
    }

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
        float norm = (g_sum > 0.0f) ? 1.0f / g_sum : 0.0f;
        out[(size_t) q_head * head_dim + tid] = (fb_fp16_t) (g_out * norm);
    }
}
