// attention_decode_f16_splitk — flash-decoding (two-pass split-K) variant
// of attention_decode_f16. Attacks the occupancy starvation of the single-
// pass kernel on Qwen3.6 decode (head_dim=256, n_heads_q=16, n_heads_kv=2):
// Single-pass: grid = {16, 1} → 16 blocks on 60 CUs = 27 % occupancy, each
// block serially iterates n_tokens. Measured 2869 µs at n_tokens≈2048 —
// ~700× off the 4 MiB / 1 TB/s HBM roofline. NOT bandwidth-bound; block
// count is the bottleneck.
// Split-K: grid = {16, n_chunks} → each block handles `CHUNK` tokens of
// one Q head. A second reduce kernel merges `n_chunks` partial (m, s, o)
// triples online-softmax style. At n_chunks=8 we land 128 blocks → >2×
// CU saturation with each block doing 1/8 the work.
// Layout of partials (written by the chunk kernel, read by combine):
// partials_m[n_heads_q * n_chunks] f32 — per-chunk local max
// partials_s[n_heads_q * n_chunks] f32 — per-chunk local sum
// partials_o[n_heads_q * n_chunks * head_dim] f32 — per-chunk local accum
// Per-chunk head index is blockIdx.x, chunk index is blockIdx.y.

#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

typedef _Float16 fb_fp16_t;

#define ATTN_SK_MAX_HEAD_DIM 512
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
    const float scale,
    const int window_size                  // SWA radius, 0 = unbounded causal
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

    int t_start = chunk * chunk_size;
    int t_end   = t_start + chunk_size;
    if (t_end > n_tokens) t_end = n_tokens;
    // SWA: clamp chunk range to the window. Query position is the last
    // appended token (n_tokens - 1). A chunk entirely outside the window
    // contributes -INF max + 0 sum, neutralised by the combine pass.
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

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    __shared__ float out_shared[ATTN_SK_MAX_HEAD_DIM];
    if (tid < head_dim) out_shared[tid] = 0.0f;
    // Tile-4: 4 score slots per warp so one __syncthreads serves 4
    // consecutive K rows. ATTN_SK_MAX_WARPS = 8 (head_dim=512) → 32
    // floats = 128 B, well under LDS budget.
    __shared__ float score_parts[ATTN_SK_MAX_WARPS * 4];
    __syncthreads();

    // Tile-4 inner loop: each iter processes four consecutive K/V tokens
    // sharing one cross-warp LDS roundtrip. The 1..3-token tail is handled
    // by the single-token loop afterwards.
    int t = t_start;
    for (; t + 3 < t_end; t += 4) {
        const size_t kv_row_0 = ((size_t) t * n_heads_kv + kv_head) * head_dim;
        const size_t stride   = (size_t) n_heads_kv * head_dim;
        const size_t kv_row_1 = kv_row_0 + stride;
        const size_t kv_row_2 = kv_row_0 + 2 * stride;
        const size_t kv_row_3 = kv_row_0 + 3 * stride;

        float partial_0 = 0.0f, partial_1 = 0.0f, partial_2 = 0.0f, partial_3 = 0.0f;
        if (tid < head_dim) {
            const float qv = q_shared[tid];
            partial_0 = qv * (float) k_cache[kv_row_0 + tid];
            partial_1 = qv * (float) k_cache[kv_row_1 + tid];
            partial_2 = qv * (float) k_cache[kv_row_2 + tid];
            partial_3 = qv * (float) k_cache[kv_row_3 + tid];
        }
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            partial_0 += __shfl_xor(partial_0, off, 64);
            partial_1 += __shfl_xor(partial_1, off, 64);
            partial_2 += __shfl_xor(partial_2, off, 64);
            partial_3 += __shfl_xor(partial_3, off, 64);
        }
        if (lane == 0) {
            score_parts[warp * 4 + 0] = partial_0;
            score_parts[warp * 4 + 1] = partial_1;
            score_parts[warp * 4 + 2] = partial_2;
            score_parts[warp * 4 + 3] = partial_3;
        }
        __syncthreads();
        float score_0 = 0.0f, score_1 = 0.0f, score_2 = 0.0f, score_3 = 0.0f;
        #pragma unroll
        for (int w = 0; w < ATTN_SK_MAX_WARPS; ++w) {
            if (w < nwarps) {
                score_0 += score_parts[w * 4 + 0];
                score_1 += score_parts[w * 4 + 1];
                score_2 += score_parts[w * 4 + 2];
                score_3 += score_parts[w * 4 + 3];
            }
        }
        score_0 *= scale;
        score_1 *= scale;
        score_2 *= scale;
        score_3 *= scale;

        const float tile_max = fmaxf(fmaxf(score_0, score_1), fmaxf(score_2, score_3));
        const float new_max  = fmaxf(running_max, tile_max);
        const float scale_old = __expf(running_max - new_max);
        const float coeff_0  = __expf(score_0 - new_max);
        const float coeff_1  = __expf(score_1 - new_max);
        const float coeff_2  = __expf(score_2 - new_max);
        const float coeff_3  = __expf(score_3 - new_max);

        if (tid < head_dim) {
            const float v_0 = (float) v_cache[kv_row_0 + tid];
            const float v_1 = (float) v_cache[kv_row_1 + tid];
            const float v_2 = (float) v_cache[kv_row_2 + tid];
            const float v_3 = (float) v_cache[kv_row_3 + tid];
            out_shared[tid] = out_shared[tid] * scale_old
                            + coeff_0 * v_0
                            + coeff_1 * v_1
                            + coeff_2 * v_2
                            + coeff_3 * v_3;
        }
        running_sum = running_sum * scale_old + coeff_0 + coeff_1 + coeff_2 + coeff_3;
        running_max = new_max;

        __syncthreads();
    }
    // Tail (at most three tokens): single-token update path.
    for (; t < t_end; ++t) {
        const size_t kv_row = ((size_t) t * n_heads_kv + kv_head) * head_dim;
        float my_partial = 0.0f;
        if (tid < head_dim) {
            my_partial = q_shared[tid] * (float) k_cache[kv_row + tid];
        }
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            my_partial += __shfl_xor(my_partial, off, 64);
        }
        if (lane == 0) score_parts[warp * 4] = my_partial;
        __syncthreads();
        float score_t = 0.0f;
        #pragma unroll
        for (int w = 0; w < ATTN_SK_MAX_WARPS; ++w) {
            if (w < nwarps) score_t += score_parts[w * 4];
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
