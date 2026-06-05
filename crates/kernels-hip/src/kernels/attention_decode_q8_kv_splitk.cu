// attention_decode_q8_kv_splitk — flash-decoding (split-K) variant of
// attention_decode_q8_kv. Same occupancy fix as the F16 split-K kernel:
// partition the n_tokens dimension across grid.y so each block does
// 1/n_chunks of the work, then a combine pass merges per-chunk
// (m, s, o[head_dim]) triples online-softmax style.
// chunk pass is now pure
// integer dp4a (`v_dot4_i32_i8`), mirroring the single-pass kernel.
// Q is quantized to Q8_0 in LDS once per chunk; K is read packed as
// int32, dotted via `__builtin_amdgcn_sdot4`; scalar K.d × Q.d
// applied once per Q8_0 block. V path stays on FP16 dequant per
// element (Phase 5 covers PV in INT8).
// Combine pass is identical math to the F16 split-K combine and to
// the prior FP-dequant Q8 split-K combine — operates on f32 partials,
// layout-agnostic. Kept as a separate symbol so the Q8 module is
// self-contained.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

#define ATTN_Q8SK_MAX_HEAD_DIM 512
#define ATTN_Q8SK_MAX_WAVES (ATTN_Q8SK_MAX_HEAD_DIM / 256)

extern "C" __global__ void flambeau_attention_decode_q8_kv_splitk_chunk(
    const fb_fp16_t* __restrict__ q,                    // [n_heads_q, head_dim]
    const flambeau_block_q8_0* __restrict__ k_cache,    // [n_tokens, n_heads_kv, head_dim/32]
    const flambeau_block_q8_0* __restrict__ v_cache,    // [n_tokens, n_heads_kv, head_dim/32]
    float* __restrict__ partials_m,                     // [n_heads_q, n_chunks]
    float* __restrict__ partials_s,                     // [n_heads_q, n_chunks]
    float* __restrict__ partials_o,                     // [n_heads_q, n_chunks, head_dim]
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,                                 // 64, 128, 256 or 512
    const int n_tokens,
    const int n_chunks,
    const int chunk_size,
    const float scale,
    const int window_size,                              // 0 = unbounded causal; >0 = SWA
    const int ring_depth                                // ring-buffer slab depth in rows; 0 = absolute
) {
    const int q_head = blockIdx.x;
    const int chunk  = blockIdx.y;
    if (q_head >= n_heads_q) return;
    const int group   = n_heads_q / n_heads_kv;
    const int kv_head = q_head / group;

    // 1 thread per int32-packed quad (4 elements). Same layout as
    // single-pass dp4a kernel.
    const int tid               = threadIdx.x;
    const int dim_quad          = tid;
    const int elem_base         = tid * 4;
    const int n_blocks_per_row  = head_dim / 32;
    const int q8_block_idx      = elem_base / 32;
    const int n_quads_per_block = 8;
    const int quad_in_block     = (elem_base % 32) / 4;

    int t_start = chunk * chunk_size;
    int t_end   = t_start + chunk_size;
    if (t_end > n_tokens) t_end = n_tokens;
    // SWA mask: keys older than (qpos - window_size + 1) are
    // excluded. Apply per chunk by raising t_start when the window
    // begins inside this chunk. Chunks entirely before the window
    // collapse to empty (t_start >= t_end). For those we write the
    // neutral-element partials and return early, skipping the
    // Q→Q8 LDS quantize phase (the dominant per-block cost beyond
    // the inner score loop).
    if (window_size > 0) {
        const int qpos = n_tokens - 1;
        int window_start = qpos - window_size + 1;
        if (window_start < 0) window_start = 0;
        if (t_start < window_start) t_start = window_start;
        if (t_start > t_end) t_start = t_end;
    }
    if (t_start >= t_end) {
        const int part_idx = q_head * n_chunks + chunk;
        if (tid == 0) {
            partials_m[part_idx] = -INFINITY;
            partials_s[part_idx] = 0.0f;
        }
        partials_o[(size_t) part_idx * head_dim + elem_base + 0] = 0.0f;
        partials_o[(size_t) part_idx * head_dim + elem_base + 1] = 0.0f;
        partials_o[(size_t) part_idx * head_dim + elem_base + 2] = 0.0f;
        partials_o[(size_t) part_idx * head_dim + elem_base + 3] = 0.0f;
        return;
    }

    // 1. Q → Q8_0 in LDS, once per (q_head, chunk) block. Same trick as
    // single-pass: amax over each 32-elem group, roundf(v/d).
    __shared__ int8_t q_qs[ATTN_Q8SK_MAX_HEAD_DIM];
    __shared__ float  q_d_block[ATTN_Q8SK_MAX_HEAD_DIM / 32];

    float qv[4];
    qv[0] = (float) q[(size_t) q_head * head_dim + elem_base + 0];
    qv[1] = (float) q[(size_t) q_head * head_dim + elem_base + 1];
    qv[2] = (float) q[(size_t) q_head * head_dim + elem_base + 2];
    qv[3] = (float) q[(size_t) q_head * head_dim + elem_base + 3];

    float amax = fmaxf(fmaxf(fabsf(qv[0]), fabsf(qv[1])),
                       fmaxf(fabsf(qv[2]), fabsf(qv[3])));
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, off, n_quads_per_block));
    }
    const float qd  = amax / 127.0f;
    const float qid = (qd != 0.0f) ? 1.0f / qd : 0.0f;
    q_qs[elem_base + 0] = (int8_t) min(127, max(-127, (int) rintf(qv[0] * qid)));
    q_qs[elem_base + 1] = (int8_t) min(127, max(-127, (int) rintf(qv[1] * qid)));
    q_qs[elem_base + 2] = (int8_t) min(127, max(-127, (int) rintf(qv[2] * qid)));
    q_qs[elem_base + 3] = (int8_t) min(127, max(-127, (int) rintf(qv[3] * qid)));
    if (quad_in_block == 0) {
        q_d_block[q8_block_idx] = qd;
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float v_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    __syncthreads();

    const int* q_qs_int = (const int*) q_qs;
    const int q_packed  = q_qs_int[dim_quad];

    for (int t = t_start; t < t_end; ++t) {
        const int t_phys = (ring_depth > 0) ? (t % ring_depth) : t;
        const size_t kv_row_blocks =
            ((size_t) t_phys * n_heads_kv + kv_head) * n_blocks_per_row;

        const flambeau_block_q8_0* k_block = k_cache + kv_row_blocks + q8_block_idx;
        const int   k_packed = ((const int*) k_block->qs)[quad_in_block];
        const float k_d      = (float) k_block->d;

        const int sumi_thread = gfx906_dp4a(k_packed, q_packed, 0);

        int sumi_block = sumi_thread;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            sumi_block += __shfl_xor(sumi_block, off, n_quads_per_block);
        }

        const float qd_block = q_d_block[q8_block_idx];
        float score_t = k_d * qd_block * (float) sumi_block;

        if (n_blocks_per_row >= 2) {
            score_t += __shfl_xor(score_t, 8,  min(n_blocks_per_row * n_quads_per_block, 64));
        }
        if (n_blocks_per_row >= 4) {
            score_t += __shfl_xor(score_t, 16, min(n_blocks_per_row * n_quads_per_block, 64));
        }
        if (n_blocks_per_row >= 8) {
            score_t += __shfl_xor(score_t, 32, min(n_blocks_per_row * n_quads_per_block, 64));
        }
        // d=512: 2 waves per block; xor chain above ends at stride 32
        // (wave-internal). Cross-wave LDS reduce so every lane sees the
        // full sum across both waves.
        if (head_dim > 256) {
            __shared__ float score_parts[ATTN_Q8SK_MAX_WAVES];
            const int warp = tid >> 6;
            const int lane = tid & 63;
            if (lane == 0) score_parts[warp] = score_t;
            __syncthreads();
            score_t = score_parts[0] + score_parts[1];
            __syncthreads();
        }
        score_t *= scale;

        float new_max   = fmaxf(running_max, score_t);
        float scale_old = __expf(running_max - new_max);
        float coeff_t   = __expf(score_t - new_max);

        const flambeau_block_q8_0* v_block = v_cache + kv_row_blocks + q8_block_idx;
        const float v_d      = (float) v_block->d;
        const int   v_packed = ((const int*) v_block->qs)[quad_in_block];
        const int v_q0 = (int)(int8_t)((v_packed >>  0) & 0xFF);
        const int v_q1 = (int)(int8_t)((v_packed >>  8) & 0xFF);
        const int v_q2 = (int)(int8_t)((v_packed >> 16) & 0xFF);
        const int v_q3 = (int)(int8_t)((v_packed >> 24) & 0xFF);

        v_out[0] = v_out[0] * scale_old + coeff_t * (v_d * (float) v_q0);
        v_out[1] = v_out[1] * scale_old + coeff_t * (v_d * (float) v_q1);
        v_out[2] = v_out[2] * scale_old + coeff_t * (v_d * (float) v_q2);
        v_out[3] = v_out[3] * scale_old + coeff_t * (v_d * (float) v_q3);
        running_sum = running_sum * scale_old + coeff_t;
        running_max = new_max;
    }

    // Write per-chunk partials. Empty-chunk handling: when n_tokens is
    // not a multiple of chunk_size, the last chunk's tail tokens give a
    // smaller t_end; running_max stays -INF / running_sum 0 — combine
    // treats the contribution as neutral.
    const int part_idx = q_head * n_chunks + chunk;
    if (tid == 0) {
        partials_m[part_idx] = running_max;
        partials_s[part_idx] = running_sum;
    }
    partials_o[(size_t) part_idx * head_dim + elem_base + 0] = v_out[0];
    partials_o[(size_t) part_idx * head_dim + elem_base + 1] = v_out[1];
    partials_o[(size_t) part_idx * head_dim + elem_base + 2] = v_out[2];
    partials_o[(size_t) part_idx * head_dim + elem_base + 3] = v_out[3];
}

// Combine pass — unchanged from the FP-dequant version. Operates on
// f32 partials so layout-agnostic. Kept as a separate symbol from the
// F16 combine kernel so the Q8 module is self-contained.
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
