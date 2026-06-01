// attention_prefill_q8_kv — GQA prefill attention with Q8_0 KV cache.
// Score path is now pure
// integer dp4a (`v_dot4_i32_i8` on gfx906) — Q for each (q_token,
// q_head) block is quantized to Q8_0 in LDS once at block start; K
// is read packed as int32 and dotted via `__builtin_amdgcn_sdot4`;
// scalar K.d × Q.d is applied once per Q8_0 block (32 elements).
// V path stays on FP16 dequant per element (Phase 5 covers PV).
// Reference: same `vec_dot_fattn_vec_KQ_q8_0` pattern as decode kernel
// in `/artefact/llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh:264`.
// Layout:
// Q: [n_q_tokens, n_heads_q, head_dim] FP16
// K/V: [n_k_tokens, n_heads_kv, head_dim/32] block_q8_0
// Out: [n_q_tokens, n_heads_q, head_dim] FP16
// Supported head_dim: {64, 128, 256}.
// Launch: blockDim = { head_dim/4 }, gridDim = { n_q_tokens, n_heads_q, 1 }.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

#define ATTN_Q8DPP_MAX_HEAD_DIM 256

extern "C" __global__ void flambeau_attention_prefill_q8_kv(
    const fb_fp16_t* __restrict__ q,                    // [n_q_tokens, n_heads_q, head_dim]
    const flambeau_block_q8_0* __restrict__ k_cache,    // [n_k_tokens, n_heads_kv, head_dim/32]
    const flambeau_block_q8_0* __restrict__ v_cache,    // [n_k_tokens, n_heads_kv, head_dim/32]
    fb_fp16_t* __restrict__ out,                        // [n_q_tokens, n_heads_q, head_dim]
    const int n_q_tokens,
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,                                 // 64, 128 or 256
    const int n_k_tokens,
    const int q_offset,                                 // global position of Q[0]
    const float scale,
    const int window_size                               // 0 = unbounded causal; >0 = SWA
) {
    const int q_token = blockIdx.x;
    const int q_head  = blockIdx.y;
    if (q_token >= n_q_tokens || q_head >= n_heads_q) return;
    const int group   = n_heads_q / n_heads_kv;
    const int kv_head = q_head / group;

    const int tid               = threadIdx.x;        // 0..head_dim/4 - 1
    const int dim_quad          = tid;
    const int elem_base         = tid * 4;
    const int n_blocks_per_row  = head_dim / 32;
    const int q8_block_idx      = elem_base / 32;
    const int n_quads_per_block = 8;
    const int quad_in_block     = (elem_base % 32) / 4;

    // Causal mask: keys with t < qpos+1 are visible.
    // SWA mask (window_size > 0): keys with t < qpos - window_size + 1
    // are excluded as well. Per-query both ends move with `q_token`.
    const int qpos = q_offset + q_token;
    int limit = qpos + 1;
    if (limit > n_k_tokens) limit = n_k_tokens;
    int t_start = 0;
    if (window_size > 0) {
        t_start = qpos - window_size + 1;
        if (t_start < 0) t_start = 0;
    }

    // 1. Q → Q8_0 in LDS, once per (q_token, q_head) block.
    __shared__ int8_t q_qs[ATTN_Q8DPP_MAX_HEAD_DIM];
    __shared__ float  q_d_block[ATTN_Q8DPP_MAX_HEAD_DIM / 32];

    float qv[4];
    qv[0] = (float) q[((size_t) q_token * n_heads_q + q_head) * head_dim + elem_base + 0];
    qv[1] = (float) q[((size_t) q_token * n_heads_q + q_head) * head_dim + elem_base + 1];
    qv[2] = (float) q[((size_t) q_token * n_heads_q + q_head) * head_dim + elem_base + 2];
    qv[3] = (float) q[((size_t) q_token * n_heads_q + q_head) * head_dim + elem_base + 3];

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

    for (int t = t_start; t < limit; ++t) {
        const size_t kv_row_blocks =
            ((size_t) t * n_heads_kv + kv_head) * n_blocks_per_row;

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
            score_t += __shfl_xor(score_t, 8,  n_blocks_per_row * n_quads_per_block);
        }
        if (n_blocks_per_row >= 4) {
            score_t += __shfl_xor(score_t, 16, n_blocks_per_row * n_quads_per_block);
        }
        if (n_blocks_per_row >= 8) {
            score_t += __shfl_xor(score_t, 32, n_blocks_per_row * n_quads_per_block);
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

    const float norm = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    out[((size_t) q_token * n_heads_q + q_head) * head_dim + elem_base + 0] = (fb_fp16_t) (v_out[0] * norm);
    out[((size_t) q_token * n_heads_q + q_head) * head_dim + elem_base + 1] = (fb_fp16_t) (v_out[1] * norm);
    out[((size_t) q_token * n_heads_q + q_head) * head_dim + elem_base + 2] = (fb_fp16_t) (v_out[2] * norm);
    out[((size_t) q_token * n_heads_q + q_head) * head_dim + elem_base + 3] = (fb_fp16_t) (v_out[3] * norm);
}
