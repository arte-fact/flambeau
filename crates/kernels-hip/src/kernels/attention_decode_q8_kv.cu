// attention_decode_q8_kv — GQA decode attention with Q8_0 KV cache.
// Score path is now pure
// integer dp4a (`v_dot4_i32_i8` on gfx906) — Q is quantized to Q8_0
// in LDS once per kernel; K is read as packed int32s (4 int8 each)
// and dotted with Q via `__builtin_amdgcn_sdot4`. Scalar dequant
// `K.d × Q.d` happens ONCE per QK8_0 block (32 elements), at the
// end of each block's dp4a chain. V path stays on FP16 dequant
// per element (Phase 5 covers PV in INT8 — INT-FlashAttention).
// Reference: `vec_dot_fattn_vec_KQ_q8_0` in
// `/artefact/llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh:264`.
// Layout:
// k_cache / v_cache: [n_tokens, n_heads_kv, head_dim/32] block_q8_0
// q: [n_heads_q, head_dim] FP16
// out: [n_heads_q, head_dim] FP16
// Supported head_dim: {64, 128, 256, 512}. block.x = head_dim / 4
// threads (= 16, 32, 64, 128). At d≤256 the block is one wave64 and
// all reductions use free `__shfl_xor`. At d=512 the block is 2 waves;
// the in-wave shfl_xor sum-of-blocks reduction is followed by a tiny
// cross-wave LDS rendezvous (`score_parts[2]` + 2 `__syncthreads`).
// Each thread owns 4 int8 K-bytes + 4 fp32 V-accumulators + 4 int8
// Q-bytes from LDS, regardless of head_dim.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

#define ATTN_Q8DP_MAX_HEAD_DIM 512
#define ATTN_Q8DP_MAX_WAVES (ATTN_Q8DP_MAX_HEAD_DIM / 256)

extern "C" __global__ void flambeau_attention_decode_q8_kv(
    const fb_fp16_t* __restrict__ q,                    // [n_heads_q, head_dim]
    const flambeau_block_q8_0* __restrict__ k_cache,    // [n_tokens, n_heads_kv, head_dim/32]
    const flambeau_block_q8_0* __restrict__ v_cache,    // [n_tokens, n_heads_kv, head_dim/32]
    fb_fp16_t* __restrict__ out,                        // [n_heads_q, head_dim]
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,                                 // 64, 128, 256 or 512
    const int n_tokens,
    const float scale,
    const int window_size                               // 0 = unbounded causal; >0 = SWA
) {
    const int q_head  = blockIdx.x;
    if (q_head >= n_heads_q) return;
    const int group   = n_heads_q / n_heads_kv;
    const int kv_head = q_head / group;

    // Each thread owns 4 elements of the head_dim vector (packed int8 → int32
    // for K-side dp4a, 4× fp32 for V-side accumulation).
    const int tid          = threadIdx.x;            // 0..head_dim/4 - 1
    const int dim_quad     = tid;                    // index in (head_dim/4)
    const int elem_base    = tid * 4;                // first dim element this thread owns
    const int n_blocks_per_row = head_dim / 32;
    const int q8_block_idx = elem_base / 32;         // which Q8_0 block this thread covers
    const int n_quads_per_block = 8;                 // 32 / 4
    const int quad_in_block = (elem_base % 32) / 4;  // 0..7

    // 1. Q → Q8_0 in LDS once per kernel. Symmetric (no s field needed —
    // K is also Q8_0 so the sum-correction term in Q8_1 cancels out).
    // Each thread writes 4 int8 quants into shared.
    __shared__ int8_t  q_qs[ATTN_Q8DP_MAX_HEAD_DIM];
    __shared__ float   q_d_block[ATTN_Q8DP_MAX_HEAD_DIM / 32];

    // Load Q[q_head, elem_base..elem_base+4] as fp32.
    float qv[4];
    qv[0] = (float) q[(size_t) q_head * head_dim + elem_base + 0];
    qv[1] = (float) q[(size_t) q_head * head_dim + elem_base + 1];
    qv[2] = (float) q[(size_t) q_head * head_dim + elem_base + 2];
    qv[3] = (float) q[(size_t) q_head * head_dim + elem_base + 3];

    // amax over the 8 threads of one Q8_0 block. Each thread starts with
    // amax of its 4 elements; reduce across the 8 quads via shfl_xor.
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

    // 2. Online-softmax stats; each thread accumulates its 4 V-out elements.
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float v_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    __syncthreads();

    // Cache the int32-packed Q for our quad — used every t.
    const int* q_qs_int = (const int*) q_qs;
    const int q_packed  = q_qs_int[dim_quad];

    // SWA: query position is the last token in the cache (n_tokens-1).
    // Keys older than (qpos - window_size + 1) are masked. window_size=0
    // disables the window — full causal range. Matches the F16 kernel's
    // SWA semantics.
    int t_start = 0;
    if (window_size > 0) {
        const int qpos = n_tokens - 1;
        t_start = qpos - window_size + 1;
        if (t_start < 0) t_start = 0;
    }
    for (int t = t_start; t < n_tokens; ++t) {
        const size_t kv_row_blocks =
            ((size_t) t * n_heads_kv + kv_head) * n_blocks_per_row;

        // 3. K-block load: this thread reads its 4 int8 K bytes as one int32.
        const flambeau_block_q8_0* k_block = k_cache + kv_row_blocks + q8_block_idx;
        const int  k_packed = ((const int*) k_block->qs)[quad_in_block];
        const float k_d     = (float) k_block->d;

        // 4. dp4a: 4×int8 K · 4×int8 Q → int32 partial.
        const int sumi_thread = gfx906_dp4a(k_packed, q_packed, 0);

        // 5. Reduce sumi across the 8 quads of one Q8_0 block via shfl_xor.
        int sumi_block = sumi_thread;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            sumi_block += __shfl_xor(sumi_block, off, n_quads_per_block);
        }
        // sumi_block is now the int32 dot for the whole 32-element Q8_0
        // block (replicated across all 8 quads of the block).

        // 6. Scalar dequant: ONE FMA per Q8_0 block (instead of one per
        // element in the FP16-dequant variant).
        const float qd_block = q_d_block[q8_block_idx];
        const float block_score = k_d * qd_block * (float) sumi_block;

        // 7. Sum block_scores across the n_blocks_per_row blocks → final
        // score for this (q_token, q_head, t). Only quad_in_block==0
        // holds a unique value per block; we shuffle them across the
        // rest of the wave and sum.
        float score_t = block_score;
        // First, broadcast lane 0 of each 8-lane group to all lanes of
        // that group (already done by the earlier reduction above; all
        // 8 lanes of one block hold the same block_score).
        // Now reduce across BLOCKS: stride n_quads_per_block (= 8) across
        // up to head_dim/4 lanes. n_blocks_per_row = 1, 2, 4 or 8.
        if (n_blocks_per_row >= 2) {
            // After this xor=8, lanes (0..7) and lanes (8..15) all see
            // sum of blocks 0+1.
            score_t += __shfl_xor(score_t, 8, min(n_blocks_per_row * n_quads_per_block, 64));
        }
        if (n_blocks_per_row >= 4) {
            score_t += __shfl_xor(score_t, 16, min(n_blocks_per_row * n_quads_per_block, 64));
        }
        if (n_blocks_per_row >= 8) {
            score_t += __shfl_xor(score_t, 32, min(n_blocks_per_row * n_quads_per_block, 64));
        }
        // d=512: 2 waves per block; the in-wave xor chain above ends at
        // stride 32 (wave-internal). Add a cross-wave LDS reduce so all
        // lanes see the full sum across both waves.
        if (head_dim > 256) {
            __shared__ float score_parts[ATTN_Q8DP_MAX_WAVES];
            const int warp = tid >> 6;
            const int lane = tid & 63;
            if (lane == 0) score_parts[warp] = score_t;
            __syncthreads();
            score_t = score_parts[0] + score_parts[1];
            __syncthreads();
        }
        score_t *= scale;

        // 8. Online softmax update.
        float new_max   = fmaxf(running_max, score_t);
        float scale_old = __expf(running_max - new_max);
        float coeff_t   = __expf(score_t - new_max);

        // 9. V-accum: dequant V on the fly. Each thread reads its 4 int8
        // V-bytes from the same Q8_0 block as its K (they share row
        // layout). Multiply by V's d and coeff_t, fold into v_out[4].
        const flambeau_block_q8_0* v_block = v_cache + kv_row_blocks + q8_block_idx;
        const float v_d = (float) v_block->d;
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

    // 10. Final write: each thread emits its 4 elements.
    const float norm = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    out[(size_t) q_head * head_dim + elem_base + 0] = (fb_fp16_t) (v_out[0] * norm);
    out[(size_t) q_head * head_dim + elem_base + 1] = (fb_fp16_t) (v_out[1] * norm);
    out[(size_t) q_head * head_dim + elem_base + 2] = (fb_fp16_t) (v_out[2] * norm);
    out[(size_t) q_head * head_dim + elem_base + 3] = (fb_fp16_t) (v_out[3] * norm);
}
