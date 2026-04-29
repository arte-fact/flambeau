// attention_decode_bf16 — GQA decode attention, BF16 Q/KV/out.
//
// MTP-4-C-4: BF16 sibling of `attention_decode_f16`. Identical math
// (online flash-attn-v2 softmax, F32 internal accumulators) — only
// the storage dtype changes. Used by the MTP BF16 forward path so
// Q / K / V / out stay in BF16 across the matmul-attention chain
// (no F16/Q8_1 hops between mmvq output and attention input).
//
// gfx906 has no native BF16 arithmetic — we lift each load to F32
// via `fb_bf16_to_f32` (lossless bit-shift) and round the output
// via `fb_f32_to_bf16` (RNE).
//
// Shapes:
//   q       : [n_heads_q, head_dim]            BF16
//   k_cache : [n_tokens, n_heads_kv, head_dim] BF16 contiguous
//   v_cache : same shape as k                  BF16
//   out     : [n_heads_q, head_dim]            BF16
//
// Launch (caller-provided):
//   blockDim = head_dim   (one thread per output lane)
//   gridDim  = n_heads_q
//   shared   = q[256] + out[256] + score_parts[4] = 2064 bytes
//
// Supports `head_dim ∈ {64, 128, 256}` — the same range the F16
// kernel covers; MTP uses 256.

#include "block_quant.cuh"

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

#define ATTN_BF16_MAX_HEAD_DIM 256
#define ATTN_BF16_MAX_WARPS (ATTN_BF16_MAX_HEAD_DIM / 64)

extern "C" __global__ void flambeau_attention_decode_bf16(
    const fb_bf16_t* __restrict__ q,          // [n_heads_q, head_dim]
    const fb_bf16_t* __restrict__ k_cache,    // [n_tokens, n_heads_kv, head_dim]
    const fb_bf16_t* __restrict__ v_cache,    // [n_tokens, n_heads_kv, head_dim]
    fb_bf16_t* __restrict__ out,              // [n_heads_q, head_dim]
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,                       // 64, 128, or 256
    const int n_tokens,
    const float scale                         // 1 / sqrt(head_dim)
) {
    const int q_head = blockIdx.x;
    if (q_head >= n_heads_q) return;
    const int group   = n_heads_q / n_heads_kv;
    const int kv_head = q_head / group;

    const int tid     = threadIdx.x;
    const int warp    = tid >> 6;
    const int lane    = tid & 63;
    const int n_warps = blockDim.x >> 6;

    // --- Load Q for this head ---
    __shared__ float q_shared[ATTN_BF16_MAX_HEAD_DIM];
    if (tid < head_dim) {
        q_shared[tid] = fb_bf16_to_f32(q[(size_t) q_head * head_dim + tid]);
    }

    // --- Running stats (per-block, replicated across threads) ---
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    __shared__ float out_shared[ATTN_BF16_MAX_HEAD_DIM];
    if (tid < head_dim) {
        out_shared[tid] = 0.0f;
    }
    __shared__ float score_parts[ATTN_BF16_MAX_WARPS];
    __syncthreads();

    // --- Inner loop over context positions ---
    for (int t = 0; t < n_tokens; ++t) {
        const size_t kv_row = ((size_t) t * n_heads_kv + kv_head) * head_dim;

        // 1. Q · K[t, kv_head] cooperatively — each thread owns one lane.
        float my_partial = 0.0f;
        if (tid < head_dim) {
            my_partial = q_shared[tid] * fb_bf16_to_f32(k_cache[kv_row + tid]);
        }
        // Warp reduce across 64 lanes.
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            my_partial += __shfl_xor(my_partial, off, 64);
        }
        // Cross-warp sum.
        if (lane == 0) {
            score_parts[warp] = my_partial;
        }
        __syncthreads();
        float score_t = 0.0f;
        #pragma unroll
        for (int w = 0; w < ATTN_BF16_MAX_WARPS; ++w) {
            if (w < n_warps) score_t += score_parts[w];
        }
        score_t *= scale;

        // 2. Online softmax update.
        const float new_max   = fmaxf(running_max, score_t);
        const float scale_old = __expf(running_max - new_max);
        const float coeff_t   = __expf(score_t - new_max);

        // 3. Per-element update of running_out.
        if (tid < head_dim) {
            const float v_t = fb_bf16_to_f32(v_cache[kv_row + tid]);
            out_shared[tid] = out_shared[tid] * scale_old + coeff_t * v_t;
        }

        running_sum = running_sum * scale_old + coeff_t;
        running_max = new_max;

        __syncthreads();
    }

    // --- Normalise & write ---
    if (tid < head_dim) {
        const float norm = 1.0f / running_sum;
        out[(size_t) q_head * head_dim + tid] =
            fb_f32_to_bf16(out_shared[tid] * norm);
    }
}
