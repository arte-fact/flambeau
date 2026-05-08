// attention_prefill_flash_tile_q8_kv — BR=4/8 LDS-tiled flash-attention
// v2 prefill kernel with Q8_0 KV cache. Mirror of `attention_prefill_flash_tile_f16`
// with the cooperative K/V load path changed to dequantize Q8_0 blocks
// on the way into the F32 LDS tile.
// Closes the prefill regression where the
// F16 path used BR=8 LDS-tiled prefill while Q8 was stuck on the oracle
// (one block per (q_token, q_head)). Both kernels now share structure;
// only the load path differs (Q8 → F32 dequant vs F16 → F32 cast).
// Score loop and online-softmax rescale are identical.
// Why the LDS tile stays F32 (not int8): the original f16 flash-tile
// upcasts to F32 on load, which is necessary for the online-softmax
// math anyway. Keeping LDS as F32 lets the inner Q·K dot stay
// register-fp32 (1 mul + 1 add per element, warp-reduced) — same as
// the F16 variant. dp4a in the score-path here is harder than in
// decode because each warp handles ONE Q row × BC=16 K rows; the
// per-K-row reduction across 64 lanes already uses warp reductions
// efficiently. The win vs the oracle path is from LDS reuse (each K
// element is loaded once per chunk and reused by BR=4-8 Q rows in the
// tile), not from dp4a.
// Shape (matches the F16 variant exactly):
// Q: [n_q_tokens, n_heads_q, head_dim] F16
// K/V: [n_k_tokens, n_heads_kv, head_dim/32] block_q8_0
// Out: [n_q_tokens, n_heads_q, head_dim] F16
// Causal: Q at global pos (q_offset + q_idx) attends to K[0..q_offset+q_idx].
// Launch (per template instantiation):
// grid = (ceil(n_q_tokens / BR), n_heads_q, 1)
// block = (WARP_SIZE=64, BR, 1) — 256 or 512 threads, 2D block
// LDS: 2 × BC × D floats. Same per-D budget as F16 variant.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "block_quant.cuh"
#include "gfx906.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

template <int D, int BR, int BC>
static __device__ __forceinline__ void flash_attn_prefill_v2_q8_impl(
    const fb_fp16_t*           __restrict__ q,        // [n_q_tokens, n_heads_q, D]
    const flambeau_block_q8_0* __restrict__ k_cache,  // [n_k_tokens, n_heads_kv, D/32]
    const flambeau_block_q8_0* __restrict__ v_cache,  // [n_k_tokens, n_heads_kv, D/32]
    fb_fp16_t*                 __restrict__ out,      // [n_q_tokens, n_heads_q, D]
    const int n_q_tokens,
    const int n_heads_q,
    const int n_heads_kv,
    const int n_k_tokens,
    const int q_offset,
    const float scale
) {
    static_assert(D == 64 || D == 128 || D == 256, "D must be 64, 128, or 256");
    static_assert(D % WARP_SIZE == 0, "D must be a multiple of WARP_SIZE");
    static_assert(D % 32 == 0, "D must be a multiple of QK8_0 (32)");

    constexpr int D_PER_LANE       = D / WARP_SIZE;
    constexpr int THREADS_PER_BLOCK = BR * WARP_SIZE;
    constexpr int BLOCKS_PER_ROW   = D / 32;   // # of Q8_0 blocks per (token, kv_head) row

    const int q_tile = blockIdx.x;
    const int h_q    = blockIdx.y;
    const int lane   = threadIdx.x;   // 0..63 — D-axis
    const int warp   = threadIdx.y;   // 0..BR-1 — Q-row axis
    const int tid    = warp * WARP_SIZE + lane;

    const int q_idx  = q_tile * BR + warp;
    const bool q_in_range = (q_idx < n_q_tokens);

    const int n_rep = n_heads_q / n_heads_kv;
    const int h_kv  = (n_rep > 1) ? (h_q / n_rep) : h_q;

    int limit = q_offset + q_idx + 1;
    if (limit > n_k_tokens) limit = n_k_tokens;

    const int last_q_idx = min(q_tile * BR + BR - 1, n_q_tokens - 1);
    int block_limit = q_offset + last_q_idx + 1;
    if (block_limit > n_k_tokens) block_limit = n_k_tokens;

    // ---- Per-lane register state (identical to F16 variant) ----
    float q_reg[D_PER_LANE];
    float o_reg[D_PER_LANE];
    #pragma unroll
    for (int i = 0; i < D_PER_LANE; ++i) {
        if (q_in_range) {
            const size_t off = ((size_t) q_idx * n_heads_q + h_q) * D
                             + lane + i * WARP_SIZE;
            q_reg[i] = (float) q[off];
        } else {
            q_reg[i] = 0.0f;
        }
        o_reg[i] = 0.0f;
    }
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // ---- Shared LDS K/V tile (F32, same size as F16 variant) ----
    __shared__ float k_lds[BC * D];
    __shared__ float v_lds[BC * D];

    const int n_chunks = (block_limit + BC - 1) / BC;
    const int tile_elems = BC * D;
    const int loads_per_thread = (tile_elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int k_start = chunk * BC;

        // Cooperative Q8 → F32 dequant load. Each thread handles
        // `loads_per_thread` linear positions in the BC × D tile. For each
        // position we read ONE int8 from the appropriate Q8_0 block and
        // ONE FP16 scale (broadcast across 32 elements).
        #pragma unroll
        for (int il = 0; il < loads_per_thread; ++il) {
            const int idx = tid + il * THREADS_PER_BLOCK;
            if (idx < tile_elems) {
                const int j   = idx / D;
                const int d   = idx % D;
                const int row = k_start + j;
                const bool valid = (row < block_limit);
                if (valid) {
                    const int blk_idx    = d / 32;
                    const int blk_off    = d & 31;
                    const size_t row_blocks =
                        ((size_t) row * n_heads_kv + h_kv) * BLOCKS_PER_ROW;
                    const flambeau_block_q8_0* k_block =
                        k_cache + row_blocks + blk_idx;
                    const flambeau_block_q8_0* v_block =
                        v_cache + row_blocks + blk_idx;
                    const float k_d = (float) k_block->d;
                    const float v_d = (float) v_block->d;
                    const int   k_q = (int)   k_block->qs[blk_off];
                    const int   v_q = (int)   v_block->qs[blk_off];
                    k_lds[idx] = k_d * (float) k_q;
                    v_lds[idx] = v_d * (float) v_q;
                } else {
                    k_lds[idx] = 0.0f;
                    v_lds[idx] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (q_in_range) {
            #pragma unroll 4
            for (int j = 0; j < BC; ++j) {
                const int row = k_start + j;

                float partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < D_PER_LANE; ++i) {
                    partial += q_reg[i] * k_lds[j * D + lane + i * WARP_SIZE];
                }
                float s_j = gfx906_warp_reduce_sum(partial) * scale;

                if (row >= limit) {
                    s_j = -INFINITY;
                }

                const float m_new = fmaxf(m_i, s_j);
                const float alpha = gfx906_fast_exp(m_i - m_new);
                const float p     = gfx906_fast_exp(s_j - m_new);

                #pragma unroll
                for (int i = 0; i < D_PER_LANE; ++i) {
                    o_reg[i] = alpha * o_reg[i]
                             + p * v_lds[j * D + lane + i * WARP_SIZE];
                }
                l_i = alpha * l_i + p;
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (q_in_range) {
        const float inv_l = gfx906_rcp(l_i);
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; ++i) {
            const size_t off = ((size_t) q_idx * n_heads_q + h_q) * D
                             + lane + i * WARP_SIZE;
            out[off] = (fb_fp16_t) (o_reg[i] * inv_l);
        }
    }
}

// ---- Extern "C" wrappers (one per supported head_dim, matching F16) ----

extern "C" __global__ __launch_bounds__(256, 2)
void flambeau_attention_prefill_flash_tile_d64_q8_kv(
    const fb_fp16_t*           __restrict__ q,
    const flambeau_block_q8_0* __restrict__ k_cache,
    const flambeau_block_q8_0* __restrict__ v_cache,
    fb_fp16_t*                 __restrict__ out,
    const int n_q_tokens,
    const int n_heads_q,
    const int n_heads_kv,
    const int n_k_tokens,
    const int q_offset,
    const float scale
) {
    flash_attn_prefill_v2_q8_impl</*D=*/64, /*BR=*/4, /*BC=*/64>(
        q, k_cache, v_cache, out,
        n_q_tokens, n_heads_q, n_heads_kv,
        n_k_tokens, q_offset, scale);
}

extern "C" __global__ __launch_bounds__(256, 2)
void flambeau_attention_prefill_flash_tile_d128_q8_kv(
    const fb_fp16_t*           __restrict__ q,
    const flambeau_block_q8_0* __restrict__ k_cache,
    const flambeau_block_q8_0* __restrict__ v_cache,
    fb_fp16_t*                 __restrict__ out,
    const int n_q_tokens,
    const int n_heads_q,
    const int n_heads_kv,
    const int n_k_tokens,
    const int q_offset,
    const float scale
) {
    flash_attn_prefill_v2_q8_impl</*D=*/128, /*BR=*/4, /*BC=*/32>(
        q, k_cache, v_cache, out,
        n_q_tokens, n_heads_q, n_heads_kv,
        n_k_tokens, q_offset, scale);
}

extern "C" __global__ __launch_bounds__(256, 2)
void flambeau_attention_prefill_flash_tile_d256_q8_kv(
    const fb_fp16_t*           __restrict__ q,
    const flambeau_block_q8_0* __restrict__ k_cache,
    const flambeau_block_q8_0* __restrict__ v_cache,
    fb_fp16_t*                 __restrict__ out,
    const int n_q_tokens,
    const int n_heads_q,
    const int n_heads_kv,
    const int n_k_tokens,
    const int q_offset,
    const float scale
) {
    flash_attn_prefill_v2_q8_impl</*D=*/256, /*BR=*/4, /*BC=*/16>(
        q, k_cache, v_cache, out,
        n_q_tokens, n_heads_q, n_heads_kv,
        n_k_tokens, q_offset, scale);
}

extern "C" __global__ __launch_bounds__(512, 1)
void flambeau_attention_prefill_flash_tile_d256_br8_q8_kv(
    const fb_fp16_t*           __restrict__ q,
    const flambeau_block_q8_0* __restrict__ k_cache,
    const flambeau_block_q8_0* __restrict__ v_cache,
    fb_fp16_t*                 __restrict__ out,
    const int n_q_tokens,
    const int n_heads_q,
    const int n_heads_kv,
    const int n_k_tokens,
    const int q_offset,
    const float scale
) {
    flash_attn_prefill_v2_q8_impl</*D=*/256, /*BR=*/8, /*BC=*/16>(
        q, k_cache, v_cache, out,
        n_q_tokens, n_heads_q, n_heads_kv,
        n_k_tokens, q_offset, scale);
}
