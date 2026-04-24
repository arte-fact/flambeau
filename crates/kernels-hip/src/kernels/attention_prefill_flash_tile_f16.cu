// attention_prefill_flash_tile_f16 — BR=4 LDS-tiled flash-attention v2
// prefill kernel, F16 I/O, F32 internal math, gfx906-tuned.
//
// V2.2.d fix 4 port of candle's `flash_attn_v2_fwd_d{64,128,256}_f32`
// (/artefact/candle/candle-hip-kernels/src/flash_attn_v2.cu:344-440),
// adapted to:
//   - F16 inputs / outputs (candle's source takes F32).
//   - Internal causal masking from q_offset (no additive mask tensor).
//   - Single-batch (B=1) contiguous layout matching our KvCache<F16Contig>.
//
// Profile motivation (2026-04-22, post-fix-3): `attention_prefill_f16`
// ran at 2994 µs / call (5× slower per-call than turbo's flash_attn_tile
// at ~600 µs). The old kernel issues one block per (q_token, q_head) and
// serially scans all K/V rows — each K/V row re-read from global once
// per Q row. This kernel replaces it with BR=4 Q rows per block, all 4
// warps sharing one BC-row K/V LDS tile → 4× global traffic reduction,
// plus online softmax so V also streams through LDS only once.
//
// Shape:
//   Q:     [n_q_tokens, n_heads_q, head_dim]       F16, row-major
//   K:     [n_k_tokens, n_heads_kv, head_dim]      F16, row-major (flambeau KV layout)
//   V:     [n_k_tokens, n_heads_kv, head_dim]      F16, row-major
//   Out:   [n_q_tokens, n_heads_q, head_dim]       F16, row-major
//
// GQA: `n_heads_q / n_heads_kv = group`. Kernel computes kv_head =
// q_head / group.
//
// Causal: Q token at global position (q_offset + q_idx) attends to K
// positions [0 .. q_offset + q_idx]. K rows beyond that contribute
// −∞ to the softmax.
//
// Launch:
//   grid  = (ceil(n_q_tokens / BR=4), n_heads_q, 1)
//   block = (WARP_SIZE=64, BR=4, 1)          — 256 threads, 2D block
//
// LDS: 2 × BC × D floats; chosen per-D (see wrappers below).
//   D=64,  BC=64 → 32 KiB
//   D=128, BC=32 → 32 KiB
//   D=256, BC=16 → 32 KiB

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "gfx906.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

typedef _Float16 fb_fp16_t;

template <int D, int BR, int BC>
static __device__ __forceinline__ void flash_attn_prefill_v2_impl(
    const fb_fp16_t* __restrict__ q,        // [n_q_tokens, n_heads_q, D]
    const fb_fp16_t* __restrict__ k_cache,  // [n_k_tokens, n_heads_kv, D]
    const fb_fp16_t* __restrict__ v_cache,  // [n_k_tokens, n_heads_kv, D]
    fb_fp16_t*       __restrict__ out,      // [n_q_tokens, n_heads_q, D]
    const int n_q_tokens,
    const int n_heads_q,
    const int n_heads_kv,
    const int n_k_tokens,
    const int q_offset,
    const float scale
) {
    static_assert(D == 64 || D == 128 || D == 256, "D must be 64, 128, or 256");
    static_assert(D % WARP_SIZE == 0, "D must be a multiple of WARP_SIZE");

    constexpr int D_PER_LANE = D / WARP_SIZE;
    constexpr int THREADS_PER_BLOCK = BR * WARP_SIZE;

    const int q_tile = blockIdx.x;
    const int h_q    = blockIdx.y;
    const int lane   = threadIdx.x;   // 0..63  — D-axis
    const int warp   = threadIdx.y;   // 0..BR-1 — Q-row axis
    const int tid    = warp * WARP_SIZE + lane;

    const int q_idx  = q_tile * BR + warp;
    const bool q_in_range = (q_idx < n_q_tokens);

    const int n_rep = n_heads_q / n_heads_kv;
    const int h_kv  = (n_rep > 1) ? (h_q / n_rep) : h_q;

    // Per-warp causal cutoff: Q[q_idx] attends to K[0..limit).
    int limit = q_offset + q_idx + 1;
    if (limit > n_k_tokens) limit = n_k_tokens;

    // Block-wide cutoff for the cooperative LDS tile load — all 4 warps
    // share one tile, so we must load enough rows for the LATEST q_idx
    // in this tile (warp = BR - 1 has the highest q_idx). If that warp
    // is past the end of n_q_tokens (last partial tile), cap at the last
    // valid q_idx's limit.
    const int last_q_idx = min(q_tile * BR + BR - 1, n_q_tokens - 1);
    int block_limit = q_offset + last_q_idx + 1;
    if (block_limit > n_k_tokens) block_limit = n_k_tokens;

    // ---- Per-lane register state ----
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

    // ---- Shared LDS K/V tile (F32) ----
    __shared__ float k_lds[BC * D];
    __shared__ float v_lds[BC * D];

    // Number of K chunks sized by block_limit (the cooperative-load's
    // upper bound), not the per-warp limit. Inner loop applies the
    // per-warp causal mask via `row >= limit`.
    const int n_chunks = (block_limit + BC - 1) / BC;
    const int tile_elems = BC * D;
    const int loads_per_thread = (tile_elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int k_start = chunk * BC;

        // Cooperative load of BC × D K/V entries into LDS. Upcast F16 → F32
        // at load time. Zero-pad rows beyond `block_limit` (the block's
        // shared upper bound — rows past individual warps' `limit` stay
        // real in LDS so other warps see them; the causal mask below
        // rejects them via `row >= limit` per-warp).
        #pragma unroll
        for (int il = 0; il < loads_per_thread; ++il) {
            const int idx = tid + il * THREADS_PER_BLOCK;
            if (idx < tile_elems) {
                const int j  = idx / D;
                const int d  = idx % D;
                const int row = k_start + j;
                const bool valid = (row < block_limit);
                if (valid) {
                    const size_t kv_off =
                        ((size_t) row * n_heads_kv + h_kv) * D + d;
                    k_lds[idx] = (float) k_cache[kv_off];
                    v_lds[idx] = (float) v_cache[kv_off];
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

                // Q · K[j] — each lane contributes D_PER_LANE terms, warp-reduce
                float partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < D_PER_LANE; ++i) {
                    partial += q_reg[i] * k_lds[j * D + lane + i * WARP_SIZE];
                }
                float s_j = gfx906_warp_reduce_sum(partial) * scale;

                // Mask K rows beyond the causal cutoff.
                if (row >= limit) {
                    s_j = -INFINITY;
                }

                // Online softmax rescale (Dao et al. flash-attention v1).
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

// ---- Extern "C" wrappers (one per supported head dim) ----

extern "C" __global__ __launch_bounds__(256, 2)
void flambeau_attention_prefill_flash_tile_d64_f16(
    const fb_fp16_t* __restrict__ q,
    const fb_fp16_t* __restrict__ k_cache,
    const fb_fp16_t* __restrict__ v_cache,
    fb_fp16_t*       __restrict__ out,
    const int n_q_tokens,
    const int n_heads_q,
    const int n_heads_kv,
    const int n_k_tokens,
    const int q_offset,
    const float scale
) {
    flash_attn_prefill_v2_impl</*D=*/64, /*BR=*/4, /*BC=*/64>(
        q, k_cache, v_cache, out,
        n_q_tokens, n_heads_q, n_heads_kv,
        n_k_tokens, q_offset, scale);
}

extern "C" __global__ __launch_bounds__(256, 2)
void flambeau_attention_prefill_flash_tile_d128_f16(
    const fb_fp16_t* __restrict__ q,
    const fb_fp16_t* __restrict__ k_cache,
    const fb_fp16_t* __restrict__ v_cache,
    fb_fp16_t*       __restrict__ out,
    const int n_q_tokens,
    const int n_heads_q,
    const int n_heads_kv,
    const int n_k_tokens,
    const int q_offset,
    const float scale
) {
    flash_attn_prefill_v2_impl</*D=*/128, /*BR=*/4, /*BC=*/32>(
        q, k_cache, v_cache, out,
        n_q_tokens, n_heads_q, n_heads_kv,
        n_k_tokens, q_offset, scale);
}

extern "C" __global__ __launch_bounds__(256, 2)
void flambeau_attention_prefill_flash_tile_d256_f16(
    const fb_fp16_t* __restrict__ q,
    const fb_fp16_t* __restrict__ k_cache,
    const fb_fp16_t* __restrict__ v_cache,
    fb_fp16_t*       __restrict__ out,
    const int n_q_tokens,
    const int n_heads_q,
    const int n_heads_kv,
    const int n_k_tokens,
    const int q_offset,
    const float scale
) {
    flash_attn_prefill_v2_impl</*D=*/256, /*BR=*/4, /*BC=*/16>(
        q, k_cache, v_cache, out,
        n_q_tokens, n_heads_q, n_heads_kv,
        n_k_tokens, q_offset, scale);
}

// V2.29.b — BR=8 variant at D=256. Doubles Q rows per block → halves
// grid.x → fewer blocks, potentially better CU fill on large
// n_q_tokens. Block = 64 × 8 = 512 threads (hits launch_bounds at 2
// waves/SIMD → 1 wave/SIMD). LDS: 2 × 16 × 256 × 4 = 32 KiB (same as
// BR=4 variant, same BC=16 tile width).
extern "C" __global__ __launch_bounds__(512, 1)
void flambeau_attention_prefill_flash_tile_d256_br8_f16(
    const fb_fp16_t* __restrict__ q,
    const fb_fp16_t* __restrict__ k_cache,
    const fb_fp16_t* __restrict__ v_cache,
    fb_fp16_t*       __restrict__ out,
    const int n_q_tokens,
    const int n_heads_q,
    const int n_heads_kv,
    const int n_k_tokens,
    const int q_offset,
    const float scale
) {
    flash_attn_prefill_v2_impl</*D=*/256, /*BR=*/8, /*BC=*/16>(
        q, k_cache, v_cache, out,
        n_q_tokens, n_heads_q, n_heads_kv,
        n_k_tokens, q_offset, scale);
}
