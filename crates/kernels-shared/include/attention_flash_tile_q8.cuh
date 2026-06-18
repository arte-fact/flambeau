#pragma once

// Flash-attention-v2 prefill with a Q8_0 K/V cache (LDS-tiled, register-resident
// O). Sibling of attention_flash_tile.cuh: same block (WARP_SIZE, BR) + register
// tiling over the WARP_SIZE seam + neutral warp_reduce_sum / fast_exp / fast_rcp,
// but the cooperative K/V tile load dequants Q8_0 (single int8 + per-block fp16
// scale — the byte read is aligned, no load_int_b2 needed).
#ifndef FT_Q8_INF
#define FT_Q8_INF __builtin_huge_valf()
#endif

template <int D, int BR, int BC>
static __device__ __forceinline__ void flash_attn_prefill_v2_q8_impl(
    const fb_fp16_t* __restrict__ q,
    const flambeau_block_q8_0* __restrict__ k_cache,
    const flambeau_block_q8_0* __restrict__ v_cache,
    fb_fp16_t* __restrict__ out,
    const int n_q_tokens, const int n_heads_q, const int n_heads_kv,
    const int n_k_tokens, const int q_offset, const float scale,
    const int window_size, const int ring_depth)
{
    static_assert(D == 64 || D == 128 || D == 256, "D must be 64/128/256");
    static_assert(D % WARP_SIZE == 0, "D must be a multiple of WARP_SIZE");
    constexpr int D_PER_LANE = D / WARP_SIZE;
    constexpr int THREADS_PER_BLOCK = BR * WARP_SIZE;
    constexpr int BLOCKS_PER_ROW = D / 32;

    const int q_tile = blockIdx.x;
    const int h_q = blockIdx.y;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * WARP_SIZE + lane;

    const int q_idx = q_tile * BR + warp;
    const bool q_in_range = (q_idx < n_q_tokens);
    const int n_rep = n_heads_q / n_heads_kv;
    const int h_kv = (n_rep > 1) ? (h_q / n_rep) : h_q;

    int limit = q_offset + q_idx + 1;
    if (limit > n_k_tokens) limit = n_k_tokens;
    int t_start = 0;
    if (window_size > 0) { t_start = q_offset + q_idx - window_size + 1; if (t_start < 0) t_start = 0; }

    const int last_q_idx = min(q_tile * BR + BR - 1, n_q_tokens - 1);
    int block_limit = q_offset + last_q_idx + 1;
    if (block_limit > n_k_tokens) block_limit = n_k_tokens;

    float q_reg[D_PER_LANE];
    float o_reg[D_PER_LANE];
#pragma unroll
    for (int i = 0; i < D_PER_LANE; ++i) {
        if (q_in_range) {
            q_reg[i] = (float) q[((size_t) q_idx * n_heads_q + h_q) * D + lane + i * WARP_SIZE];
        } else {
            q_reg[i] = 0.0f;
        }
        o_reg[i] = 0.0f;
    }
    float m_i = -FT_Q8_INF;
    float l_i = 0.0f;

    __shared__ float k_lds[BC * D];
    __shared__ float v_lds[BC * D];

    const int n_chunks = (block_limit + BC - 1) / BC;
    const int tile_elems = BC * D;
    const int loads_per_thread = (tile_elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int min_t_start = 0;
    if (window_size > 0) {
        min_t_start = q_tile * BR + q_offset - window_size + 1;
        if (min_t_start < 0) min_t_start = 0;
    }
    const int first_active_chunk = min_t_start / BC;

    for (int chunk = first_active_chunk; chunk < n_chunks; ++chunk) {
        const int k_start = chunk * BC;
#pragma unroll
        for (int il = 0; il < loads_per_thread; ++il) {
            const int idx = tid + il * THREADS_PER_BLOCK;
            if (idx < tile_elems) {
                const int j = idx / D;
                const int d = idx % D;
                const int row = k_start + j;
                if (row < block_limit) {
                    const int blk_idx = d / 32;
                    const int blk_off = d & 31;
                    const int row_phys = (ring_depth > 0) ? (row % ring_depth) : row;
                    const size_t row_blocks = ((size_t) row_phys * n_heads_kv + h_kv) * BLOCKS_PER_ROW;
                    const flambeau_block_q8_0* k_block = k_cache + row_blocks + blk_idx;
                    const flambeau_block_q8_0* v_block = v_cache + row_blocks + blk_idx;
                    k_lds[idx] = (float) k_block->d * (float) (int) k_block->qs[blk_off];
                    v_lds[idx] = (float) v_block->d * (float) (int) v_block->qs[blk_off];
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
                const float s_j = warp_reduce_sum(partial) * scale;
                const bool masked = (row >= limit) || (row < t_start);
                if (!masked) {
                    const float m_new = fmaxf(m_i, s_j);
                    const float alpha = fast_exp(m_i - m_new);
                    const float p = fast_exp(s_j - m_new);
#pragma unroll
                    for (int i = 0; i < D_PER_LANE; ++i) {
                        o_reg[i] = alpha * o_reg[i] + p * v_lds[j * D + lane + i * WARP_SIZE];
                    }
                    l_i = alpha * l_i + p;
                    m_i = m_new;
                }
            }
        }
        __syncthreads();
    }

    if (q_in_range) {
        const float inv_l = fast_rcp(l_i);
#pragma unroll
        for (int i = 0; i < D_PER_LANE; ++i) {
            out[((size_t) q_idx * n_heads_q + h_q) * D + lane + i * WARP_SIZE] = (fb_fp16_t) (o_reg[i] * inv_l);
        }
    }
}
