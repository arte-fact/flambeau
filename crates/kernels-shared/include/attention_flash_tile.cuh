#pragma once

// Flash-attention-v2 prefill (LDS-tiled, register-resident O). Block =
// (WARP_SIZE, BR): each warp owns one Q row, lanes hold D/WARP_SIZE elements of
// the D axis. Backend-neutral: WARP_SIZE, warp_reduce_sum, fast_exp, fast_rcp
// come from the arch seam, so the register tiling + the K·Q reduce scale to
// wave64 and wave32 (the only divergence is sum partition → within the cert
// envelope; no bit-identity contract, unlike attention_decode).
#ifndef FLASH_TILE_INF
#define FLASH_TILE_INF __builtin_huge_valf()
#endif

template <int D, int BR, int BC>
static __device__ __forceinline__ void flash_attn_prefill_v2_impl(
    const fb_fp16_t* __restrict__ q,
    const fb_fp16_t* __restrict__ k_cache,
    const fb_fp16_t* __restrict__ v_cache,
    fb_fp16_t* __restrict__ out,
    const int n_q_tokens, const int n_heads_q, const int n_heads_kv,
    const int n_k_tokens, const int q_offset, const float scale,
    const int window_size, const int ring_depth)
{
    static_assert(D == 64 || D == 128 || D == 256 || D == 512, "D must be 64/128/256/512");
    static_assert(D % WARP_SIZE == 0, "D must be a multiple of WARP_SIZE");
    constexpr int D_PER_LANE = D / WARP_SIZE;
    constexpr int THREADS_PER_BLOCK = BR * WARP_SIZE;

    const int q_tile = blockIdx.x;
    const int h_q = blockIdx.y;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * WARP_SIZE + lane;

    const int q_idx = q_tile * BR + warp;
    const bool q_in_range = (q_idx < n_q_tokens);
    const int n_rep = n_heads_q / n_heads_kv;
    const int h_kv = (n_rep > 1) ? (h_q / n_rep) : h_q;

    const int qpos_global = q_offset + q_idx;
    int limit = qpos_global + 1;
    if (limit > n_k_tokens) limit = n_k_tokens;
    int swa_min = 0;
    if (window_size > 0) { swa_min = qpos_global - window_size + 1; if (swa_min < 0) swa_min = 0; }

    const int last_q_idx = min(q_tile * BR + BR - 1, n_q_tokens - 1);
    int block_limit = q_offset + last_q_idx + 1;
    if (block_limit > n_k_tokens) block_limit = n_k_tokens;
    int block_swa_min = 0;
    if (window_size > 0) {
        const int first_q_idx = q_tile * BR;
        block_swa_min = (q_offset + first_q_idx) - window_size + 1;
        if (block_swa_min < 0) block_swa_min = 0;
    }

    float q_reg[D_PER_LANE];
    float o_reg[D_PER_LANE];
#pragma unroll
    for (int i = 0; i < D_PER_LANE; ++i) {
        if (q_in_range) {
            const size_t off = ((size_t) q_idx * n_heads_q + h_q) * D + lane + i * WARP_SIZE;
            q_reg[i] = (float) q[off];
        } else {
            q_reg[i] = 0.0f;
        }
        o_reg[i] = 0.0f;
    }
    float m_i = -FLASH_TILE_INF;
    float l_i = 0.0f;

    __shared__ float k_lds[BC * D];
    __shared__ float v_lds[BC * D];

    const int n_chunks_end = (block_limit + BC - 1) / BC;
    const int n_chunks_start = block_swa_min / BC;
    const int tile_elems = BC * D;
    const int loads_per_thread = (tile_elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int chunk = n_chunks_start; chunk < n_chunks_end; ++chunk) {
        const int k_start = chunk * BC;
#pragma unroll
        for (int il = 0; il < loads_per_thread; ++il) {
            const int idx = tid + il * THREADS_PER_BLOCK;
            if (idx < tile_elems) {
                const int j = idx / D;
                const int d = idx % D;
                const int row = k_start + j;
                if (row < block_limit) {
                    const int row_phys = (ring_depth > 0) ? (row % ring_depth) : row;
                    const size_t kv_off = ((size_t) row_phys * n_heads_kv + h_kv) * D + d;
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
                float partial = 0.0f;
#pragma unroll
                for (int i = 0; i < D_PER_LANE; ++i) {
                    partial += q_reg[i] * k_lds[j * D + lane + i * WARP_SIZE];
                }
                const float s_j = warp_reduce_sum(partial) * scale;
                const bool masked = (row >= limit) || (row < swa_min);
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
            const size_t off = ((size_t) q_idx * n_heads_q + h_q) * D + lane + i * WARP_SIZE;
            out[off] = (fb_fp16_t) (o_reg[i] * inv_l);
        }
    }
}
