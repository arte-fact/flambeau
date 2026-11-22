#pragma once
// Register-tiled prefill MoE MMQ (tile8) for Q5_K. Like Q4_K but the 5-bit
// weight reassembles a high bit from `qh`: g_v[j] = nibble(qs) | ((qh>>s_bit & 1)
// << 4). Per super-block, the 8 sub-block dots fold into sumf_d / sumf_m (scaled
// by the 6-bit (scale,min) via flambeau_q4k_scale_min), then the block-level
// d/dmin close it: sums += d·sumf_d − dmin·sumf_m. Register-tiled (64 rows × 8
// sorted slot-cols, no LDS/no warp ops). gate_up fuses two weights; down is the
// single-weight per-pair sibling. Backend-neutral: `dp4a` from the arch header;
// int loads via `load_int_b2`.
#include "quant_traits.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_q5_k_gate_up_tile8_row(
    const flambeau_block_q5_K* __restrict__ gate_w,
    const flambeau_block_q5_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    OutT* __restrict__ gate_out,
    OutT* __restrict__ up_out,
    int n_rows, int n_tokens, int top_k, int n_sb_per_row, int n_experts)
{
    constexpr int MMQ_Y  = 64;
    constexpr int TILE_N = 8;
    constexpr int q8_per_super = QK_K / QK8_1;

    const int tile_m = blockIdx.x * MMQ_Y;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid    = threadIdx.x;

    __shared__ int padded_total_shared;
    if (tid == 0) padded_total_shared = padded_offsets[n_experts];
    __syncthreads();
    if (tile_n >= padded_total_shared) return;

    const int row     = tile_m + tid;
    const bool row_ok = (row < n_rows);

    const int first_pair = sorted_pair_idx_padded[tile_n];
    const int expert = expert_ids[first_pair];

    int slot_token[TILE_N];
    int slot_out_idx[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int pair = sorted_pair_idx_padded[tile_n + c];
        slot_token[c] = pair / top_k;
        slot_out_idx[c] = pair;
    }

    float sums_gate[TILE_N];
    float sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) { sums_gate[c] = 0.0f; sums_up[c] = 0.0f; }

    for (int ib = 0; ib < n_sb_per_row; ++ib) {
        float g_d = 0.0f, g_dmin = 0.0f, u_d = 0.0f, u_dmin = 0.0f;
        uint8_t g_sub_sc[8] = {0}, g_sub_m[8] = {0};
        uint8_t u_sub_sc[8] = {0}, u_sub_m[8] = {0};
        const flambeau_block_q5_K* gbx = nullptr;
        const flambeau_block_q5_K* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_d = (float) gbx->d; g_dmin = (float) gbx->dmin;
            u_d = (float) ubx->d; u_dmin = (float) ubx->dmin;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                flambeau_q4k_scale_min(j, gbx->scales, &g_sub_sc[j], &g_sub_m[j]);
                flambeau_q4k_scale_min(j, ubx->scales, &u_sub_sc[j], &u_sub_m[j]);
            }
        }

        float g_sumf_d[TILE_N], g_sumf_m[TILE_N], u_sumf_d[TILE_N], u_sumf_m[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            g_sumf_d[c] = 0.0f; g_sumf_m[c] = 0.0f; u_sumf_d[c] = 0.0f; u_sumf_m[c] = 0.0f;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int il    = sub >> 1;
            const int half  = sub & 1;
            const int s_bit = 2 * il + half;

            int g_v[8] = {0};
            int u_v[8] = {0};
            if (row_ok) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int g_ql_word = load_int_b2((const int8_t*) (gbx->qs + 32 * il), j);
                    const int g_qh_word = load_int_b2((const int8_t*) gbx->qh, j);
                    const int u_ql_word = load_int_b2((const int8_t*) (ubx->qs + 32 * il), j);
                    const int u_qh_word = load_int_b2((const int8_t*) ubx->qh, j);
                    const int g_nib4 = (half == 0) ? (g_ql_word & 0x0F0F0F0F) : ((g_ql_word >> 4) & 0x0F0F0F0F);
                    const int u_nib4 = (half == 0) ? (u_ql_word & 0x0F0F0F0F) : ((u_ql_word >> 4) & 0x0F0F0F0F);
                    const int g_hi4 = ((g_qh_word >> s_bit) & 0x01010101) << 4;
                    const int u_hi4 = ((u_qh_word >> s_bit) & 0x01010101) << 4;
                    g_v[j] = g_nib4 | g_hi4;
                    u_v[j] = u_nib4 | u_hi4;
                }
            }

            const float g_sc_f = (float) g_sub_sc[sub];
            const float g_m_f  = (float) g_sub_m[sub];
            const float u_sc_f = (float) u_sub_sc[sub];
            const float u_m_f  = (float) u_sub_m[sub];

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_token[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;

                int g_sumi_d = 0, g_sumi_y = 0, u_sumi_d = 0, u_sumi_y = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = load_int_b2(by->qs, j);
                    g_sumi_d = dp4a(g_v[j], y_j, g_sumi_d);
                    g_sumi_y = dp4a(0x01010101, y_j, g_sumi_y);
                    u_sumi_d = dp4a(u_v[j], y_j, u_sumi_d);
                    u_sumi_y = dp4a(0x01010101, y_j, u_sumi_y);
                }
                g_sumf_d[c] += d8 * (float) g_sumi_d * g_sc_f;
                g_sumf_m[c] += d8 * (float) g_sumi_y * g_m_f;
                u_sumf_d[c] += d8 * (float) u_sumi_d * u_sc_f;
                u_sumf_m[c] += d8 * (float) u_sumi_y * u_m_f;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            sums_gate[c] += g_d * g_sumf_d[c] - g_dmin * g_sumf_m[c];
            sums_up[c]   += u_d * u_sumf_d[c] - u_dmin * u_sumf_m[c];
        }
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_out_idx[c] * n_rows + row;
        gate_out[out_idx] = sums_gate[c];
        up_out[out_idx]   = sums_up[c];
    }
    (void) n_tokens;
}

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_q5_k_down_tile8_row(
    const flambeau_block_q5_K* __restrict__ down_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int top_k, int n_sb_per_row, int n_experts)
{
    constexpr int MMQ_Y  = 64;
    constexpr int TILE_N = 8;
    constexpr int q8_per_super = QK_K / QK8_1;

    const int tile_m = blockIdx.x * MMQ_Y;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid    = threadIdx.x;

    __shared__ int padded_total_shared;
    if (tid == 0) padded_total_shared = padded_offsets[n_experts];
    __syncthreads();
    if (tile_n >= padded_total_shared) return;

    const int row     = tile_m + tid;
    const bool row_ok = (row < n_rows);

    const int first_pair = sorted_pair_idx_padded[tile_n];
    const int expert = expert_ids[first_pair];

    int slot_pair[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) slot_pair[c] = sorted_pair_idx_padded[tile_n + c];

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < n_sb_per_row; ++ib) {
        float w_d = 0.0f, w_dmin = 0.0f;
        uint8_t sub_sc[8] = {0}, sub_m[8] = {0};
        const flambeau_block_q5_K* bx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            bx = &down_w[w_row_off];
            w_d = (float) bx->d; w_dmin = (float) bx->dmin;
            #pragma unroll
            for (int j = 0; j < 8; ++j) flambeau_q4k_scale_min(j, bx->scales, &sub_sc[j], &sub_m[j]);
        }

        float sumf_d[TILE_N], sumf_m[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) { sumf_d[c] = 0.0f; sumf_m[c] = 0.0f; }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int il    = sub >> 1;
            const int half  = sub & 1;
            const int s_bit = 2 * il + half;

            int w_v[8] = {0};
            if (row_ok) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = load_int_b2((const int8_t*) (bx->qs + 32 * il), j);
                    const int qh_word = load_int_b2((const int8_t*) bx->qh, j);
                    const int nib4 = (half == 0) ? (ql_word & 0x0F0F0F0F) : ((ql_word >> 4) & 0x0F0F0F0F);
                    const int hi4  = ((qh_word >> s_bit) & 0x01010101) << 4;
                    w_v[j] = nib4 | hi4;
                }
            }

            const float sc_f = (float) sub_sc[sub];
            const float m_f  = (float) sub_m[sub];

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_pair[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;

                int sumi_d = 0, sumi_y = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = load_int_b2(by->qs, j);
                    sumi_d = dp4a(w_v[j], y_j, sumi_d);
                    sumi_y = dp4a(0x01010101, y_j, sumi_y);
                }
                sumf_d[c] += d8 * (float) sumi_d * sc_f;
                sumf_m[c] += d8 * (float) sumi_y * m_f;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) sums[c] += w_d * sumf_d[c] - w_dmin * sumf_m[c];
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_pair[c] * n_rows + row;
        dst[out_idx] = sums[c];
    }
    (void) top_k; (void) n_tokens;
}
