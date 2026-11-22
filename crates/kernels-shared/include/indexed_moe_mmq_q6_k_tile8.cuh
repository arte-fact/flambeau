#pragma once
// Register-tiled prefill MoE MMQ (tile8) for Q6_K. The 6-bit weight reassembles
// from a 4-bit nibble in `ql` and a 2-bit high field in `qh`:
// v = nib4(ql) | (((qh >> qh_shift) << 4) & 0x30). The -32 bias is folded after
// the dot via the identity (v-32)·y = v·y - 32·Σy, so the dp4a runs on the raw
// unsigned weights and Σy comes from `dp4a(0x01010101, y)`. Q6_K carries 16
// signed int8 sub-block scales (one per 16 elements), so a 32-element sub-block
// spans two scale slots (a/b). Register-tiled (64 rows × 8 sorted slot-cols, no
// LDS / no warp ops). gate_up fuses two weights; down is the single-weight
// per-pair sibling. Backend-neutral: `dp4a` from the arch header; int loads via
// `load_int_b2`.
#include "quant_traits.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_q6_k_gate_up_tile8_row(
    const flambeau_block_q6_K* __restrict__ gate_w,
    const flambeau_block_q6_K* __restrict__ up_w,
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
        float g_super_d = 0.0f, u_super_d = 0.0f;
        int8_t g_sc_buf[16] = {0}, u_sc_buf[16] = {0};
        const flambeau_block_q6_K* gbx = nullptr;
        const flambeau_block_q6_K* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_super_d = (float) gbx->d;
            u_super_d = (float) ubx->d;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                g_sc_buf[j] = gbx->scales[j];
                u_sc_buf[j] = ubx->scales[j];
            }
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int h        = sub >> 2;
            const int q_idx    = sub & 3;
            const int qh_shift = 2 * q_idx;

            int g_v[8] = {0};
            int u_v[8] = {0};
            if (row_ok) {
                const int ql_base = 64 * h + ((q_idx & 1) ? 32 : 0);
                const int qh_base = 32 * h;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int g_ql_word = load_int_b2((const int8_t*) (gbx->ql + ql_base), j);
                    const int g_qh_word = load_int_b2((const int8_t*) (gbx->qh + qh_base), j);
                    const int u_ql_word = load_int_b2((const int8_t*) (ubx->ql + ql_base), j);
                    const int u_qh_word = load_int_b2((const int8_t*) (ubx->qh + qh_base), j);

                    const int g_nib4 = (q_idx < 2) ? (g_ql_word & 0x0F0F0F0F) : ((g_ql_word >> 4) & 0x0F0F0F0F);
                    const int u_nib4 = (q_idx < 2) ? (u_ql_word & 0x0F0F0F0F) : ((u_ql_word >> 4) & 0x0F0F0F0F);
                    const int g_hi4  = ((g_qh_word >> qh_shift) << 4) & 0x30303030;
                    const int u_hi4  = ((u_qh_word >> qh_shift) << 4) & 0x30303030;

                    g_v[j] = g_nib4 | g_hi4;
                    u_v[j] = u_nib4 | u_hi4;
                }
            }

            const int g_sc_a = (int) g_sc_buf[8 * h + 2 * q_idx + 0];
            const int g_sc_b = (int) g_sc_buf[8 * h + 2 * q_idx + 1];
            const int u_sc_a = (int) u_sc_buf[8 * h + 2 * q_idx + 0];
            const int u_sc_b = (int) u_sc_buf[8 * h + 2 * q_idx + 1];

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_token[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;

                int g_sumi_a = 0, g_sumi_b = 0, g_sumi_y_a = 0, g_sumi_y_b = 0;
                int u_sumi_a = 0, u_sumi_b = 0, u_sumi_y_a = 0, u_sumi_y_b = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int y_j = load_int_b2(by->qs, j);
                    g_sumi_a   = dp4a(g_v[j], y_j, g_sumi_a);
                    g_sumi_y_a = dp4a(0x01010101, y_j, g_sumi_y_a);
                    u_sumi_a   = dp4a(u_v[j], y_j, u_sumi_a);
                    u_sumi_y_a = dp4a(0x01010101, y_j, u_sumi_y_a);
                }
                #pragma unroll
                for (int j = 4; j < 8; ++j) {
                    const int y_j = load_int_b2(by->qs, j);
                    g_sumi_b   = dp4a(g_v[j], y_j, g_sumi_b);
                    g_sumi_y_b = dp4a(0x01010101, y_j, g_sumi_y_b);
                    u_sumi_b   = dp4a(u_v[j], y_j, u_sumi_b);
                    u_sumi_y_b = dp4a(0x01010101, y_j, u_sumi_y_b);
                }

                const int g_corr_a = g_sumi_a - 32 * g_sumi_y_a;
                const int g_corr_b = g_sumi_b - 32 * g_sumi_y_b;
                const int u_corr_a = u_sumi_a - 32 * u_sumi_y_a;
                const int u_corr_b = u_sumi_b - 32 * u_sumi_y_b;

                sums_gate[c] += g_super_d * d8 * ((float) (g_sc_a * g_corr_a + g_sc_b * g_corr_b));
                sums_up[c]   += u_super_d * d8 * ((float) (u_sc_a * u_corr_a + u_sc_b * u_corr_b));
            }
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
static __device__ __forceinline__ void indexed_moe_mmq_q6_k_down_tile8_row(
    const flambeau_block_q6_K* __restrict__ down_w,
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
        float super_d = 0.0f;
        int8_t sc_buf[16] = {0};
        const flambeau_block_q6_K* bx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            bx = &down_w[w_row_off];
            super_d = (float) bx->d;
            #pragma unroll
            for (int j = 0; j < 16; ++j) sc_buf[j] = bx->scales[j];
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int h        = sub >> 2;
            const int q_idx    = sub & 3;
            const int qh_shift = 2 * q_idx;

            int v[8] = {0};
            if (row_ok) {
                const int ql_base = 64 * h + ((q_idx & 1) ? 32 : 0);
                const int qh_base = 32 * h;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = load_int_b2((const int8_t*) (bx->ql + ql_base), j);
                    const int qh_word = load_int_b2((const int8_t*) (bx->qh + qh_base), j);
                    const int nib4 = (q_idx < 2) ? (ql_word & 0x0F0F0F0F) : ((ql_word >> 4) & 0x0F0F0F0F);
                    const int hi4  = ((qh_word >> qh_shift) << 4) & 0x30303030;
                    v[j] = nib4 | hi4;
                }
            }

            const int sc_a = (int) sc_buf[8 * h + 2 * q_idx + 0];
            const int sc_b = (int) sc_buf[8 * h + 2 * q_idx + 1];

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_pair[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;

                int sumi_a = 0, sumi_b = 0, sumi_y_a = 0, sumi_y_b = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int y_j = load_int_b2(by->qs, j);
                    sumi_a   = dp4a(v[j], y_j, sumi_a);
                    sumi_y_a = dp4a(0x01010101, y_j, sumi_y_a);
                }
                #pragma unroll
                for (int j = 4; j < 8; ++j) {
                    const int y_j = load_int_b2(by->qs, j);
                    sumi_b   = dp4a(v[j], y_j, sumi_b);
                    sumi_y_b = dp4a(0x01010101, y_j, sumi_y_b);
                }

                const int corr_a = sumi_a - 32 * sumi_y_a;
                const int corr_b = sumi_b - 32 * sumi_y_b;

                sums[c] += super_d * d8 * ((float) (sc_a * corr_a + sc_b * corr_b));
            }
        }
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_pair[c] * n_rows + row;
        dst[out_idx] = sums[c];
    }
    (void) top_k; (void) n_tokens;
}
