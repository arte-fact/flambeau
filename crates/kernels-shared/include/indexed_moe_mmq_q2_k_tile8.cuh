#pragma once
// Register-tiled prefill MoE MMQ (tile8) for Q2_K. The 2-bit weight is unpacked
// from `qs` by a per-sub shift: v = (qs >> (2*(sub&3))) & 0x03, with the 32-byte
// chunk selected by (sub>>2). Q2_K carries 16 uint8 scale bytes, each packing a
// 4-bit scale (low nibble) and a 4-bit min (high nibble); a 32-element sub-block
// spans two such bytes (a/b). The affine dot folds into sumf_d / sumf_m per
// super-block, closed by d/dmin: sums += d·sumf_d - dmin·sumf_m. Register-tiled
// (64 rows × 8 sorted slot-cols, no LDS / no warp ops). gate_up fuses two
// weights; down is the single-weight per-pair sibling. Backend-neutral: `dp4a`
// from the arch header; int loads via `load_int_b2`.
#include "quant_traits.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_q2_k_gate_up_tile8_row(
    const flambeau_block_q2_K* __restrict__ gate_w,
    const flambeau_block_q2_K* __restrict__ up_w,
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
        uint8_t g_sc[16] = {0}, u_sc[16] = {0};
        const flambeau_block_q2_K* gbx = nullptr;
        const flambeau_block_q2_K* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_d = (float) gbx->d; g_dmin = (float) gbx->dmin;
            u_d = (float) ubx->d; u_dmin = (float) ubx->dmin;
            #pragma unroll
            for (int j = 0; j < 16; ++j) { g_sc[j] = gbx->scales[j]; u_sc[j] = ubx->scales[j]; }
        }

        float g_sumf_d[TILE_N], g_sumf_m[TILE_N], u_sumf_d[TILE_N], u_sumf_m[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            g_sumf_d[c] = 0.0f; g_sumf_m[c] = 0.0f; u_sumf_d[c] = 0.0f; u_sumf_m[c] = 0.0f;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int chunk_idx = sub >> 2;
            const int shift     = 2 * (sub & 3);

            int g_v[8] = {0};
            int u_v[8] = {0};
            if (row_ok) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int gw = load_int_b2((const int8_t*) (gbx->qs + chunk_idx * 32), j);
                    const int uw = load_int_b2((const int8_t*) (ubx->qs + chunk_idx * 32), j);
                    g_v[j] = (gw >> shift) & 0x03030303;
                    u_v[j] = (uw >> shift) & 0x03030303;
                }
            }

            const int g_sca = (int) (g_sc[2 * sub + 0] & 0xF);
            const int g_scb = (int) (g_sc[2 * sub + 1] & 0xF);
            const int g_ma  = (int) (g_sc[2 * sub + 0] >> 4);
            const int g_mb  = (int) (g_sc[2 * sub + 1] >> 4);
            const int u_sca = (int) (u_sc[2 * sub + 0] & 0xF);
            const int u_scb = (int) (u_sc[2 * sub + 1] & 0xF);
            const int u_ma  = (int) (u_sc[2 * sub + 0] >> 4);
            const int u_mb  = (int) (u_sc[2 * sub + 1] >> 4);

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_token[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;

                int g_sumi_a = 0, g_sumi_b = 0, u_sumi_a = 0, u_sumi_b = 0;
                int sumi_y_a = 0, sumi_y_b = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int y_j = load_int_b2(by->qs, j);
                    g_sumi_a = dp4a(g_v[j], y_j, g_sumi_a);
                    u_sumi_a = dp4a(u_v[j], y_j, u_sumi_a);
                    sumi_y_a = dp4a(0x01010101, y_j, sumi_y_a);
                }
                #pragma unroll
                for (int j = 4; j < 8; ++j) {
                    const int y_j = load_int_b2(by->qs, j);
                    g_sumi_b = dp4a(g_v[j], y_j, g_sumi_b);
                    u_sumi_b = dp4a(u_v[j], y_j, u_sumi_b);
                    sumi_y_b = dp4a(0x01010101, y_j, sumi_y_b);
                }

                g_sumf_d[c] += d8 * (float) (g_sca * g_sumi_a + g_scb * g_sumi_b);
                g_sumf_m[c] += d8 * (float) (g_ma  * sumi_y_a + g_mb  * sumi_y_b);
                u_sumf_d[c] += d8 * (float) (u_sca * u_sumi_a + u_scb * u_sumi_b);
                u_sumf_m[c] += d8 * (float) (u_ma  * sumi_y_a + u_mb  * sumi_y_b);
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
static __device__ __forceinline__ void indexed_moe_mmq_q2_k_down_tile8_row(
    const flambeau_block_q2_K* __restrict__ down_w,
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
        float super_d = 0.0f, super_dmin = 0.0f;
        uint8_t sc_buf[16] = {0};
        const flambeau_block_q2_K* bx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            bx = &down_w[w_row_off];
            super_d = (float) bx->d; super_dmin = (float) bx->dmin;
            #pragma unroll
            for (int j = 0; j < 16; ++j) sc_buf[j] = bx->scales[j];
        }

        float sumf_d[TILE_N], sumf_m[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) { sumf_d[c] = 0.0f; sumf_m[c] = 0.0f; }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int chunk_idx = sub >> 2;
            const int shift     = 2 * (sub & 3);

            int v[8] = {0};
            if (row_ok) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int qw = load_int_b2((const int8_t*) (bx->qs + chunk_idx * 32), j);
                    v[j] = (qw >> shift) & 0x03030303;
                }
            }

            const int sc_a = (int) (sc_buf[2 * sub + 0] & 0xF);
            const int sc_b = (int) (sc_buf[2 * sub + 1] & 0xF);
            const int m_a  = (int) (sc_buf[2 * sub + 0] >> 4);
            const int m_b  = (int) (sc_buf[2 * sub + 1] >> 4);

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

                sumf_d[c] += d8 * (float) (sc_a * sumi_a + sc_b * sumi_b);
                sumf_m[c] += d8 * (float) (m_a  * sumi_y_a + m_b  * sumi_y_b);
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) sums[c] += super_d * sumf_d[c] - super_dmin * sumf_m[c];
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_pair[c] * n_rows + row;
        dst[out_idx] = sums[c];
    }
    (void) top_k; (void) n_tokens;
}
