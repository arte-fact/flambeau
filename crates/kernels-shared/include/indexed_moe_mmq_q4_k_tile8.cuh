#pragma once
// Register-tiled prefill MoE MMQ (tile8) for Q4_K. MMQ_Y=64 weight rows (one per
// thread) × TILE_N=8 sorted slot-columns: each thread decodes its row's Q4_K
// super-block (d/dmin + 8 sub-block 6-bit (scale,min) via flambeau_q4k_scale_min,
// nibble qs) once per super-block and reuses it across the 8 slot activations.
// The affine dot per sub-block is `d8·(sumi·d·sc − Σy·dmin·m)` with the Σy term
// from `dp4a(0x01010101, y)`. No LDS / no cross-thread comms — block height is a
// logical tile, not a hardware warp. `gate_up` fuses two weights sharing one
// per-token activation; `down` is the single-weight per-pair sibling. Backend-
// neutral: `dp4a` from the arch header; int loads via `load_int_b2`.
#include "quant_traits.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_q4_k_gate_up_tile8_row(
    const flambeau_block_q4_K* __restrict__ gate_w,
    const flambeau_block_q4_K* __restrict__ up_w,
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
    constexpr int q8_per_super = QK_K / QK8_1;   // 8

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
        const int t = pair / top_k;
        slot_token[c] = t;
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

        const flambeau_block_q4_K* gbx = nullptr;
        const flambeau_block_q4_K* ubx = nullptr;
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

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int il   = sub >> 1;
            const int half = sub & 1;

            int g_v[8] = {0};
            int u_v[8] = {0};
            if (row_ok) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int gw = load_int_b2((const int8_t*) (gbx->qs + 32 * il), j);
                    const int uw = load_int_b2((const int8_t*) (ubx->qs + 32 * il), j);
                    g_v[j] = (half == 0) ? (gw & 0x0F0F0F0F) : ((gw >> 4) & 0x0F0F0F0F);
                    u_v[j] = (half == 0) ? (uw & 0x0F0F0F0F) : ((uw >> 4) & 0x0F0F0F0F);
                }
            }

            const float gd_sc = g_d * (float) g_sub_sc[sub];
            const float gd_m  = g_dmin * (float) g_sub_m[sub];
            const float ud_sc = u_d * (float) u_sub_sc[sub];
            const float ud_m  = u_dmin * (float) u_sub_m[sub];

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_token[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;

                int sumi_gd = 0, sumi_ud = 0, sumi_y = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = load_int_b2(by->qs, j);
                    sumi_gd = dp4a(g_v[j], y_j, sumi_gd);
                    sumi_ud = dp4a(u_v[j], y_j, sumi_ud);
                    sumi_y  = dp4a(0x01010101, y_j, sumi_y);
                }
                const float sumi_y_f = (float) sumi_y;
                sums_gate[c] += d8 * ((float) sumi_gd * gd_sc - sumi_y_f * gd_m);
                sums_up[c]   += d8 * ((float) sumi_ud * ud_sc - sumi_y_f * ud_m);
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
static __device__ __forceinline__ void indexed_moe_mmq_q4_k_down_tile8_row(
    const flambeau_block_q4_K* __restrict__ down_w,
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
        const flambeau_block_q4_K* bx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            bx = &down_w[w_row_off];
            w_d = (float) bx->d; w_dmin = (float) bx->dmin;
            #pragma unroll
            for (int j = 0; j < 8; ++j) flambeau_q4k_scale_min(j, bx->scales, &sub_sc[j], &sub_m[j]);
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int il   = sub >> 1;
            const int half = sub & 1;

            int w_v[8] = {0};
            if (row_ok) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int qw = load_int_b2((const int8_t*) (bx->qs + 32 * il), j);
                    w_v[j] = (half == 0) ? (qw & 0x0F0F0F0F) : ((qw >> 4) & 0x0F0F0F0F);
                }
            }

            const float d_sc = w_d * (float) sub_sc[sub];
            const float d_m  = w_dmin * (float) sub_m[sub];

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
                sums[c] += d8 * ((float) sumi_d * d_sc - (float) sumi_y * d_m);
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
