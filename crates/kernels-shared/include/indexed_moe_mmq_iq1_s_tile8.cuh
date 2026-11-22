#pragma once
// Register-tiled prefill MoE MMQ (tile8) for IQ1_S. 256-elem super-blocks: an
// 11-bit index into `IQ1S_GRID` (uint64 of 8 signed-i8 magnitudes) per group,
// with a per-sub-block 3-bit scale and a ±IQ1_DELTA bias term applied as
// `dl * (d8 * sumi + delta * s8)`. Register-tiled (64 rows × 8 sorted slot-cols,
// no LDS / no warp ops). gate_up fuses two weights; down is the single-weight
// per-pair sibling. Backend-neutral: `dp4a` from the arch header,
// `IQ1S_GRID`/`IQ1_DELTA` from iq_grid.cuh, int loads via `load_int_b2`.
#include "quant_traits.cuh"
#include "iq_grid.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_iq1_s_gate_up_tile8_row(
    const flambeau_block_iq1_s* __restrict__ gate_w,
    const flambeau_block_iq1_s* __restrict__ up_w,
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
    const int expert     = expert_ids[first_pair];

    int slot_token[TILE_N];
    int slot_out_idx[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int pair = sorted_pair_idx_padded[tile_n + c];
        slot_token[c]   = pair / top_k;
        slot_out_idx[c] = pair;
    }

    float sums_gate[TILE_N], sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) { sums_gate[c] = 0.0f; sums_up[c] = 0.0f; }

    (void) n_tokens;

    for (int ib = 0; ib < n_sb_per_row; ++ib) {
        float g_super_d = 0.0f, u_super_d = 0.0f;
        const flambeau_block_iq1_s* gbx = nullptr;
        const flambeau_block_iq1_s* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_super_d = (float) gbx->d;
            u_super_d = (float) ubx->d;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int g_v[8] = {0}, u_v[8] = {0};
            float g_dl = 0.0f, u_dl = 0.0f;
            float g_delta = 0.0f, u_delta = 0.0f;
            if (row_ok) {
                const int g_qh = (int) gbx->qh[2 * sub] | ((int) gbx->qh[2 * sub + 1] << 8);
                const int u_qh = (int) ubx->qh[2 * sub] | ((int) ubx->qh[2 * sub + 1] << 8);
                g_dl    = g_super_d * (2.0f * (float)((g_qh >> 12) & 7) + 1.0f);
                u_dl    = u_super_d * (2.0f * (float)((u_qh >> 12) & 7) + 1.0f);
                g_delta = (g_qh & 0x8000) ? -IQ1_DELTA : IQ1_DELTA;
                u_delta = (u_qh & 0x8000) ? -IQ1_DELTA : IQ1_DELTA;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int g_idx = (int) gbx->qs[4 * sub + l] | (((g_qh >> (3 * l)) & 7) << 8);
                    const int u_idx = (int) ubx->qs[4 * sub + l] | (((u_qh >> (3 * l)) & 7) << 8);
                    const uint64_t g_u64 = IQ1S_GRID[g_idx];
                    const uint64_t u_u64 = IQ1S_GRID[u_idx];
                    g_v[2 * l + 0] = (int)(g_u64 & 0xFFFFFFFF);
                    g_v[2 * l + 1] = (int)((g_u64 >> 32) & 0xFFFFFFFF);
                    u_v[2 * l + 0] = (int)(u_u64 & 0xFFFFFFFF);
                    u_v[2 * l + 1] = (int)((u_u64 >> 32) & 0xFFFFFFFF);
                }
            }

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_token[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const float s8 = (float) by->s;
                int g_sumi = 0, u_sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = load_int_b2(by->qs, j);
                    g_sumi = dp4a(g_v[j], y_j, g_sumi);
                    u_sumi = dp4a(u_v[j], y_j, u_sumi);
                }
                sums_gate[c] += g_dl * (d8 * (float) g_sumi + g_delta * s8);
                sums_up[c]   += u_dl * (d8 * (float) u_sumi + u_delta * s8);
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
}

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_iq1_s_down_tile8_row(
    const flambeau_block_iq1_s* __restrict__ down_w,
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
    const int expert     = expert_ids[first_pair];

    int slot_pair[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) slot_pair[c] = sorted_pair_idx_padded[tile_n + c];

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;
    (void) n_tokens; (void) top_k;

    for (int ib = 0; ib < n_sb_per_row; ++ib) {
        float super_d = 0.0f;
        const flambeau_block_iq1_s* bx = nullptr;
        if (row_ok) {
            bx = &down_w[((size_t) expert * n_rows + row) * n_sb_per_row + ib];
            super_d = (float) bx->d;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int v[8] = {0};
            float dl = 0.0f, delta = 0.0f;
            if (row_ok) {
                const int qh = (int) bx->qh[2 * sub] | ((int) bx->qh[2 * sub + 1] << 8);
                dl    = super_d * (2.0f * (float)((qh >> 12) & 7) + 1.0f);
                delta = (qh & 0x8000) ? -IQ1_DELTA : IQ1_DELTA;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int idx = (int) bx->qs[4 * sub + l] | (((qh >> (3 * l)) & 7) << 8);
                    const uint64_t g_u64 = IQ1S_GRID[idx];
                    v[2 * l + 0] = (int)(g_u64 & 0xFFFFFFFF);
                    v[2 * l + 1] = (int)((g_u64 >> 32) & 0xFFFFFFFF);
                }
            }

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_pair[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const float s8 = (float) by->s;
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) sumi = dp4a(v[j], load_int_b2(by->qs, j), sumi);
                sums[c] += dl * (d8 * (float) sumi + delta * s8);
            }
        }
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        dst[(size_t) slot_pair[c] * n_rows + row] = sums[c];
    }
}
