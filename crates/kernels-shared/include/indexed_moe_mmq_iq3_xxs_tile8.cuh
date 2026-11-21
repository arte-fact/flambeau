#pragma once
// Register-tiled prefill MoE MMQ (tile8) for IQ3_XXS. 256-elem super-blocks (8
// Q8_1 sub-blocks each): per 8-byte sub-block, 8 codebook-index bytes select
// IQ3XXS_GRID u32 entries (4 signed int8 magnitudes each), an aux32 supplies
// four 7-bit KSIGNS_IQ2XS sign-LUT indices and a 4-bit scale nibble (scale =
// (0.5 + (aux32 >> 28)) * 0.5). Each grid entry is sign-applied, packed
// 4-per-int, and dp4a'd against the Q8_1 activation. Register-tiled (64 rows ×
// 8 sorted slot-cols, no LDS / no warp ops). gate_up fuses two weights; down is
// the single-weight per-pair sibling. Backend-neutral: `dp4a` from the arch
// header, `IQ3XXS_GRID` / `KSIGNS_IQ2XS` from iq_grid.cuh, int loads via
// `load_int_b2`.
#include "quant_traits.cuh"
#include "iq_grid.cuh"

static __device__ __forceinline__ int flambeau_iq3_xxs_apply_signs_packed(int g_u32, int sign4) {
    int out = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int mag = (g_u32 >> (8 * j)) & 0xFF;
        const int neg = (sign4 >> j) & 1;
        const int v   = neg ? -mag : mag;
        out |= (v & 0xFF) << (8 * j);
    }
    return out;
}

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_iq3_xxs_gate_up_tile8_row(
    const flambeau_block_iq3_xxs* __restrict__ gate_w,
    const flambeau_block_iq3_xxs* __restrict__ up_w,
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
        float g_d = 0.0f, u_d = 0.0f;
        const flambeau_block_iq3_xxs* gbx = nullptr;
        const flambeau_block_iq3_xxs* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_d = (float) gbx->d;
            u_d = (float) ubx->d;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int g_v[8] = {0};
            int u_v[8] = {0};
            float g_sf = 0.0f, u_sf = 0.0f;
            if (row_ok) {
                const uint8_t* g_scs = gbx->qs + QK_K / 4;
                const uint8_t* u_scs = ubx->qs + QK_K / 4;
                const uint8_t* g_sp = g_scs + 4 * sub;
                const uint8_t* u_sp = u_scs + 4 * sub;
                const uint32_t g_aux32 =
                      (uint32_t) g_sp[0] | ((uint32_t) g_sp[1] << 8)
                    | ((uint32_t) g_sp[2] << 16) | ((uint32_t) g_sp[3] << 24);
                const uint32_t u_aux32 =
                      (uint32_t) u_sp[0] | ((uint32_t) u_sp[1] << 8)
                    | ((uint32_t) u_sp[2] << 16) | ((uint32_t) u_sp[3] << 24);
                g_sf = (0.5f + (float)(g_aux32 >> 28)) * 0.5f;
                u_sf = (0.5f + (float)(u_aux32 >> 28)) * 0.5f;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int g1_idx = (int) gbx->qs[sub * 8 + 2 * l + 0];
                    const int g2_idx = (int) gbx->qs[sub * 8 + 2 * l + 1];
                    const int u1_idx = (int) ubx->qs[sub * 8 + 2 * l + 0];
                    const int u2_idx = (int) ubx->qs[sub * 8 + 2 * l + 1];
                    const int g1 = (int) IQ3XXS_GRID[g1_idx];
                    const int g2 = (int) IQ3XXS_GRID[g2_idx];
                    const int uu1 = (int) IQ3XXS_GRID[u1_idx];
                    const int uu2 = (int) IQ3XXS_GRID[u2_idx];
                    const uint8_t g_sb = KSIGNS_IQ2XS[(g_aux32 >> (7 * l)) & 0x7F];
                    const uint8_t u_sb = KSIGNS_IQ2XS[(u_aux32 >> (7 * l)) & 0x7F];
                    g_v[2 * l + 0] = flambeau_iq3_xxs_apply_signs_packed(g1,  g_sb        & 0x0F);
                    g_v[2 * l + 1] = flambeau_iq3_xxs_apply_signs_packed(g2, (g_sb >> 4)  & 0x0F);
                    u_v[2 * l + 0] = flambeau_iq3_xxs_apply_signs_packed(uu1, u_sb        & 0x0F);
                    u_v[2 * l + 1] = flambeau_iq3_xxs_apply_signs_packed(uu2, (u_sb >> 4) & 0x0F);
                }
            }

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_token[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                int g_sumi = 0, u_sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = load_int_b2((const int8_t*) by->qs, j);
                    g_sumi = dp4a(g_v[j], y_j, g_sumi);
                    u_sumi = dp4a(u_v[j], y_j, u_sumi);
                }
                sums_gate[c] += g_d * d8 * (float) g_sumi * g_sf;
                sums_up[c]   += u_d * d8 * (float) u_sumi * u_sf;
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
static __device__ __forceinline__ void indexed_moe_mmq_iq3_xxs_down_tile8_row(
    const flambeau_block_iq3_xxs* __restrict__ down_w,
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
        float d = 0.0f;
        const flambeau_block_iq3_xxs* bx = nullptr;
        if (row_ok) {
            bx = &down_w[((size_t) expert * n_rows + row) * n_sb_per_row + ib];
            d  = (float) bx->d;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int v[8] = {0};
            float sf = 0.0f;
            if (row_ok) {
                const uint8_t* scs = bx->qs + QK_K / 4;
                const uint8_t* sp  = scs + 4 * sub;
                const uint32_t aux32 =
                      (uint32_t) sp[0] | ((uint32_t) sp[1] << 8)
                    | ((uint32_t) sp[2] << 16) | ((uint32_t) sp[3] << 24);
                sf = (0.5f + (float)(aux32 >> 28)) * 0.5f;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int g1_idx = (int) bx->qs[sub * 8 + 2 * l + 0];
                    const int g2_idx = (int) bx->qs[sub * 8 + 2 * l + 1];
                    const int g1 = (int) IQ3XXS_GRID[g1_idx];
                    const int g2 = (int) IQ3XXS_GRID[g2_idx];
                    const uint8_t sb = KSIGNS_IQ2XS[(aux32 >> (7 * l)) & 0x7F];
                    v[2 * l + 0] = flambeau_iq3_xxs_apply_signs_packed(g1, sb        & 0x0F);
                    v[2 * l + 1] = flambeau_iq3_xxs_apply_signs_packed(g2, (sb >> 4) & 0x0F);
                }
            }

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_pair[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) sumi = dp4a(v[j], load_int_b2((const int8_t*) by->qs, j), sumi);
                sums[c] += d * d8 * (float) sumi * sf;
            }
        }
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        dst[(size_t) slot_pair[c] * n_rows + row] = sums[c];
    }
}
