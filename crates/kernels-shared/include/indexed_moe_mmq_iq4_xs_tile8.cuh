#pragma once
// Register-tiled prefill MoE MMQ (tile8) for IQ4_XS. Super-block of 8 subs, each
// a 32-elem group with its own 6-bit scale (flambeau_iq4_xs_scale from the 4-bit
// scales_l nibbles + 2-bit scales_h high field). Each 4-bit code maps through
// the IQ4_NL LUT to a signed int8, packed 4-per-int, dp4a against the Q8_1 sub;
// reconstruction is symmetric (d·ls·LUT[code]). Register-tiled (64 rows × 8
// sorted slot-cols, no LDS / no warp ops). gate_up fuses two weights; down is
// the single-weight per-pair sibling. Backend-neutral: `dp4a` from the arch
// header, IQ4_NL LUT + scale helper from iq_grid.cuh, int loads via load_int_b2.
#include "quant_traits.cuh"
#include "iq_grid.cuh"

static __device__ __forceinline__ int flambeau_pack_iq4xs_lut(int nibbles) {
    const int b0 = (int) flambeau_iq4nl_lut(nibbles & 0xFF);
    const int b1 = (int) flambeau_iq4nl_lut((nibbles >>  8) & 0xFF);
    const int b2 = (int) flambeau_iq4nl_lut((nibbles >> 16) & 0xFF);
    const int b3 = (int) flambeau_iq4nl_lut((nibbles >> 24) & 0xFF);
    return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
}

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_iq4_xs_gate_up_tile8_row(
    const flambeau_block_iq4_xs* __restrict__ gate_w,
    const flambeau_block_iq4_xs* __restrict__ up_w,
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
        float g_d = 0.0f, u_d = 0.0f;
        uint16_t g_sh = 0, u_sh = 0;
        const flambeau_block_iq4_xs* gbx = nullptr;
        const flambeau_block_iq4_xs* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_d = (float) gbx->d; u_d = (float) ubx->d;
            g_sh = gbx->scales_h; u_sh = ubx->scales_h;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int g_v[8] = {0};
            int u_v[8] = {0};
            int g_ls = 0, u_ls = 0;
            if (row_ok) {
                g_ls = flambeau_iq4_xs_scale(sub, g_sh, gbx->scales_l);
                u_ls = flambeau_iq4_xs_scale(sub, u_sh, ubx->scales_l);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int gw = load_int_b2((const int8_t*) (gbx->qs + 16 * sub), j);
                    const int uw = load_int_b2((const int8_t*) (ubx->qs + 16 * sub), j);
                    g_v[j]     = flambeau_pack_iq4xs_lut(gw        & 0x0F0F0F0F);
                    g_v[j + 4] = flambeau_pack_iq4xs_lut((gw >> 4) & 0x0F0F0F0F);
                    u_v[j]     = flambeau_pack_iq4xs_lut(uw        & 0x0F0F0F0F);
                    u_v[j + 4] = flambeau_pack_iq4xs_lut((uw >> 4) & 0x0F0F0F0F);
                }
            }

            const float g_lsf = (float) g_ls;
            const float u_lsf = (float) u_ls;

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_token[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                int g_sumi = 0, u_sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = load_int_b2(by->qs, j);
                    g_sumi = dp4a(g_v[j], y_j, g_sumi);
                    u_sumi = dp4a(u_v[j], y_j, u_sumi);
                }
                sums_gate[c] += g_d * d8 * (float) g_sumi * g_lsf;
                sums_up[c]   += u_d * d8 * (float) u_sumi * u_lsf;
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
static __device__ __forceinline__ void indexed_moe_mmq_iq4_xs_down_tile8_row(
    const flambeau_block_iq4_xs* __restrict__ down_w,
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
        float d = 0.0f;
        uint16_t sh = 0;
        const flambeau_block_iq4_xs* bx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            bx = &down_w[w_row_off];
            d = (float) bx->d;
            sh = bx->scales_h;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int v[8] = {0};
            int ls = 0;
            if (row_ok) {
                ls = flambeau_iq4_xs_scale(sub, sh, bx->scales_l);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int w = load_int_b2((const int8_t*) (bx->qs + 16 * sub), j);
                    v[j]     = flambeau_pack_iq4xs_lut(w        & 0x0F0F0F0F);
                    v[j + 4] = flambeau_pack_iq4xs_lut((w >> 4) & 0x0F0F0F0F);
                }
            }
            const float lsf = (float) ls;

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_pair[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) sumi = dp4a(v[j], load_int_b2(by->qs, j), sumi);
                sums[c] += d * d8 * (float) sumi * lsf;
            }
        }
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) dst[(size_t) slot_pair[c] * n_rows + row] = sums[c];
    (void) top_k; (void) n_tokens;
}
