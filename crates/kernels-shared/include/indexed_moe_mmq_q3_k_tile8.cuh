#pragma once
// Register-tiled prefill MoE MMQ (tile8) for Q3_K. The 3-bit weight is a 2-bit
// field from `qs` (per-sub shift) OR'd with one high bit from `hmask`:
// v = ((qs >> shift) & 0x03) | (((hmask >> hmask_bit_pos) << 2) & 0x04), biased
// -4 and folded after the dot via (v-4)·y = v·y - 4·Σy. Scales unpack to 16
// signed values via flambeau_q3k_unpack_scales (biased -32); a 32-element sub
// spans two scale slots (a/b). Register-tiled (64 rows × 8 sorted slot-cols, no
// LDS / no warp ops). gate_up fuses two weights; down is the single-weight
// per-pair sibling. Backend-neutral: `dp4a` from the arch header; int loads via
// `load_int_b2` (Q3_K blocks are 2-byte aligned, so the 2-byte-safe read holds).
#include "quant_traits.cuh"

static __device__ __forceinline__ void flambeau_q3k_tile8_decode_sub(
    bool row_ok, const flambeau_block_q3_K* __restrict__ bx,
    int blk128, int shift_iter, int v[8])
{
    if (!row_ok) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = 0;
        return;
    }
    const int shift         = 2 * shift_iter;
    const int hmask_bit_pos = shift_iter + 4 * blk128;
    const int8_t* qs_base = (const int8_t*) (bx->qs + blk128 * 32);
    const int8_t* hm_base = (const int8_t*) bx->hmask;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned ql_word = (unsigned) load_int_b2(qs_base, j);
        const unsigned qh_word = (unsigned) load_int_b2(hm_base, j);
        const int raw2 = (int) ((ql_word >> shift) & 0x03030303u);
        const int hi   = (int) (((qh_word >> hmask_bit_pos) << 2) & 0x04040404u);
        v[j] = raw2 | hi;
    }
}

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_q3_k_gate_up_tile8_row(
    const flambeau_block_q3_K* __restrict__ gate_w,
    const flambeau_block_q3_K* __restrict__ up_w,
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
        int8_t g_sc[16] = {0}, u_sc[16] = {0};
        const flambeau_block_q3_K* gbx = nullptr;
        const flambeau_block_q3_K* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_d = (float) gbx->d;
            u_d = (float) ubx->d;
            flambeau_q3k_unpack_scales(gbx->scales, g_sc);
            flambeau_q3k_unpack_scales(ubx->scales, u_sc);
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int blk128     = sub >> 2;
            const int shift_iter = sub & 3;

            int g_v[8], u_v[8];
            flambeau_q3k_tile8_decode_sub(row_ok, gbx, blk128, shift_iter, g_v);
            flambeau_q3k_tile8_decode_sub(row_ok, ubx, blk128, shift_iter, u_v);

            const int g_sca = (int) g_sc[2 * sub + 0] - 32;
            const int g_scb = (int) g_sc[2 * sub + 1] - 32;
            const int u_sca = (int) u_sc[2 * sub + 0] - 32;
            const int u_scb = (int) u_sc[2 * sub + 1] - 32;

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

                const int g_corr_a = g_sumi_a - 4 * sumi_y_a;
                const int g_corr_b = g_sumi_b - 4 * sumi_y_b;
                const int u_corr_a = u_sumi_a - 4 * sumi_y_a;
                const int u_corr_b = u_sumi_b - 4 * sumi_y_b;

                sums_gate[c] += g_d * d8 * ((float) (g_sca * g_corr_a + g_scb * g_corr_b));
                sums_up[c]   += u_d * d8 * ((float) (u_sca * u_corr_a + u_scb * u_corr_b));
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
static __device__ __forceinline__ void indexed_moe_mmq_q3_k_down_tile8_row(
    const flambeau_block_q3_K* __restrict__ down_w,
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
        const flambeau_block_q3_K* bx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            bx = &down_w[w_row_off];
            super_d = (float) bx->d;
            flambeau_q3k_unpack_scales(bx->scales, sc_buf);
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int blk128     = sub >> 2;
            const int shift_iter = sub & 3;

            int v[8];
            flambeau_q3k_tile8_decode_sub(row_ok, bx, blk128, shift_iter, v);

            const int sc_a = (int) sc_buf[2 * sub + 0] - 32;
            const int sc_b = (int) sc_buf[2 * sub + 1] - 32;

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

                const int corr_a = sumi_a - 4 * sumi_y_a;
                const int corr_b = sumi_b - 4 * sumi_y_b;

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
