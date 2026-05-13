#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef QK_K
#define QK_K 256
#endif
#ifndef QK8_1
#define QK8_1 32
#endif

#define MMQ_Y 64
#define TILE_N 8

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 2)
void flambeau_indexed_moe_mmq_q6_k_gate_up_tile8_dp4a_q8_1(
    const flambeau_block_q6_K* __restrict__ gate_w,
    const flambeau_block_q6_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    float*      __restrict__ gate_out,
    float*      __restrict__ up_out,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row,
    const int n_experts
) {
    const int tile_m = blockIdx.x * WARP_SIZE;
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
        const int s = pair - t * top_k;
        slot_token[c] = t;
        slot_out_idx[c] = t * top_k + s;
    }

    const int blocks_per_row_x = n_sb_per_row;
    constexpr int q8_per_super = QK_K / QK8_1;

    float sums_gate[TILE_N];
    float sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) { sums_gate[c] = 0.0f; sums_up[c] = 0.0f; }

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float g_super_d = 0.0f, u_super_d = 0.0f;
        int8_t g_sc_buf[16] = {0};
        int8_t u_sc_buf[16] = {0};
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
            const int h     = sub >> 2;
            const int q_idx = sub & 3;
            const int qh_shift = 2 * q_idx;

            int g_v[8] = {0};
            int u_v[8] = {0};
            if (row_ok) {
                const int ql_base = 64 * h + ((q_idx & 1) ? 32 : 0);
                const int qh_base = 32 * h;
                const int* g_ql_words = (const int*) (gbx->ql + ql_base);
                const int* g_qh_words = (const int*) (gbx->qh + qh_base);
                const int* u_ql_words = (const int*) (ubx->ql + ql_base);
                const int* u_qh_words = (const int*) (ubx->qh + qh_base);

                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int g_ql_word = g_ql_words[j];
                    const int g_qh_word = g_qh_words[j];
                    const int u_ql_word = u_ql_words[j];
                    const int u_qh_word = u_qh_words[j];

                    const int g_nib4 = (q_idx < 2)
                        ? ((g_ql_word >> 0) & 0x0F0F0F0F)
                        : ((g_ql_word >> 4) & 0x0F0F0F0F);
                    const int u_nib4 = (q_idx < 2)
                        ? ((u_ql_word >> 0) & 0x0F0F0F0F)
                        : ((u_ql_word >> 4) & 0x0F0F0F0F);

                    const int g_hi4 = ((g_qh_word >> qh_shift) << 4) & 0x30303030;
                    const int u_hi4 = ((u_qh_word >> qh_shift) << 4) & 0x30303030;

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
                const int* y_packed = (const int*) by->qs;

                int g_sumi_a = 0, g_sumi_b = 0;
                int g_sumi_y_a = 0, g_sumi_y_b = 0;
                int u_sumi_a = 0, u_sumi_b = 0;
                int u_sumi_y_a = 0, u_sumi_y_b = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    g_sumi_a = dp4a(g_v[j], y_packed[j], g_sumi_a);
                    g_sumi_y_a = dp4a(0x01010101, y_packed[j], g_sumi_y_a);
                    u_sumi_a = dp4a(u_v[j], y_packed[j], u_sumi_a);
                    u_sumi_y_a = dp4a(0x01010101, y_packed[j], u_sumi_y_a);
                }
                #pragma unroll
                for (int j = 4; j < 8; ++j) {
                    g_sumi_b = dp4a(g_v[j], y_packed[j], g_sumi_b);
                    g_sumi_y_b = dp4a(0x01010101, y_packed[j], g_sumi_y_b);
                    u_sumi_b = dp4a(u_v[j], y_packed[j], u_sumi_b);
                    u_sumi_y_b = dp4a(0x01010101, y_packed[j], u_sumi_y_b);
                }

                const int g_corr_a = g_sumi_a - 32 * g_sumi_y_a;
                const int g_corr_b = g_sumi_b - 32 * g_sumi_y_b;
                const int u_corr_a = u_sumi_a - 32 * u_sumi_y_a;
                const int u_corr_b = u_sumi_b - 32 * u_sumi_y_b;

                sums_gate[c] += g_super_d * d8 *
                    ((float)(g_sc_a * g_corr_a + g_sc_b * g_corr_b));
                sums_up[c]   += u_super_d * d8 *
                    ((float)(u_sc_a * u_corr_a + u_sc_b * u_corr_b));
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
