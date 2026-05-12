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
void flambeau_indexed_moe_mmq_q5_k_gate_up_tile8_dp4a_q8_1(
    const flambeau_block_q5_K* __restrict__ gate_w,
    const flambeau_block_q5_K* __restrict__ up_w,
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
        float g_d = 0.0f, g_dmin = 0.0f;
        float u_d = 0.0f, u_dmin = 0.0f;
        uint8_t g_sub_sc[8] = {0};
        uint8_t g_sub_m [8] = {0};
        uint8_t u_sub_sc[8] = {0};
        uint8_t u_sub_m [8] = {0};
        const flambeau_block_q5_K* gbx = nullptr;
        const flambeau_block_q5_K* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_d    = (float) gbx->d;
            g_dmin = (float) gbx->dmin;
            u_d    = (float) ubx->d;
            u_dmin = (float) ubx->dmin;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                flambeau_q4k_scale_min(j, gbx->scales, &g_sub_sc[j], &g_sub_m[j]);
                flambeau_q4k_scale_min(j, ubx->scales, &u_sub_sc[j], &u_sub_m[j]);
            }
        }

        float g_sumf_d[TILE_N];
        float g_sumf_m[TILE_N];
        float u_sumf_d[TILE_N];
        float u_sumf_m[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            g_sumf_d[c] = 0.0f; g_sumf_m[c] = 0.0f;
            u_sumf_d[c] = 0.0f; u_sumf_m[c] = 0.0f;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int il    = sub >> 1;
            const int half  = sub & 1;
            const int s_bit = 2 * il + half;

            int g_v[8] = {0};
            int u_v[8] = {0};
            if (row_ok) {
                const int* g_ql_ptr = (const int*) (gbx->qs + 32 * il);
                const int* g_qh_ptr = (const int*) gbx->qh;
                const int* u_ql_ptr = (const int*) (ubx->qs + 32 * il);
                const int* u_qh_ptr = (const int*) ubx->qh;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int g_ql_word = g_ql_ptr[j];
                    const int g_qh_word = g_qh_ptr[j];
                    const int u_ql_word = u_ql_ptr[j];
                    const int u_qh_word = u_qh_ptr[j];

                    const int g_nib4 = (half == 0)
                        ? (g_ql_word & 0x0F0F0F0F)
                        : ((g_ql_word >> 4) & 0x0F0F0F0F);
                    const int u_nib4 = (half == 0)
                        ? (u_ql_word & 0x0F0F0F0F)
                        : ((u_ql_word >> 4) & 0x0F0F0F0F);

                    const int g_hi4 = ((g_qh_word >> s_bit) & 0x01010101) << 4;
                    const int u_hi4 = ((u_qh_word >> s_bit) & 0x01010101) << 4;

                    g_v[j] = g_nib4 | g_hi4;
                    u_v[j] = u_nib4 | u_hi4;
                }
            }

            const float g_sc_f = (float) g_sub_sc[sub];
            const float g_m_f  = (float) g_sub_m [sub];
            const float u_sc_f = (float) u_sub_sc[sub];
            const float u_m_f  = (float) u_sub_m [sub];

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_token[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const int* y_packed = (const int*) by->qs;

                int g_sumi_d = 0, g_sumi_y = 0;
                int u_sumi_d = 0, u_sumi_y = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = y_packed[j];
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
}
