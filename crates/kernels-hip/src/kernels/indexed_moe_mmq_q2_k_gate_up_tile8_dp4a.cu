// Q2_K MoE MMQ tile8 (gate_up). Two affine Q2_K weights against a shared
// per-token activation slab. Output indexed via the (token, slot) decoded
// from `sorted_pair_idx_padded[tile_n + c]`.

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
void flambeau_indexed_moe_mmq_q2_k_gate_up_tile8_dp4a_q8_1(
    const flambeau_block_q2_K* __restrict__ gate_w,
    const flambeau_block_q2_K* __restrict__ up_w,
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

    constexpr int q8_per_super = QK_K / QK8_1;

    float sums_gate[TILE_N];
    float sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) { sums_gate[c] = 0.0f; sums_up[c] = 0.0f; }

    (void) n_tokens;

    for (int ib = 0; ib < n_sb_per_row; ++ib) {
        float g_d = 0.0f, g_dmin = 0.0f;
        float u_d = 0.0f, u_dmin = 0.0f;
        uint8_t g_sc[16] = {0};
        uint8_t u_sc[16] = {0};
        const flambeau_block_q2_K* gbx = nullptr;
        const flambeau_block_q2_K* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_d    = (float) gbx->d;
            g_dmin = (float) gbx->dmin;
            u_d    = (float) ubx->d;
            u_dmin = (float) ubx->dmin;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                g_sc[j] = gbx->scales[j];
                u_sc[j] = ubx->scales[j];
            }
        }

        float g_sumf_d[TILE_N], g_sumf_m[TILE_N];
        float u_sumf_d[TILE_N], u_sumf_m[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            g_sumf_d[c] = 0.0f; g_sumf_m[c] = 0.0f;
            u_sumf_d[c] = 0.0f; u_sumf_m[c] = 0.0f;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int chunk_idx  = sub >> 2;
            const int shift_iter = sub & 3;
            const int shift      = 2 * shift_iter;

            int g_v[8] = {0};
            int u_v[8] = {0};
            if (row_ok) {
                const int* g_qs = (const int*) (gbx->qs + chunk_idx * 32);
                const int* u_qs = (const int*) (ubx->qs + chunk_idx * 32);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    g_v[j] = (g_qs[j] >> shift) & 0x03030303;
                    u_v[j] = (u_qs[j] >> shift) & 0x03030303;
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
                const int* y_packed = (const int*) by->qs;

                int g_sumi_a = 0, g_sumi_b = 0;
                int u_sumi_a = 0, u_sumi_b = 0;
                int sumi_y_a = 0, sumi_y_b = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    g_sumi_a = dp4a(g_v[j], y_packed[j], g_sumi_a);
                    u_sumi_a = dp4a(u_v[j], y_packed[j], u_sumi_a);
                    sumi_y_a = dp4a(0x01010101, y_packed[j], sumi_y_a);
                }
                #pragma unroll
                for (int j = 4; j < 8; ++j) {
                    g_sumi_b = dp4a(g_v[j], y_packed[j], g_sumi_b);
                    u_sumi_b = dp4a(u_v[j], y_packed[j], u_sumi_b);
                    sumi_y_b = dp4a(0x01010101, y_packed[j], sumi_y_b);
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
}
