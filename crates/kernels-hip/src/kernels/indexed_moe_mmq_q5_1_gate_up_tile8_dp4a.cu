#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef QK5_1
#define QK5_1 32
#endif
#ifndef QK8_1
#define QK8_1 32
#endif

#define MMQ_Y 64
#define TILE_N 8

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ int expand_bits8(unsigned int qh, int start) {
    int out = 0;
    out |= ((qh >> (start + 0)) & 1u);
    out |= ((qh >> (start + 1)) & 1u) << 8;
    out |= ((qh >> (start + 2)) & 1u) << 16;
    out |= ((qh >> (start + 3)) & 1u) << 24;
    return out;
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_indexed_moe_mmq_q5_1_gate_up_tile8_dp4a_q8_1(
    const flambeau_block_q5_1* __restrict__ gate_w,
    const flambeau_block_q5_1* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    float*      __restrict__ gate_out,
    float*      __restrict__ up_out,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_blocks_per_row,
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

    const int blocks_per_row_x = n_blocks_per_row;

    float sums_gate[TILE_N];
    float sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) { sums_gate[c] = 0.0f; sums_up[c] = 0.0f; }

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float g_d = 0.0f, g_m = 0.0f;
        float u_d = 0.0f, u_m = 0.0f;
        int g_nib[8] = {0};
        int g_bit[8] = {0};
        int u_nib[8] = {0};
        int u_bit[8] = {0};
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * blocks_per_row_x + ib;
            const flambeau_block_q5_1* gbx = &gate_w[w_row_off];
            const flambeau_block_q5_1* ubx = &up_w[w_row_off];
            g_d = (float) gbx->d;
            g_m = (float) gbx->m;
            u_d = (float) ubx->d;
            u_m = (float) ubx->m;
            const int* g_ql = (const int*) gbx->qs;
            const int* u_ql = (const int*) ubx->qs;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int gw = g_ql[j];
                const int uw = u_ql[j];
                g_nib[j]     = (gw >> 0) & 0x0F0F0F0F;
                g_nib[j + 4] = (gw >> 4) & 0x0F0F0F0F;
                u_nib[j]     = (uw >> 0) & 0x0F0F0F0F;
                u_nib[j + 4] = (uw >> 4) & 0x0F0F0F0F;
            }
            const unsigned int g_qh = *((const unsigned int*) gbx->qh);
            const unsigned int u_qh = *((const unsigned int*) ubx->qh);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                g_bit[j]     = expand_bits8(g_qh, j * 4);
                g_bit[j + 4] = expand_bits8(g_qh, j * 4 + 16);
                u_bit[j]     = expand_bits8(u_qh, j * 4);
                u_bit[j + 4] = expand_bits8(u_qh, j * 4 + 16);
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const flambeau_block_q8_1* by =
                &y[(size_t) slot_token[c] * n_blocks_per_row + ib];
            const float d8 = (float) by->d;
            const float s8 = (float) by->s;
            const int* y_packed = (const int*) by->qs;

            int sumi_g_nib = 0, sumi_g_bit = 0;
            int sumi_u_nib = 0, sumi_u_bit = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int y_j = y_packed[j];
                sumi_g_nib = dp4a(g_nib[j], y_j, sumi_g_nib);
                sumi_g_bit = dp4a(g_bit[j], y_j, sumi_g_bit);
                sumi_u_nib = dp4a(u_nib[j], y_j, sumi_u_nib);
                sumi_u_bit = dp4a(u_bit[j], y_j, sumi_u_bit);
            }
            sums_gate[c] += g_d * d8 * (float) (sumi_g_nib + 16 * sumi_g_bit) + g_m * s8;
            sums_up[c]   += u_d * d8 * (float) (sumi_u_nib + 16 * sumi_u_bit) + u_m * s8;
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
