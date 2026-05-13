// indexed_moe_mmq_iq4_nl_gate_up_tile8_dp4a — IQ4_NL MoE MMQ tile8 gate+up.
// 32-elem blocks (no super-block); inner loops one block at a time vs
// the 8-sub-block IQ4_XS / IQ3_S variants. Phase 4 Slice C.

#include "block_quant.cuh"
#include "iq_grid.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef QK4_0
#define QK4_0 32
#endif
#ifndef QK8_1
#define QK8_1 32
#endif

#define MMQ_Y 64
#define TILE_N 8

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}
static __device__ __forceinline__ int pack_iq4_lut(int nibbles) {
    const int b0 = (int) flambeau_iq4nl_lut(nibbles & 0xFF);
    const int b1 = (int) flambeau_iq4nl_lut((nibbles >>  8) & 0xFF);
    const int b2 = (int) flambeau_iq4nl_lut((nibbles >> 16) & 0xFF);
    const int b3 = (int) flambeau_iq4nl_lut((nibbles >> 24) & 0xFF);
    return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_indexed_moe_mmq_iq4_nl_gate_up_tile8_dp4a_q8_1(
    const flambeau_block_iq4_nl* __restrict__ gate_w,
    const flambeau_block_iq4_nl* __restrict__ up_w,
    const flambeau_block_q8_1*   __restrict__ y,
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
    const int expert     = expert_ids[first_pair];

    int slot_token[TILE_N];
    int slot_out_idx[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int pair = sorted_pair_idx_padded[tile_n + c];
        const int t = pair / top_k;
        const int s = pair - t * top_k;
        slot_token[c]   = t;
        slot_out_idx[c] = t * top_k + s;
    }

    float sums_gate[TILE_N];
    float sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) { sums_gate[c] = 0.0f; sums_up[c] = 0.0f; }

    (void) n_tokens;

    for (int ib = 0; ib < n_blocks_per_row; ++ib) {
        float g_d = 0.0f, u_d = 0.0f;
        int g_v[8] = {0};
        int u_v[8] = {0};
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_blocks_per_row + ib;
            const flambeau_block_iq4_nl* gbx = &gate_w[w_row_off];
            const flambeau_block_iq4_nl* ubx = &up_w[w_row_off];
            g_d = (float) gbx->d;
            u_d = (float) ubx->d;
            const int* g_ql = (const int*) gbx->qs;
            const int* u_ql = (const int*) ubx->qs;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int gw = g_ql[j];
                const int uw = u_ql[j];
                g_v[j]     = pack_iq4_lut(gw        & 0x0F0F0F0F);
                g_v[j + 4] = pack_iq4_lut((gw >> 4) & 0x0F0F0F0F);
                u_v[j]     = pack_iq4_lut(uw        & 0x0F0F0F0F);
                u_v[j + 4] = pack_iq4_lut((uw >> 4) & 0x0F0F0F0F);
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const flambeau_block_q8_1* by = &y[(size_t) slot_token[c] * n_blocks_per_row + ib];
            const float d8 = (float) by->d;
            const int* y_packed = (const int*) by->qs;
            int g_sumi = 0, u_sumi = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                g_sumi = dp4a(g_v[j], y_packed[j], g_sumi);
                u_sumi = dp4a(u_v[j], y_packed[j], u_sumi);
            }
            sums_gate[c] += g_d * d8 * (float) g_sumi;
            sums_up[c]   += u_d * d8 * (float) u_sumi;
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
