// indexed_moe_mmq_q4_0_down_tile8_dp4a — down-projection MoE MMQ
// for Q4_0. Same tile8 structure as the gate+up sibling, but:
// - one weight tensor (`ffn_down_exps`, not fused)
// - activation is per-PAIR: `y[pair_idx, :]` indexed directly by sorted_pair_idx
// (treats each (token, slot) as its own "effective token" with top_k=1)
// Matches `indexed_moe_mmq_q4_k_down_tile8_dp4a.cu` contract
// with Q4_0 weight decode.

#include "block_quant.cuh"
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

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_indexed_moe_mmq_q4_0_down_tile8_dp4a_q8_1(
    const flambeau_block_q4_0* __restrict__ w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    float*      __restrict__ dst,
    const int n_rows,
    const int n_tokens,             // = n_pairs for the down path
    const int top_k,                // = 1 for the down path
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

    int slot_pair[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        slot_pair[c] = sorted_pair_idx_padded[tile_n + c];
    }

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < n_blocks_per_row; ++ib) {
        float w_d = 0.0f;
        int w_v[8] = {0};
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_blocks_per_row + ib;
            const flambeau_block_q4_0* wbx = &w[w_row_off];
            w_d = (float) wbx->d;
            const int* ql = (const int*) wbx->qs;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int qw = ql[j];
                w_v[j]     = (qw >> 0) & 0x0F0F0F0F;
                w_v[j + 4] = (qw >> 4) & 0x0F0F0F0F;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const flambeau_block_q8_1* by =
                &y[(size_t) slot_pair[c] * n_blocks_per_row + ib];
            const float d8 = (float) by->d;
            const float s8 = (float) by->s;
            const int* y_packed = (const int*) by->qs;

            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sumi = dp4a(w_v[j], y_packed[j], sumi);
            }
            sums[c] += w_d * (d8 * (float) sumi - 8.0f * s8);
        }
    }

    if (!row_ok) return;

    // Down-projection output layout: [n_pairs, n_rows] (top_k_inner=1).
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_pair[c] * n_rows + row;
        dst[out_idx] = sums[c];
    }
    (void) top_k; (void) n_tokens;
}
