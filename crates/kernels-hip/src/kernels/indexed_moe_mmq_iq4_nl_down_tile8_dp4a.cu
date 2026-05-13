// indexed_moe_mmq_iq4_nl_down_tile8_dp4a — IQ4_NL MoE MMQ tile8 down.
// Phase 4 Slice D. 32-elem blocks (no super-block).

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
void flambeau_indexed_moe_mmq_iq4_nl_down_tile8_dp4a_q8_1(
    const flambeau_block_iq4_nl* __restrict__ down_w,
    const flambeau_block_q8_1*   __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    float*      __restrict__ dst,
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

    int slot_pair[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) slot_pair[c] = sorted_pair_idx_padded[tile_n + c];

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;
    (void) n_tokens; (void) top_k;

    for (int ib = 0; ib < n_blocks_per_row; ++ib) {
        float d = 0.0f;
        int v[8] = {0};
        if (row_ok) {
            const flambeau_block_iq4_nl* bx =
                &down_w[((size_t) expert * n_rows + row) * n_blocks_per_row + ib];
            d = (float) bx->d;
            const int* ql = (const int*) bx->qs;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int w = ql[j];
                v[j]     = pack_iq4_lut(w        & 0x0F0F0F0F);
                v[j + 4] = pack_iq4_lut((w >> 4) & 0x0F0F0F0F);
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const flambeau_block_q8_1* by = &y[(size_t) slot_pair[c] * n_blocks_per_row + ib];
            const float d8 = (float) by->d;
            const int* y_packed = (const int*) by->qs;
            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) sumi = dp4a(v[j], y_packed[j], sumi);
            sums[c] += d * d8 * (float) sumi;
        }
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        dst[(size_t) slot_pair[c] * n_rows + row] = sums[c];
    }
}
