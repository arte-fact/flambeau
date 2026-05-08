// indexed_moe_mmq_q4_1_down_tile8_dp4a — down-projection
// MoE MMQ for Q4_1 weights × Q8_1 activation.
// Direct port of `indexed_moe_mmq_q4_0_down_tile8_dp4a.cu` (8.c) with
// the Q4_1 reconstruction: `y_real = q · d + m`. The dot product per
// block becomes `d_x · d_y · sumi + m_x · s_y` instead of Q4_0's bias-
// correction `d_x · (d_y · sumi - 8 · s_y)`.
// Used by Coder-Next-Q4_0 where `ffn_down_exps` is Q4_1 (gate/up are
// Q4_0). Pre-this-kernel, the PP and TP MoE prefill paths fell through
// to MMVQ-per-token because `q4_0_use_tile8` rejects Q4_1 down (only
// accepts Q4_0/Q8_0). With this kernel landed both PP and TP can take
// the tile8 fast path on Coder-Next at L≥32.

#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef QK4_1
#define QK4_1 32
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
void flambeau_indexed_moe_mmq_q4_1_down_tile8_dp4a_q8_1(
    const flambeau_block_q4_1* __restrict__ w,
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
        float w_m = 0.0f;
        int w_v[8] = {0};
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_blocks_per_row + ib;
            const flambeau_block_q4_1* wbx = &w[w_row_off];
            w_d = (float) wbx->d;
            w_m = (float) wbx->m;
            // Q4_1 nibble layout matches Q4_0: 16 packed bytes,
            // low nibble at byte i → element i, high nibble → element i+16.
            // The DP4A path reads 4 packed ints (4 bytes each) and splits
            // each into low (>>0) and high (>>4) nibble groups.
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
            // Q4_1 affine: y_real · w_real = d_x·d_y·sumi + m_x·s_y.
            // s_y already encodes d_y · sum(q8) per ggml's Q8_1 convention.
            sums[c] += w_d * d8 * (float) sumi + w_m * s8;
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
