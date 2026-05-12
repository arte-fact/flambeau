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
void flambeau_indexed_moe_mmq_q5_1_down_tile8_dp4a_q8_1(
    const flambeau_block_q5_1* __restrict__ w,
    const flambeau_block_q8_1* __restrict__ y,
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
        float w_d = 0.0f, w_m = 0.0f;
        int w_nib[8] = {0};
        int w_bit[8] = {0};
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_blocks_per_row + ib;
            const flambeau_block_q5_1* wbx = &w[w_row_off];
            w_d = (float) wbx->d;
            w_m = (float) wbx->m;
            const int* ql = (const int*) wbx->qs;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int qw = ql[j];
                w_nib[j]     = (qw >> 0) & 0x0F0F0F0F;
                w_nib[j + 4] = (qw >> 4) & 0x0F0F0F0F;
            }
            const unsigned int qh = *((const unsigned int*) wbx->qh);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                w_bit[j]     = expand_bits8(qh, j * 4);
                w_bit[j + 4] = expand_bits8(qh, j * 4 + 16);
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const flambeau_block_q8_1* by =
                &y[(size_t) slot_pair[c] * n_blocks_per_row + ib];
            const float d8 = (float) by->d;
            const float s8 = (float) by->s;
            const int* y_packed = (const int*) by->qs;

            int sumi_nib = 0;
            int sumi_bit = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sumi_nib = dp4a(w_nib[j], y_packed[j], sumi_nib);
                sumi_bit = dp4a(w_bit[j], y_packed[j], sumi_bit);
            }
            sums[c] += w_d * d8 * (float) (sumi_nib + 16 * sumi_bit) + w_m * s8;
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
