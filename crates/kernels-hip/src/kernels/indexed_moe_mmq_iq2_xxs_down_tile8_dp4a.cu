// indexed_moe_mmq_iq2_xxs_down_tile8_dp4a — IQ2_XXS MoE MMQ tile8 down.

#include "block_quant.cuh"
#include "iq_grid.cuh"
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
static __device__ __forceinline__ int apply_signs_packed(int g_u32, int sign4) {
    int out = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int mag = (g_u32 >> (8 * j)) & 0xFF;
        const int neg = (sign4 >> j) & 1;
        const int v   = neg ? -mag : mag;
        out |= (v & 0xFF) << (8 * j);
    }
    return out;
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_indexed_moe_mmq_iq2_xxs_down_tile8_dp4a_q8_1(
    const flambeau_block_iq2_xxs* __restrict__ down_w,
    const flambeau_block_q8_1*    __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    float*      __restrict__ dst,
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
    const int expert     = expert_ids[first_pair];

    int slot_pair[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) slot_pair[c] = sorted_pair_idx_padded[tile_n + c];

    constexpr int q8_per_super = QK_K / QK8_1;
    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;
    (void) n_tokens; (void) top_k;

    for (int ib = 0; ib < n_sb_per_row; ++ib) {
        float d = 0.0f;
        const flambeau_block_iq2_xxs* bx = nullptr;
        if (row_ok) {
            bx = &down_w[((size_t) expert * n_rows + row) * n_sb_per_row + ib];
            d  = (float) bx->d;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int v[8] = {0};
            float sf = 0.0f;
            if (row_ok) {
                const uint8_t* sp = bx->qs + 8 * sub;
                const uint32_t aux0 = (uint32_t) sp[0] | ((uint32_t) sp[1] << 8)
                                    | ((uint32_t) sp[2] << 16) | ((uint32_t) sp[3] << 24);
                const uint32_t aux1 = (uint32_t) sp[4] | ((uint32_t) sp[5] << 8)
                                    | ((uint32_t) sp[6] << 16) | ((uint32_t) sp[7] << 24);
                sf = (0.5f + (float)(aux1 >> 28)) * 0.25f;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int idx = (int)((aux0 >> (8 * l)) & 0xFF);
                    const uint64_t g_u64 = IQ2XXS_GRID[idx];
                    const int g_lo = (int)(g_u64 & 0xFFFFFFFF);
                    const int g_hi = (int)((g_u64 >> 32) & 0xFFFFFFFF);
                    const uint8_t sb = KSIGNS_IQ2XS[(aux1 >> (7 * l)) & 0x7F];
                    v[2 * l + 0] = apply_signs_packed(g_lo, sb        & 0x0F);
                    v[2 * l + 1] = apply_signs_packed(g_hi, (sb >> 4) & 0x0F);
                }
            }

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_pair[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const int* y_packed = (const int*) by->qs;
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) sumi = dp4a(v[j], y_packed[j], sumi);
                sums[c] += d * d8 * (float) sumi * sf;
            }
        }
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        dst[(size_t) slot_pair[c] * n_rows + row] = sums[c];
    }
}
