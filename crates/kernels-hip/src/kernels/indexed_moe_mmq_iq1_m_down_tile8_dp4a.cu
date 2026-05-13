// indexed_moe_mmq_iq1_m_down_tile8_dp4a — IQ1_M MoE MMQ tile8 down.
// IQ1_M has no per-block `d` — reassembled from spread nibbles in `scales`.
// Per-l (dl, delta) vary; per-l sum_qi via dp4a-with-ones. Phase 4 Slice D.

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
static __device__ __forceinline__ float iq1m_reassemble_d(const uint8_t* __restrict__ scales) {
    const int sc0 = (int) scales[0] | ((int) scales[1] << 8);
    const int sc1 = (int) scales[2] | ((int) scales[3] << 8);
    const int sc2 = (int) scales[4] | ((int) scales[5] << 8);
    const int sc3 = (int) scales[6] | ((int) scales[7] << 8);
    const int d_bits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0)
                     | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000);
    fb_fp16_t d_fp16 = *reinterpret_cast<const fb_fp16_t*>(&d_bits);
    return (float) d_fp16;
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_indexed_moe_mmq_iq1_m_down_tile8_dp4a_q8_1(
    const flambeau_block_iq1_m* __restrict__ down_w,
    const flambeau_block_q8_1*  __restrict__ y,
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
        const flambeau_block_iq1_m* bx = nullptr;
        if (row_ok) {
            bx = &down_w[((size_t) expert * n_rows + row) * n_sb_per_row + ib];
            d  = iq1m_reassemble_d(bx->scales);
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int v[8] = {0};
            float dl_l[4] = {0};
            float delta_l[4] = {0};
            if (row_ok) {
                const int sw = (int) bx->scales[2 * (sub >> 1)]
                             | ((int) bx->scales[2 * (sub >> 1) + 1] << 8);
                const int sh1 = 6 * (sub & 1);
                const int sh2 = sh1 + 3;
                const float dl1 = d * (2.0f * (float)((sw >> sh1) & 7) + 1.0f);
                const float dl2 = d * (2.0f * (float)((sw >> sh2) & 7) + 1.0f);
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int qh_pick    = (l < 2) ? (2 * sub) : (2 * sub + 1);
                    const uint8_t qh     = bx->qh[qh_pick];
                    const int shift_idx  = 8 - 4 * (l & 1);
                    const int idx = (int) bx->qs[4 * sub + l]
                                  | (((int) qh << shift_idx) & 0x700);
                    const int delta_bit  = (l & 1) == 0 ? 0x08 : 0x80;
                    dl_l[l]    = (l < 2) ? dl1 : dl2;
                    delta_l[l] = (qh & delta_bit) ? -IQ1_DELTA : IQ1_DELTA;
                    const uint64_t g_u64 = IQ1S_GRID[idx];
                    v[2 * l + 0] = (int)(g_u64 & 0xFFFFFFFF);
                    v[2 * l + 1] = (int)((g_u64 >> 32) & 0xFFFFFFFF);
                }
            }

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_pair[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const int* y_packed = (const int*) by->qs;
                float partial = 0.0f;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    int sumi = 0, sumi_y = 0;
                    sumi = dp4a(v[2 * l + 0], y_packed[2 * l + 0], sumi);
                    sumi = dp4a(v[2 * l + 1], y_packed[2 * l + 1], sumi);
                    sumi_y = dp4a(0x01010101, y_packed[2 * l + 0], sumi_y);
                    sumi_y = dp4a(0x01010101, y_packed[2 * l + 1], sumi_y);
                    partial += dl_l[l] * d8 * ((float) sumi + delta_l[l] * (float) sumi_y);
                }
                sums[c] += partial;
            }
        }
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        dst[(size_t) slot_pair[c] * n_rows + row] = sums[c];
    }
}
