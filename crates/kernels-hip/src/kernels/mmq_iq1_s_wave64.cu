// mmq_iq1_s_wave64 — wave64 MMQ for IQ1_S × Q8_1 activation.
// IQ1_S grid stores SIGNED i8 already, plus a per-sub-block ±delta offset.
// Reconstruction: y = dl * (g_i8 + delta). Expanding:
//   partial = dl * (g_i8 + delta) * (d_y * qi)
//           = dl * d_y * sum(g_i8 * qi) + dl * delta * d_y * sum(qi)
//           = dl * (d_y * sumi + delta * s)
// where `s = d_y * sum(qi)` is stored in `by->s` (Q8_1 second f16 field).
// Per sub-block: 4 × 11-bit codebook indices into IQ1S_GRID (2048 × u64).

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

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_iq1_s_wave64_q8_1(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float*      __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst
) {
    const int tile_m = blockIdx.x * WARP_SIZE;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid    = threadIdx.x;

    const int row     = tile_m + tid;
    const bool row_ok = (row < nrows_x);

    const flambeau_block_iq1_s* x = (const flambeau_block_iq1_s*) vx;
    const flambeau_block_q8_1*  y = (const flambeau_block_q8_1*)  vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f;
        const flambeau_block_iq1_s* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d = (float) bx->d;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int v[8] = {0};
            float dl = 0.0f;
            float delta = 0.0f;
            if (row_ok) {
                const int qh_u16 = (int) bx->qh[2 * sub]
                                 | ((int) bx->qh[2 * sub + 1] << 8);
                dl = super_d * (2.0f * (float)((qh_u16 >> 12) & 7) + 1.0f);
                delta = (qh_u16 & 0x8000) ? -IQ1_DELTA : IQ1_DELTA;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int idx = (int) bx->qs[4 * sub + l]
                                  | (((qh_u16 >> (3 * l)) & 7) << 8);
                    const uint64_t g_u64 = IQ1S_GRID[idx];
                    v[2 * l + 0] = (int)(g_u64 & 0xFFFFFFFF);
                    v[2 * l + 1] = (int)((g_u64 >> 32) & 0xFFFFFFFF);
                }
            }

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int col = tile_n + c;
                if (col >= ncols_y) break;
                const flambeau_block_q8_1* by =
                    &y[(size_t) col * blocks_per_col_y + ib * q8_per_super + sub];
                const float d_y = (float) by->d;
                const float s_y = (float) by->s;
                const int* y_packed = (const int*) by->qs;
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) sumi = dp4a(v[j], y_packed[j], sumi);
                sums[c] += dl * (d_y * (float) sumi + delta * s_y);
            }
        }
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int col = tile_n + c;
        if (col < ncols_y && row < nrows_dst) {
            dst[(size_t) col * nrows_dst + row] = sums[c];
        }
    }
}
