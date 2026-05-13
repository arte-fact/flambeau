// mmq_q8_K_wave64 — wave64 MMQ for Q8_K weights × Q8_1 activation.
// Mirrors the Q5_K/Q6_K wave64 structure (MMQ_Y=64, TILE_N=8, 64 threads).
// Q8_K decode is the simplest K-quant: f32 d × i8 qs[256], no nibble unpack,
// no per-sub-block scale, no min. Each super-block contributes
//   sums[c] += d_x * Σ_{sub=0..7} d_y[sub] * dot32(qs_sub, y_qs_sub)
//
// Tile shape:
//   MMQ_Y  = 64   (one wave64; thread = output row)
//   TILE_N = 8    (output cols per tile, unrolled)
//   Grid   = (⌈nrows_x / 64⌉, ⌈ncols_y / 8⌉)
//   Block  = 64 threads
//
// Args (same scalar + ptr layout as the K-quant wave64 family):
//   vx, vy, dst, ncols_x (K), nrows_x (N), ncols_y (M), nrows_y (K), nrows_dst.

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

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_q8_K_wave64_q8_1(
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

    const flambeau_block_q8_K* x = (const flambeau_block_q8_K*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float x_d = 0.0f;
        const flambeau_block_q8_K* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            x_d = bx->d;
        }

        float sumf[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) sumf[c] = 0.0f;

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int x_packed[8] = {0};
            if (row_ok) {
                const int* qs_ptr = (const int*) (bx->qs + sub * 32);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    x_packed[j] = qs_ptr[j];
                }
            }

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int col = tile_n + c;
                if (col >= ncols_y) break;

                const flambeau_block_q8_1* by =
                    &y[(size_t) col * blocks_per_col_y + ib * q8_per_super + sub];
                const float d_y = (float) by->d;
                const int* y_packed = (const int*) by->qs;

                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sumi = dp4a(x_packed[j], y_packed[j], sumi);
                }
                sumf[c] += d_y * (float) sumi;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            sums[c] += x_d * sumf[c];
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
