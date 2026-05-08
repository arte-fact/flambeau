// mmq_q8_0_wave64_tile16 — TILE_N=16 variant of Q8_0 wave64 MMQ.
// At TILE_N=8 on Qwen3.6-35B prefill shapes the kernel is HBM-bound
// (MemUnitBusy ~97%, VALUBusy ~19%): grid.y = ncols_y/8 = 64 col-blocks
// all read the same weight rows, and the 8 MiB weight matrix doesn't
// fit in L2 (4 MiB per GPU) → ~72× redundant weight reads.
// TILE_N=16 doubles activations per block, so the weight tile decodes
// once per thread and is reused across 16 cols. Halves weight HBM
// bandwidth. Per-thread VGPR grows modestly (16 accumulators vs 8) but
// waves/SIMD stays at 8 (32 VGPR/wave is within budget).

#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef QK8_0
#define QK8_0 32
#endif
#ifndef QK8_1
#define QK8_1 32
#endif

#define MMQ_Y 64
#define TILE_N 16

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_q8_0_wave64_tile16_q8_1(
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

    const flambeau_block_q8_0* x = (const flambeau_block_q8_0*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK8_0;
    const int blocks_per_col_y = nrows_y / QK8_1;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int b = 0; b < blocks_per_row_x; ++b) {
        float x_d = 0.0f;
        int x_packed[8] = {0};
        if (row_ok) {
            const flambeau_block_q8_0* bx =
                &x[(size_t) row * blocks_per_row_x + b];
            x_d = (float) bx->d;
            const int* qs_words = (const int*) bx->qs;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                x_packed[j] = qs_words[j];
            }
        }

        // Inner col loop: 16 cols share the same decoded weight above.
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const int col = tile_n + c;
            if (col >= ncols_y) break;

            const flambeau_block_q8_1* by =
                &y[(size_t) col * blocks_per_col_y + b];
            const float y_d = (float) by->d;
            const int* y_packed = (const int*) by->qs;

            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sumi = dp4a(x_packed[j], y_packed[j], sumi);
            }
            sums[c] += x_d * y_d * (float) sumi;
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
