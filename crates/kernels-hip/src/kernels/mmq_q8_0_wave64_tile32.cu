// mmq_q8_0_wave64_tile32 — V2.31.d experimental TILE_N=32 variant.
//
// Port of V2.7 `mmq_q8_0_wave64_tile16` with the inner col-tile width
// doubled 16 → 32. Rationale: at tile16 V2.7 measured
// MemUnitBusy≈83 % on 27B prefill shapes — still memory-bound. Each
// weight tile is decoded once per thread and reused across TILE_N
// activation columns; doubling TILE_N halves weight HBM bandwidth
// again (32 cols per decoded tile vs 16).
//
// Risk (per V2.9.b lesson): per-thread FP32 accumulator count doubles
// 16 → 32. VGPR pressure goes from ~32 to ~60 per thread. On gfx906
// at __launch_bounds__(WARP_SIZE, 1), scratch spill possible. Opt-in
// via FLAMBEAU_VARIANT=q8_tile32 — compare before committing.

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
#define TILE_N 32

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_q8_0_wave64_tile32_q8_1(
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

        // Inner col loop: 32 cols share the same decoded weight above.
        // Do NOT `#pragma unroll` — would explode the 8-dp4a inner
        // loop 32× and push VGPR past wave-threshold. Let the compiler
        // schedule.
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

    for (int c = 0; c < TILE_N; ++c) {
        const int col = tile_n + c;
        if (col < ncols_y && row < nrows_dst) {
            dst[(size_t) col * nrows_dst + row] = sums[c];
        }
    }
}
