// mmq_q8_0_wave64 — wave64 MMQ for Q8_0 × Q8_1 activation.
// alternative to `mmq_q8_0_4warp` targeting the dense-attention
// Q8_0 MMQ path (attn_q/k/v/o on Qwen3.6-35B, ~310 calls × 1.1 ms = 337 ms
// / 19 % of prefill time). Same wave64 MMQ_Y=64 × TILE_N=8 pattern used
// by /K-quant kernels, but simpler dequant: Q8_0 blocks
// have just `d(fp16) + qs[32 int8]`, no super-block, no min, no qh.
// Tile shape:
// MMQ_Y = 64 (one wave64 per output-row quad; each thread = 1 row)
// TILE_N = 8 (8 output cols per tile, loop-unrolled per thread)
// Grid = (⌈nrows_x / 64⌉, ⌈ncols_y / 8⌉)
// Block = 64 threads (one warp)
// Q8_0 dequant math:
// per K=32 block: y_j = d_x * d_y * Σ_i (x_i * y_i)
// DP4A packs 4 int8×int8 per call → QK8_0=32 means 8 DP4A per block.
// Args (8 scalar + 3 ptr — matches K-quant wave64 signature for the shared
// `mmq_wave64_launch` path):
// vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst

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
#define TILE_N 8

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_q8_0_wave64_q8_1(
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

    const int blocks_per_row_x = ncols_x / QK8_0;    // Q8_0 blocks per row
    const int blocks_per_col_y = nrows_y / QK8_1;    // Q8_1 blocks per col
    // For Q8_0/Q8_1 with same block size: blocks_per_row_x == blocks_per_col_y.

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int b = 0; b < blocks_per_row_x; ++b) {
        // Weight block for this row.
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
