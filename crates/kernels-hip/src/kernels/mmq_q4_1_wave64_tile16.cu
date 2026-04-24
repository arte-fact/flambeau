// mmq_q4_1_wave64_tile16 — V2.29.e TILE_N=16 variant of Q4_1 wave64 MMQ.
//
// Direct port of the V2.7 TILE_N=16 pattern from
// `mmq_q8_0_wave64_tile16` to Q4_1's nibble-pair layout + per-block
// (d, m) reconstruction.
//
// Motivation from V2.29.a audit: `flambeau_mmq_q4_1_4warp_lds_q8_1`
// is 40.8 % of 9B Q4_1 Mesh<4> prefill wall time (5137 ms / 12600 ms
// total kernel time). Biggest single kernel in the run. The Q8_0
// tile16 pattern halves weight-HBM bandwidth by reusing each decoded
// weight tile across 16 output columns instead of 8 → direct
// analogue for Q4_1.
//
// Tile shape:
//   MMQ_Y  = 64 (one wave64; each thread = 1 output row)
//   TILE_N = 16 (16 output cols per block)
//   Grid   = (⌈nrows_x / 64⌉, ⌈ncols_y / 16⌉)
//   Block  = 64 threads (one wave)
//
// Per-thread VGPR at TILE_N=16:
//   sums[16]: 16 float accumulators        (16 VGPR)
//   v[8]:     decoded Q4_1 weights         (8 VGPR)
//   x_d, x_m: per-block scale + min        (2 VGPR)
//   loop vars + intermediates              (~8 VGPR)
//   total ≈ 34 VGPR → fits comfortably in 2 waves/SIMD (gfx906 has
//   256 VGPR/SIMD, 1 wave = 64 threads × 34 VGPR = 2176 / 4 SIMD per
//   CU ≈ 544 VGPR cost / 1024 avail per CU → fine).
//
// Q4_1 block layout (20 bytes, 32 elements):
//   d  fp16  — scale
//   m  fp16  — min offset
//   qs uint8[16] — nibble pairs; byte i's LOW nibble = element i,
//                  HIGH = element i+16.
//
// Reconstruction: x_real_i = d · q_i + m  (q_i unsigned in [0, 15]).
//
// Dot with Q8_1:
//   dot = d · d_y · sumi + m · y_s
// where sumi = DP4A(q_packed, y_packed) and y_s = by->s (= d_y · Σ q8).
//
// Per-block DP4A pack (same as wave64 TILE_N=8 parent):
//   v[0..3] = low nibbles of qs[0..15]  (elements 0..15, 4-per-int32)
//   v[4..7] = high nibbles of qs[0..15] (elements 16..31)
//
// V2.10.b lesson applied: no per-block sumf_* transient arrays. Fold
// `d · d_y · sumi + m · y_s` into persistent `sums[c]` in the inner
// loop.

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
#define TILE_N 16

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_q4_1_wave64_tile16_q8_1(
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

    const flambeau_block_q4_1* x = (const flambeau_block_q4_1*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK4_1;
    const int blocks_per_col_y = nrows_y / QK8_1;
    // Both 32-element blocks → blocks_per_row_x == blocks_per_col_y.

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int b = 0; b < blocks_per_row_x; ++b) {
        // Decode Q4_1 weight block for this thread's row ONCE. The
        // decoded v[0..7] is then reused across 16 output columns
        // below (vs 8 in TILE_N=8) → halves weight HBM bandwidth.
        float x_d = 0.0f, x_m = 0.0f;
        int v[8] = {0};
        if (row_ok) {
            const flambeau_block_q4_1* bx =
                &x[(size_t) row * blocks_per_row_x + b];
            x_d = (float) bx->d;
            x_m = (float) bx->m;
            const int* ql_words = (const int*) bx->qs;  // 4 int32 = 16 bytes

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int qw = ql_words[j];
                v[j]     = (qw >> 0) & 0x0F0F0F0F;  // elements 0..15 (low nibbles)
                v[j + 4] = (qw >> 4) & 0x0F0F0F0F;  // elements 16..31 (high nibbles)
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
            const float y_s = (float) by->s;
            const int* y_packed = (const int*) by->qs;

            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sumi = dp4a(v[j], y_packed[j], sumi);
            }

            // Fold d·d_y·sumi + m·y_s into persistent sum.
            sums[c] += x_d * y_d * ((float) sumi) + x_m * y_s;
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
