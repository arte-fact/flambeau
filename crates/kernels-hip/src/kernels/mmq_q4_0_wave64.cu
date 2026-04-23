// mmq_q4_0_wave64 — V2.28.a wave64 MMQ for Q4_0 × Q8_1 activation.
//
// V2.27 head-to-head bench showed Qwen3.6-35B-A3B-Q4_0 prefill at 12% of
// llama.cpp (132 vs 1118 tok/s) because V2.23 shipped Q4_0 MMVQ only and
// `qmatmul()` falls back to row-by-row at L>1. Closing that gap needs a
// proper MMQ tile. This kernel is the direct Q4_0 sibling of
// `mmq_q4_1_wave64.cu` — identical structure with the simpler no-min
// bias-correction identity (V2.23 MMVQ already uses this form):
//
//   dot = Σ_i (d · (q_i - 8)) · d_y · q8_i
//       = d · d_y · Σ q_i · q8_i  -  8 · d · d_y · Σ q8_i
//       = d · d_y · sumi          -  8 · d · y_s
//       = d · (d_y · sumi - 8 · y_s)
//
// where sumi = DP4A(q_packed, y_packed) and y_s = by->s = d_y · Σ q8_i
// (pre-computed in the Q8_1 block header).
//
// Tile shape (same family as `mmq_q4_1_wave64` / `mmq_q5_K_wave64`):
//   MMQ_Y  = 64 (one wave64; each thread = 1 output row)
//   TILE_N = 8  (8 output cols per block, loop-unrolled)
//   Grid   = (⌈nrows_x / 64⌉, ⌈ncols_y / 8⌉)
//   Block  = 64 threads
//
// Q4_0 block layout (18 bytes, 32 elements):
//   d  fp16       — scale
//   qs uint8[16]  — nibble pairs; byte i's LOW nibble = element i, HIGH = i+16
//
// Reconstruction: x_real_i = d · (q_i - 8)  (q_i unsigned in [0, 15]).
//
// V2.10.b lesson: no per-block sumf_* transient arrays — fold the correction
// into the persistent `sums[c]` inside the inner loop.

#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef QK4_0
#define QK4_0 32
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
void flambeau_mmq_q4_0_wave64_q8_1(
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

    const flambeau_block_q4_0* x = (const flambeau_block_q4_0*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK4_0;
    const int blocks_per_col_y = nrows_y / QK8_1;
    // Q4_0 and Q8_1 both have 32-element blocks → blocks_per_row_x == blocks_per_col_y.

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int b = 0; b < blocks_per_row_x; ++b) {
        // Decode Q4_0 weight block for this thread's row.
        float x_d = 0.0f;
        int v[8] = {0};
        if (row_ok) {
            const flambeau_block_q4_0* bx =
                &x[(size_t) row * blocks_per_row_x + b];
            x_d = (float) bx->d;
            const int* ql_words = (const int*) bx->qs;  // 4 int32 = 16 bytes

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int qw = ql_words[j];
                v[j]     = (qw >> 0) & 0x0F0F0F0F;      // elements 0..15 (low nibbles)
                v[j + 4] = (qw >> 4) & 0x0F0F0F0F;      // elements 16..31 (high nibbles)
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const int col = tile_n + c;
            if (col >= ncols_y) break;

            const flambeau_block_q8_1* by =
                &y[(size_t) col * blocks_per_col_y + b];
            const float y_d = (float) by->d;
            const float y_s = (float) by->s;           // = y_d · Σ q8 (pre-computed)
            const int* y_packed = (const int*) by->qs;

            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sumi = dp4a(v[j], y_packed[j], sumi);
            }

            // Fold d·(d_y·sumi − 8·y_s) directly into persistent sum.
            // Equivalent to: x_d·y_d·sumi + x_d·(−8·y_s).
            sums[c] += x_d * (y_d * ((float) sumi) - 8.0f * y_s);
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
