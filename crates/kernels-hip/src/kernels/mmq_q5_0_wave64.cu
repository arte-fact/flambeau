// mmq_q5_0_wave64 — V2.30.a wave64 MMQ for Q5_0 × Q8_1 activation.
//
// Direct sibling of `mmq_q4_0_wave64.cu` (V2.28.a) with the 5th-bit ladder
// from V2.23's `mmvq_q5_0.cu` grafted into the inner loop. Q5_0 closes a
// low-traffic gap in Qwen3.6-35B-A3B-Q4_0 (20/40 layers use Q5_0 for
// ffn_*_shexp); the V2.27 head-to-head never measured it as a primary
// bottleneck, but it's a cheap port and clears the dtype out of MMVQ-only
// status.
//
// Q5_0 block layout (22 bytes, 32 elements):
//   d  fp16       — scale
//   qh uint8[4]   — 32 "5th bits", one per element (packed LSB-first)
//   qs uint8[16]  — nibble pairs; byte i's LOW nibble = element i, HIGH = i+16
//
// Reconstruction: x_real_i = d · (q5_i - 16)  where q5_i = (qh_i << 4) | nibble_i.
//
// Identity (exactly mirrors V2.23 mmvq_q5_0 arithmetic, lifted to tile-M):
//   (q5 - 16) · y  =  nibble · y  +  16·bit · y  -  16·y
//                  =  dp4a(nibble, y)  +  16·dp4a(bit, y)  -  16·s_y
// per block, where s_y = d_y · Σ q8 is the Q8_1 pre-computed sum.
//
// Tile shape (same family as mmq_q4_0_wave64):
//   MMQ_Y  = 64, TILE_N = 8, 64 threads / block (1 wave64), 1 output row/thread.

#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef QK5_0
#define QK5_0 32
#endif
#ifndef QK8_1
#define QK8_1 32
#endif

#define MMQ_Y 64
#define TILE_N 8

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Expand 4 consecutive qh bits starting at `start` into a packed int32
// with each byte = 0 or 1 (same primitive as V2.23 mmvq_q5_0). Used to
// feed the 5th-bit side through DP4A alongside the low-nibble side.
static __device__ __forceinline__ int expand_bits4(unsigned int qh, int start) {
    int out = 0;
    out |= ((qh >> (start + 0)) & 1u);
    out |= ((qh >> (start + 1)) & 1u) << 8;
    out |= ((qh >> (start + 2)) & 1u) << 16;
    out |= ((qh >> (start + 3)) & 1u) << 24;
    return out;
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_q5_0_wave64_q8_1(
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

    const flambeau_block_q5_0* x = (const flambeau_block_q5_0*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK5_0;
    const int blocks_per_col_y = nrows_y / QK8_1;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int b = 0; b < blocks_per_row_x; ++b) {
        // Decode Q5_0 weight block for this thread's row.
        float x_d = 0.0f;
        int v_nib[8] = {0};
        int v_bit[8] = {0};
        if (row_ok) {
            const flambeau_block_q5_0* bx =
                &x[(size_t) row * blocks_per_row_x + b];
            x_d = (float) bx->d;
            const int* ql_words = (const int*) bx->qs;  // 4 int32 = 16 bytes

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int qw = ql_words[j];
                v_nib[j]     = (qw >> 0) & 0x0F0F0F0F;  // elements 0..15 nibbles
                v_nib[j + 4] = (qw >> 4) & 0x0F0F0F0F;  // elements 16..31 nibbles
            }

            // Expand qh into 8 packed-byte int32s matching the nibble layout.
            // Elements 0..15 live in bits 0..15 of qh; 16..31 in bits 16..31.
            const unsigned int qh = *((const unsigned int*) bx->qh);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                v_bit[j]     = expand_bits4(qh, j * 4);         // bits 0..15
                v_bit[j + 4] = expand_bits4(qh, j * 4 + 16);    // bits 16..31
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const int col = tile_n + c;
            if (col >= ncols_y) break;

            const flambeau_block_q8_1* by =
                &y[(size_t) col * blocks_per_col_y + b];
            const float y_d = (float) by->d;
            const float y_s = (float) by->s;   // d_y · Σ q8
            const int* y_packed = (const int*) by->qs;

            int sumi_nib = 0;
            int sumi_bit = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sumi_nib = dp4a(v_nib[j], y_packed[j], sumi_nib);
                sumi_bit = dp4a(v_bit[j], y_packed[j], sumi_bit);
            }

            // Per-block: d · ((nibble + 16·bit) · y - 16·y_s)
            //          = d · (d_y · (sumi_nib + 16·sumi_bit) - 16·y_s)
            // Since s_y = d_y · Σ q8 already, the −16·d·y_s correction
            // is one FMA per block (no per-lane split like the dense
            // mmvq kernel's ×0.25 since here we aren't warp-reducing).
            sums[c] += x_d * (y_d * (float) (sumi_nib + 16 * sumi_bit) - 16.0f * y_s);
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
