// Q5_1 block layout (24 bytes, 32 elements): fp16 d + fp16 m + uint8 qh[4]
// + uint8 qs[16]. Reconstruction: x_i = d · q5_i + m where q5_i = (bit_i << 4)
// | nibble_i, unsigned in [0, 31]. Dot product with Q8_1:
//   dot = d · d_y · (Σ nibble·q8 + 16·Σ bit·q8) + m · y_s.

#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef QK5_1
#define QK5_1 32
#endif
#ifndef QK8_1
#define QK8_1 32
#endif

#define MMQ_Y  64
#define TILE_N 8

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ int expand_bits4(unsigned int qh, int start) {
    int out = 0;
    out |= ((qh >> (start + 0)) & 1u);
    out |= ((qh >> (start + 1)) & 1u) << 8;
    out |= ((qh >> (start + 2)) & 1u) << 16;
    out |= ((qh >> (start + 3)) & 1u) << 24;
    return out;
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_q5_1_wave64_q8_1(
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

    const flambeau_block_q5_1* x = (const flambeau_block_q5_1*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK5_1;
    const int blocks_per_col_y = nrows_y / QK8_1;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int b = 0; b < blocks_per_row_x; ++b) {
        float x_d = 0.0f, x_m = 0.0f;
        int v_nib[8] = {0};
        int v_bit[8] = {0};
        if (row_ok) {
            const flambeau_block_q5_1* bx =
                &x[(size_t) row * blocks_per_row_x + b];
            x_d = (float) bx->d;
            x_m = (float) bx->m;
            const int* ql_words = (const int*) bx->qs;

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int qw = ql_words[j];
                v_nib[j]     = (qw >> 0) & 0x0F0F0F0F;
                v_nib[j + 4] = (qw >> 4) & 0x0F0F0F0F;
            }

            const unsigned int qh = *((const unsigned int*) bx->qh);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                v_bit[j]     = expand_bits4(qh, j * 4);
                v_bit[j + 4] = expand_bits4(qh, j * 4 + 16);
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const int col = tile_n + c;
            if (col >= ncols_y) break;

            const flambeau_block_q8_1* by =
                &y[(size_t) col * blocks_per_col_y + b];
            const float y_d = (float) by->d;
            const float y_s = (float) by->s;
            const int* y_packed = (const int*) by->qs;

            int sumi_nib = 0;
            int sumi_bit = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sumi_nib = dp4a(v_nib[j], y_packed[j], sumi_nib);
                sumi_bit = dp4a(v_bit[j], y_packed[j], sumi_bit);
            }

            sums[c] += x_d * y_d * (float) (sumi_nib + 16 * sumi_bit) + x_m * y_s;
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
