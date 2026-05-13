// mmq_iq3_s_wave64 — wave64 MMQ for IQ3_S × Q8_1 activation.
// Same tile shape and outer skeleton as `mmq_iq4_xs_wave64.cu` /
// `mmq_q4_K_wave64.cu`: 64 threads = one wave64, 64 output rows per block,
// each thread owns one row and TILE_N=8 columns. Inner loop is dp4a over
// 8 i32 weight words × 8 i32 activation words per sub-block.
//
// IQ3_S weight decode per sub-block ib32 (32 elements):
//   sc_byte  = scales[ib32 >> 1]
//   db       = d * (1.0 + 2.0 * ((ib32 & 1) ? sc_byte>>4 : sc_byte&0xF))
//   qh_byte  = qh[ib32]
//   For l ∈ [0,4):  (each l = 8 elements = 2 codebook lookups)
//     g1_idx   = qs[ib32*8 + 2*l + 0] | ((qh_byte << (8 - 2*l)) & 0x100)
//     g2_idx   = qs[ib32*8 + 2*l + 1] | ((qh_byte << (7 - 2*l)) & 0x100)
//     g1_u32   = IQ3S_GRID[g1_idx]                  // 4 unsigned-i8 magnitudes
//     g2_u32   = IQ3S_GRID[g2_idx]                  // 4 unsigned-i8 magnitudes
//     signs_b  = signs[ib32*4 + l]                   // bits 0..3 → g1, 4..7 → g2
//     v[2*l+0] = apply_signs(g1_u32, signs_b & 0x0F)
//     v[2*l+1] = apply_signs(g2_u32, signs_b >> 4)
//
// `apply_signs` conditionally negates each of 4 packed i8s based on the
// corresponding bit of the 4-bit sign mask. Each element ∈ [-62, 62],
// so the packed i32 stays a valid `dp4a` input.

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

// Apply 4 sign bits to 4 packed unsigned-i8 magnitudes, returning the
// signed-i8 packed result. Each bit set ↔ that lane is negated.
static __device__ __forceinline__ int apply_signs_packed(int g_u32, int sign4) {
    int out = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int mag = (g_u32 >> (8 * j)) & 0xFF;
        const int neg = (sign4 >> j) & 1;
        const int v   = neg ? -mag : mag;
        out |= (v & 0xFF) << (8 * j);
    }
    return out;
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_iq3_s_wave64_q8_1(
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

    const flambeau_block_iq3_s* x = (const flambeau_block_iq3_s*) vx;
    const flambeau_block_q8_1*  y = (const flambeau_block_q8_1*)  vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;  // 8 sub-blocks

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f;
        const flambeau_block_iq3_s* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d = (float) bx->d;
        }

        float sumf[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) sumf[c] = 0.0f;

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int v[8] = {0};
            float db_no_super = 0.0f;
            if (row_ok) {
                const uint8_t sc_byte   = bx->scales[sub >> 1];
                const int     sc_nibble = (sub & 1) ? (sc_byte >> 4) : (sc_byte & 0x0F);
                db_no_super = 1.0f + 2.0f * (float) sc_nibble;

                const uint8_t qh_byte = bx->qh[sub];
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int g1_idx = (int) bx->qs[sub * 8 + 2 * l + 0]
                                     | (((int) qh_byte << (8 - 2 * l)) & 0x100);
                    const int g2_idx = (int) bx->qs[sub * 8 + 2 * l + 1]
                                     | (((int) qh_byte << (7 - 2 * l)) & 0x100);
                    const int g1_u32 = (int) IQ3S_GRID[g1_idx];
                    const int g2_u32 = (int) IQ3S_GRID[g2_idx];
                    const int signs  = (int) bx->signs[sub * 4 + l];
                    v[2 * l + 0] = apply_signs_packed(g1_u32,  signs        & 0x0F);
                    v[2 * l + 1] = apply_signs_packed(g2_u32, (signs >> 4)  & 0x0F);
                }
            }

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int col = tile_n + c;
                if (col >= ncols_y) break;

                const flambeau_block_q8_1* by =
                    &y[(size_t) col * blocks_per_col_y + ib * q8_per_super + sub];
                const float d8 = (float) by->d;

                const int* y_packed = (const int*) by->qs;

                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sumi = dp4a(v[j], y_packed[j], sumi);
                }
                sumf[c] += d8 * ((float) sumi) * db_no_super;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            sums[c] += super_d * sumf[c];
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
