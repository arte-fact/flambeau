// mmq_iq3_xxs_wave64 — wave64 MMQ for IQ3_XXS × Q8_1 activation.
// Same outer shape as mmq_iq3_s_wave64.cu; per-element decode differs:
//   - aux32 packed at qs[64 + 4*ib32]: 4-bit scale at bits 28..31 +
//     4 × 7-bit sign-LUT indices in bits 0..27.
//   - 8 codebook indices per ib32 at qs[ib32*8..(ib32+1)*8], looking up
//     IQ3XXS_GRID (256 × u32) → 4 unsigned-i8 magnitudes per entry.
//   - Sign byte = KSIGNS_IQ2XS[(aux32 >> (7*l)) & 0x7F] (shared with IQ2XS).
// Phase 4 Slice A.

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
void flambeau_mmq_iq3_xxs_wave64_q8_1(
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

    const flambeau_block_iq3_xxs* x = (const flambeau_block_iq3_xxs*) vx;
    const flambeau_block_q8_1*    y = (const flambeau_block_q8_1*)    vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;  // 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f;
        const flambeau_block_iq3_xxs* bx = nullptr;
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
            float scale_factor = 0.0f;
            if (row_ok) {
                const uint8_t* scs = bx->qs + QK_K / 4;
                const uint8_t* sp = scs + 4 * sub;
                const uint32_t aux32 =
                      (uint32_t) sp[0] | ((uint32_t) sp[1] << 8)
                    | ((uint32_t) sp[2] << 16) | ((uint32_t) sp[3] << 24);
                // IQ3_XXS uses *0.5 (not *0.25 like IQ2_XXS) — mirrors the
                // host `dequant_iq3_xxs` and the IQ3_XXS MMVQ.
                scale_factor = (0.5f + (float)(aux32 >> 28)) * 0.5f;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int g1_idx = (int) bx->qs[sub * 8 + 2 * l + 0];
                    const int g2_idx = (int) bx->qs[sub * 8 + 2 * l + 1];
                    const int g1_u32 = (int) IQ3XXS_GRID[g1_idx];
                    const int g2_u32 = (int) IQ3XXS_GRID[g2_idx];
                    const uint8_t sign_byte = KSIGNS_IQ2XS[(aux32 >> (7 * l)) & 0x7F];
                    v[2 * l + 0] = apply_signs_packed(g1_u32, sign_byte & 0x0F);
                    v[2 * l + 1] = apply_signs_packed(g2_u32, (sign_byte >> 4) & 0x0F);
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
                for (int j = 0; j < 8; ++j) sumi = dp4a(v[j], y_packed[j], sumi);
                sumf[c] += d_y * (float) sumi * scale_factor;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) sums[c] += super_d * sumf[c];
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
