// mmq_iq2_xxs_wave64 — wave64 MMQ for IQ2_XXS × Q8_1 activation.
// Per ib32 = 32 elements = 1 Q8_1 block. Pair of u32 words read from
// `qs[ib32*8 .. ib32*8+8]`:
//   aux0 → 4 × 8-bit codebook indices into IQ2XXS_GRID (256 × u64;
//          each entry = 8 unsigned-i8 magnitudes packed in u64).
//   aux1 → 4 × 7-bit sign-LUT indices (bits 0..27) + 4-bit scale (bits 28..31).
// Each codebook lookup yields a full 8-element group; per ib32 there are
// 4 such groups = 32 elements. Sign byte = KSIGNS_IQ2XS[(aux1 >> (7*l)) & 0x7F].
// db = (0.5 + (aux1 >> 28)) * 0.25; partial = super_d * db * dp4a(weights, act).
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
void flambeau_mmq_iq2_xxs_wave64_q8_1(
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

    const flambeau_block_iq2_xxs* x = (const flambeau_block_iq2_xxs*) vx;
    const flambeau_block_q8_1*    y = (const flambeau_block_q8_1*)    vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f;
        const flambeau_block_iq2_xxs* bx = nullptr;
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
                const uint8_t* sp = bx->qs + 8 * sub;
                const uint32_t aux0 =
                      (uint32_t) sp[0] | ((uint32_t) sp[1] << 8)
                    | ((uint32_t) sp[2] << 16) | ((uint32_t) sp[3] << 24);
                const uint32_t aux1 =
                      (uint32_t) sp[4] | ((uint32_t) sp[5] << 8)
                    | ((uint32_t) sp[6] << 16) | ((uint32_t) sp[7] << 24);
                scale_factor = (0.5f + (float)(aux1 >> 28)) * 0.25f;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int idx = (int)((aux0 >> (8 * l)) & 0xFF);
                    const uint64_t g_u64 = IQ2XXS_GRID[idx];
                    const int g_lo = (int)(g_u64 & 0xFFFFFFFF);
                    const int g_hi = (int)((g_u64 >> 32) & 0xFFFFFFFF);
                    const uint8_t sign_byte = KSIGNS_IQ2XS[(aux1 >> (7 * l)) & 0x7F];
                    v[2 * l + 0] = apply_signs_packed(g_lo, sign_byte & 0x0F);
                    v[2 * l + 1] = apply_signs_packed(g_hi, (sign_byte >> 4) & 0x0F);
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
