// mmq_iq2_s_wave64 — wave64 MMQ for IQ2_S × Q8_1 activation.
// Per sub-block ib32 (32 elements): 4 codebook indices in `qs[4*ib32 + l]`,
// each promoted to a 10-bit index via 2 high bits from `qh[ib32]`.
// Codebook = IQ2S_GRID (1024 × u64). Sign masks are stored in
// `qs[32 + 4*ib32 + l]` directly (per-byte, no LUT). Scale chain mirrors
// IQ2_XS: db = (0.5 + nibble) * 0.25, nibble from scales[ib32 >> 1].

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

static __device__ __forceinline__ int apply_signs_byte(int g_u32, uint8_t signs, int byte_off) {
    // `signs` is 8 sign bits; bits [byte_off*4 .. byte_off*4+4) cover this g_u32.
    int out = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int mag = (g_u32 >> (8 * j)) & 0xFF;
        const int bit = byte_off * 4 + j;
        const int neg = (signs >> bit) & 1;
        const int v   = neg ? -mag : mag;
        out |= (v & 0xFF) << (8 * j);
    }
    return out;
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_iq2_s_wave64_q8_1(
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

    const flambeau_block_iq2_s* x = (const flambeau_block_iq2_s*) vx;
    const flambeau_block_q8_1*  y = (const flambeau_block_q8_1*)  vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f;
        const flambeau_block_iq2_s* bx = nullptr;
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
                const uint8_t sc_byte = bx->scales[sub >> 1];
                const int sc_nibble = (sub & 1) ? (sc_byte >> 4) : (sc_byte & 0x0F);
                scale_factor = (0.5f + (float) sc_nibble) * 0.25f;
                const uint8_t qh_byte = bx->qh[sub];
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int idx = (int) bx->qs[4 * sub + l]
                                  | (((int) qh_byte << (8 - 2 * l)) & 0x300);
                    const uint64_t g_u64 = IQ2S_GRID[idx];
                    const int g_lo = (int)(g_u64 & 0xFFFFFFFF);
                    const int g_hi = (int)((g_u64 >> 32) & 0xFFFFFFFF);
                    const uint8_t signs = bx->qs[32 + 4 * sub + l];
                    v[2 * l + 0] = apply_signs_byte(g_lo, signs, 0);
                    v[2 * l + 1] = apply_signs_byte(g_hi, signs, 1);
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
