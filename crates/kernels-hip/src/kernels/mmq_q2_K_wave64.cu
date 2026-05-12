// mmq_q2_K_wave64 — wave64 MMQ for Q2_K × Q8_1 activation.
// Same wave64 / MMQ_Y=64 / TILE_N=8 / DP4A structure as Q5_K / Q6_K wave64.
//
// Q2_K layout (84 B, 256 elements):
//   scales[16] — packed 4-bit (scale, min): lo nibble = scale, hi = min
//   qs[64]     — 2 bits per element, 4 elements / byte
//   d          — fp16 super-block scale
//   dmin       — fp16 super-block min
//
// Per-element reconstruction:
//   raw_2bit  = (qs[chunk_idx*32 + qi] >> shift) & 3
//   y = d * (scales[is] & 0xF) * raw_2bit - dmin * (scales[is] >> 4)
//
// Per Q8_1 sub-block of 32 elements:
//   chunk_idx  = sub >> 2
//   shift_iter = sub & 3        (shift = 2*shift_iter)
//   Two Q2_K sub-blocks (16 elements each): is_a = 2*sub, is_b = 2*sub+1.
//
// Affine reduction (same shape as Q4_K wave64):
//   sums[c] += d_x * Σ_sub d_y[sub] * (sc_a * dot_a + sc_b * dot_b)
//           - dmin_x * Σ_sub d_y[sub] * (m_a * Σy_a + m_b * Σy_b)
//
// Q2_K block size 84 B is a multiple of 4, so int32 reads on scales[] and
// qs[] are naturally aligned across consecutive blocks (unlike Q3_K).

#include "block_quant.cuh"
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

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_q2_K_wave64_q8_1(
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

    const flambeau_block_q2_K* x = (const flambeau_block_q2_K*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;  // 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f, super_dmin = 0.0f;
        uint8_t sc_buf[16] = {0};
        const flambeau_block_q2_K* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d    = (float) bx->d;
            super_dmin = (float) bx->dmin;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                sc_buf[j] = bx->scales[j];
            }
        }

        float sumf_d[TILE_N];
        float sumf_m[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) { sumf_d[c] = 0.0f; sumf_m[c] = 0.0f; }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int chunk_idx  = sub >> 2;
            const int shift_iter = sub & 3;
            const int shift      = 2 * shift_iter;

            int v[8] = {0};
            if (row_ok) {
                const int* qs_words = (const int*) (bx->qs + chunk_idx * 32);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = qs_words[j];
                    v[j] = (ql_word >> shift) & 0x03030303;
                }
            }

            const int sc_a = (int) (sc_buf[2 * sub + 0] & 0xF);
            const int sc_b = (int) (sc_buf[2 * sub + 1] & 0xF);
            const int m_a  = (int) (sc_buf[2 * sub + 0] >> 4);
            const int m_b  = (int) (sc_buf[2 * sub + 1] >> 4);

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int col = tile_n + c;
                if (col >= ncols_y) break;

                const flambeau_block_q8_1* by =
                    &y[(size_t) col * blocks_per_col_y + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const int* y_packed = (const int*) by->qs;

                int sumi_a = 0, sumi_b = 0;
                int sumi_y_a = 0, sumi_y_b = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sumi_a = dp4a(v[j], y_packed[j], sumi_a);
                    sumi_y_a = dp4a(0x01010101, y_packed[j], sumi_y_a);
                }
                #pragma unroll
                for (int j = 4; j < 8; ++j) {
                    sumi_b = dp4a(v[j], y_packed[j], sumi_b);
                    sumi_y_b = dp4a(0x01010101, y_packed[j], sumi_y_b);
                }

                sumf_d[c] += d8 * (float) (sc_a * sumi_a + sc_b * sumi_b);
                sumf_m[c] += d8 * (float) (m_a  * sumi_y_a + m_b  * sumi_y_b);
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            sums[c] += super_d * sumf_d[c] - super_dmin * sumf_m[c];
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
