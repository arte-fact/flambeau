// mmq_q3_K_wave64 — wave64 MMQ for Q3_K × Q8_1 activation.
// Same wave64 / MMQ_Y=64 / TILE_N=8 / DP4A structure as Q5_K / Q6_K wave64.
//
// Q3_K layout (110 B, 256 elements):
//   hmask[32] — 1 bit per element (high bit of 3-bit quant)
//   qs[64]    — 2 low bits per element, 4 elements / byte
//   scales[12]— packed 6-bit signed scales × 16
//   d         — fp16 super-block scale
//
// Per-element reconstruction:
//   raw_2bit  = (qs[blk128*32 + qi] >> shift) & 3
//   hmask_bit = (hmask[qi] >> hmask_bit_pos) & 1
//   q (signed in -4..3) = raw_2bit + 4*hmask_bit - 4
//   y = d * (scales[is] - 32) * q
//
// Bias-correction identity (Q6_K-style):
//   (raw_4bit - 4) · y = raw_4bit · y - 4 · Σy
// where raw_4bit = raw_2bit | (hmask_bit << 2) ∈ [0, 7].
//
// Q3_K block stride is 110 B (not multiple of 4), so hmask[], qs[] and
// scales[] alternate 4-byte / 2-byte alignment across consecutive blocks.
// All multi-byte reads go through `load_u32_unaligned` to stay correct on
// the 2-byte-aligned half.

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

static __device__ __forceinline__ uint32_t q3k_mmq_load_u32(const uint8_t* p) {
    return (uint32_t) p[0]
         | ((uint32_t) p[1] << 8)
         | ((uint32_t) p[2] << 16)
         | ((uint32_t) p[3] << 24);
}

static __device__ __forceinline__ void q3k_mmq_unpack_scales(
    const uint8_t* __restrict__ scales,
    int8_t out[16]
) {
    const uint32_t k1 = 0x0303'0303u;
    const uint32_t k2 = 0x0f0f'0f0fu;
    uint32_t aux[4];
    aux[0] = q3k_mmq_load_u32(scales);
    aux[1] = q3k_mmq_load_u32(scales + 4);
    const uint32_t tmp = q3k_mmq_load_u32(scales + 8);
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    aux[0] = (aux[0] & k2) | ((tmp & k1) << 4);
    aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
    const uint8_t* bytes = (const uint8_t*) aux;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i] = (int8_t) bytes[i];
    }
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_q3_K_wave64_q8_1(
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

    const flambeau_block_q3_K* x = (const flambeau_block_q3_K*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;  // 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f;
        int8_t sc_buf[16] = {0};
        const flambeau_block_q3_K* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d = (float) bx->d;
            q3k_mmq_unpack_scales(bx->scales, sc_buf);
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int blk128        = sub >> 2;
            const int shift_iter    = sub & 3;
            const int shift         = 2 * shift_iter;
            const int hmask_bit_pos = shift_iter + 4 * blk128;

            int v[8] = {0};
            if (row_ok) {
                const uint8_t* qs_base = bx->qs + blk128 * 32;
                const uint8_t* hm_base = bx->hmask;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t ql_word = q3k_mmq_load_u32(qs_base + j * 4);
                    const uint32_t qh_word = q3k_mmq_load_u32(hm_base + j * 4);
                    const int raw2 = (int) ((ql_word >> shift) & 0x03030303u);
                    const int hi   = (int) (((qh_word >> hmask_bit_pos) << 2) & 0x04040404u);
                    v[j] = raw2 | hi;
                }
            }

            const int sc_a = (int) sc_buf[2 * sub + 0] - 32;
            const int sc_b = (int) sc_buf[2 * sub + 1] - 32;

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

                const int corrected_a = sumi_a - 4 * sumi_y_a;
                const int corrected_b = sumi_b - 4 * sumi_y_b;

                sums[c] += super_d * d8 *
                    ((float)(sc_a * corrected_a + sc_b * corrected_b));
            }
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
