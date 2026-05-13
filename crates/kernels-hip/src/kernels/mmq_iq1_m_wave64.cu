// mmq_iq1_m_wave64 — wave64 MMQ for IQ1_M × Q8_1 activation.
// Same arithmetic as IQ1_S (signed-i8 grid + ±delta offset, bias-corrected
// via Q8_1 `s` field), but:
//   - No per-block `d` field — reassembled from top nibble of each of the
//     4 u16 `scales` words (see `iq1m_reassemble_d`).
//   - Per `l` within a sub-block carries its own dl and delta:
//       qh_byte = (l < 2) ? qh[2*sub] : qh[2*sub + 1]
//       shift   = 8 - 4*(l & 1)
//       idx     = qs[4*sub + l] | ((qh_byte << shift) & 0x700)
//       delta_bit = (l & 1) == 0 ? 0x08 : 0x80
//       dl_pair = (l < 2) ? dl1 : dl2  with dl1/dl2 from scale-word nibbles
// dl1, dl2 share the qh byte and are computed once per sub-block.
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

static __device__ __forceinline__ float iq1m_reassemble_d(const uint8_t* __restrict__ scales) {
    const int sc0 = (int) scales[0] | ((int) scales[1] << 8);
    const int sc1 = (int) scales[2] | ((int) scales[3] << 8);
    const int sc2 = (int) scales[4] | ((int) scales[5] << 8);
    const int sc3 = (int) scales[6] | ((int) scales[7] << 8);
    const int d_bits = (sc0 >> 12)
                     | ((sc1 >> 8) & 0x00F0)
                     | ((sc2 >> 4) & 0x0F00)
                     | (sc3 & 0xF000);
    fb_fp16_t d_fp16 = *reinterpret_cast<const fb_fp16_t*>(&d_bits);
    return (float) d_fp16;
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_iq1_m_wave64_q8_1(
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

    const flambeau_block_iq1_m* x = (const flambeau_block_iq1_m*) vx;
    const flambeau_block_q8_1*  y = (const flambeau_block_q8_1*)  vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float d = 0.0f;
        const flambeau_block_iq1_m* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            d = iq1m_reassemble_d(bx->scales);
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            // For each of the 4 inner l's, capture (dl, delta) and packed weights.
            int v[8] = {0};
            float dl_l[4] = {0};
            float delta_l[4] = {0};
            if (row_ok) {
                const int sc_word = (int) bx->scales[2 * (sub >> 1)]
                                  | ((int) bx->scales[2 * (sub >> 1) + 1] << 8);
                const int shift1 = 6 * (sub & 1);
                const int shift2 = shift1 + 3;
                const float dl1 = d * (2.0f * (float)((sc_word >> shift1) & 7) + 1.0f);
                const float dl2 = d * (2.0f * (float)((sc_word >> shift2) & 7) + 1.0f);
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int qh_pick = (l < 2) ? (2 * sub) : (2 * sub + 1);
                    const uint8_t qh_byte = bx->qh[qh_pick];
                    const int shift_idx  = 8 - 4 * (l & 1);
                    const int idx = (int) bx->qs[4 * sub + l]
                                  | (((int) qh_byte << shift_idx) & 0x700);
                    const int delta_bit = (l & 1) == 0 ? 0x08 : 0x80;
                    dl_l[l]    = (l < 2) ? dl1 : dl2;
                    delta_l[l] = (qh_byte & delta_bit) ? -IQ1_DELTA : IQ1_DELTA;
                    const uint64_t g_u64 = IQ1S_GRID[idx];
                    v[2 * l + 0] = (int)(g_u64 & 0xFFFFFFFF);
                    v[2 * l + 1] = (int)((g_u64 >> 32) & 0xFFFFFFFF);
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
                // Each l owns 2 packed-i32 weight words and 2 packed-i32
                // activation words; dl/delta vary per l. The bias-correction
                // term `dl_l * delta_l * d_y * sum_qi_l` needs the per-l
                // sum-of-qi (8 elements), computed via dp4a-with-ones.
                float partial = 0.0f;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    int sumi = 0;
                    int sumi_y = 0;
                    sumi   = dp4a(v[2 * l + 0], y_packed[2 * l + 0], sumi);
                    sumi   = dp4a(v[2 * l + 1], y_packed[2 * l + 1], sumi);
                    sumi_y = dp4a(0x01010101, y_packed[2 * l + 0], sumi_y);
                    sumi_y = dp4a(0x01010101, y_packed[2 * l + 1], sumi_y);
                    partial += dl_l[l] * d_y * ((float) sumi + delta_l[l] * (float) sumi_y);
                }
                sums[c] += partial;
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
