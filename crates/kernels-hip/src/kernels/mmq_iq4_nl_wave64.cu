// mmq_iq4_nl_wave64 — wave64 MMQ for IQ4_NL × Q8_1 activation.
// IQ4_NL is the 32-elem-block sibling of IQ4_XS — no sub-block scale, no
// super-block. Reconstruction: y = d * KVALUES_IQ4NL[code]. Treats each
// IQ4_NL block as a single 32-element tile (matches the Q8_1 block grain).
// Phase 4 Slice A.

#include "block_quant.cuh"
#include "iq_grid.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef QK8_1
#define QK8_1 32
#endif
#ifndef QK4_0
#define QK4_0 32
#endif

#define MMQ_Y 64
#define TILE_N 8

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ int pack_iq4_lut(int nibbles) {
    const int b0 = (int) flambeau_iq4nl_lut(nibbles & 0xFF);
    const int b1 = (int) flambeau_iq4nl_lut((nibbles >>  8) & 0xFF);
    const int b2 = (int) flambeau_iq4nl_lut((nibbles >> 16) & 0xFF);
    const int b3 = (int) flambeau_iq4nl_lut((nibbles >> 24) & 0xFF);
    return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_iq4_nl_wave64_q8_1(
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

    const flambeau_block_iq4_nl* x = (const flambeau_block_iq4_nl*) vx;
    const flambeau_block_q8_1*   y = (const flambeau_block_q8_1*)   vy;

    const int blocks_per_row_x = ncols_x / QK4_0;
    const int blocks_per_col_y = nrows_y / QK8_1;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        int v[8] = {0};
        float d_w = 0.0f;
        if (row_ok) {
            const flambeau_block_iq4_nl* bx =
                &x[(size_t) row * blocks_per_row_x + ib];
            d_w = (float) bx->d;
            const int* ql_ptr = (const int*) bx->qs;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int word = ql_ptr[j];
                v[j]     = pack_iq4_lut(word        & 0x0F0F0F0F);
                v[j + 4] = pack_iq4_lut((word >> 4) & 0x0F0F0F0F);
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const int col = tile_n + c;
            if (col >= ncols_y) break;

            const flambeau_block_q8_1* by =
                &y[(size_t) col * blocks_per_col_y + ib];
            const float d_y = (float) by->d;
            const int* y_packed = (const int*) by->qs;

            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) sumi = dp4a(v[j], y_packed[j], sumi);
            sums[c] += d_w * d_y * (float) sumi;
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
