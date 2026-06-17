#pragma once
// Templated IQ4_NL single-pass tiled MMQ (prefill GEMM). IQ4_NL is the
// 32-elem-block sibling of IQ4_XS — fp16 `d`, no sub-block scale, codebook
// reconstruction `y = d · KVALUES_IQ4NL[code]`. Each block is one 32-element
// tile (matches the Q8_1 grain). Backend-neutral: `dp4a` / `flambeau_iq4nl_lut`
// from the shared headers, int loads through `load_int_b2` (qs sits at
// block-offset 2). Warp-size agnostic via MMQ_Y (no LDS). `OutT` is the dest
// dtype.
#include "quant_traits.cuh"
#include "iq_grid.cuh"
#include "mmvq_store.cuh"

static __device__ __forceinline__ int flambeau_pack_iq4_lut(int nibbles) {
    const int b0 = (int) flambeau_iq4nl_lut(nibbles & 0xFF);
    const int b1 = (int) flambeau_iq4nl_lut((nibbles >>  8) & 0xFF);
    const int b2 = (int) flambeau_iq4nl_lut((nibbles >> 16) & 0xFF);
    const int b3 = (int) flambeau_iq4nl_lut((nibbles >> 24) & 0xFF);
    return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
}

template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_iq4_nl_wave64_row(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    OutT* __restrict__ dst,
    int ncols_x, int nrows_x, int ncols_y, int nrows_y, int nrows_dst)
{
    const int tile_m = blockIdx.x * MMQ_Y;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid    = threadIdx.x;

    const int row     = tile_m + tid;
    const bool row_ok = row < nrows_x;

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
            const flambeau_block_iq4_nl* bx = &x[(size_t) row * blocks_per_row_x + ib];
            d_w = (float) bx->d;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int word = load_int_b2((const int8_t*) bx->qs, j);
                v[j]     = flambeau_pack_iq4_lut(word        & 0x0F0F0F0F);
                v[j + 4] = flambeau_pack_iq4_lut((word >> 4) & 0x0F0F0F0F);
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const int col = tile_n + c;
            if (col >= ncols_y) break;
            const flambeau_block_q8_1* by = &y[(size_t) col * blocks_per_col_y + ib];
            const float d_y = (float) by->d;

            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) sumi = dp4a(v[j], load_int_b2(by->qs, j), sumi);
            sums[c] += d_w * d_y * (float) sumi;
        }
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int col = tile_n + c;
        if (col < ncols_y && row < nrows_dst) mmvq_store<OutT>(dst, col * nrows_dst + row, sums[c]);
    }
}
