#pragma once
// Templated Q4_0 single-pass tiled MMQ (prefill GEMM). Same independent-thread
// `MMQ_Y×TILE_N` tile as the Q8_0 wave64 kernel (each thread one weight row,
// loops its TILE_N cols, no LDS / no reduce → warp-size agnostic via MMQ_Y),
// with Q4_0's nibble decode (4 int32 → 8 lo/hi planes) and the `(q-8)` dp4a
// bias identity `x_d·(y_d·Σq·q8 − 8·y_s)` (`y_s = d_y·Σq8`, in the Q8_1 header).
// Backend-neutral: `dp4a` from the arch header, misaligned `qs`-as-int loads
// through `load_int_b2`. `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_store.cuh"

template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_q4_0_wave64_row(
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

    const flambeau_block_q4_0* x = (const flambeau_block_q4_0*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK4_0;
    const int blocks_per_col_y = nrows_y / QK8_1;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int b = 0; b < blocks_per_row_x; ++b) {
        float x_d = 0.0f;
        int v[8] = {0};
        if (row_ok) {
            const flambeau_block_q4_0* bx = &x[(size_t) row * blocks_per_row_x + b];
            x_d = (float) bx->d;
            #pragma unroll
            for (int j = 0; j < QK4_0 / 8; ++j) {
                const int qw = load_int_b2((const int8_t*) bx->qs, j);
                v[j]     = qw & 0x0F0F0F0F;
                v[j + 4] = (qw >> 4) & 0x0F0F0F0F;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const int col = tile_n + c;
            if (col >= ncols_y) break;
            const flambeau_block_q8_1* by = &y[(size_t) col * blocks_per_col_y + b];
            const float y_d = (float) by->d;
            const float y_s = (float) by->s;
            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) sumi = dp4a(v[j], load_int_b2(by->qs, j), sumi);
            sums[c] += x_d * (y_d * (float) sumi - 8.0f * y_s);
        }
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int col = tile_n + c;
        if (col < ncols_y && row < nrows_dst) mmvq_store<OutT>(dst, col * nrows_dst + row, sums[c]);
    }
}
