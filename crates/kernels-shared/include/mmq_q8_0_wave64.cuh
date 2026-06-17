#pragma once
// Templated Q8_0 single-pass tiled MMQ (prefill GEMM): `dst[col,row] =
// Σ_k x[row,k]·y[col,k]` for Q8_0 weights × Q8_1 activation. A `MMQ_Y`-thread
// block computes a `MMQ_Y` (rows) × `TILE_N` (cols) output tile — each thread
// owns one weight row and loops its TILE_N columns, accumulating over all K
// blocks with `dp4a`. No LDS, no cross-lane reduction: every thread is
// independent, so this is warp-size agnostic once the row stride uses `MMQ_Y`
// (the tile height / block thread count, a design constant) rather than the
// wave size. Backend-neutral: `dp4a` from the arch header, the misaligned
// `qs`-as-int loads through `load_int_b2`. `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_store.cuh"

template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_q8_0_wave64_row(
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

    const flambeau_block_q8_0* x = (const flambeau_block_q8_0*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK8_0;
    const int blocks_per_col_y = nrows_y / QK8_1;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int b = 0; b < blocks_per_row_x; ++b) {
        float x_d = 0.0f;
        int x_packed[QK8_0 / 4] = {0};
        if (row_ok) {
            const flambeau_block_q8_0* bx = &x[(size_t) row * blocks_per_row_x + b];
            x_d = (float) bx->d;
            #pragma unroll
            for (int j = 0; j < QK8_0 / 4; ++j) x_packed[j] = load_int_b2(bx->qs, j);
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const int col = tile_n + c;
            if (col >= ncols_y) break;
            const flambeau_block_q8_1* by = &y[(size_t) col * blocks_per_col_y + b];
            const float y_d = (float) by->d;
            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < QK8_0 / 4; ++j) sumi = dp4a(x_packed[j], load_int_b2(by->qs, j), sumi);
            sums[c] += x_d * y_d * (float) sumi;
        }
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int col = tile_n + c;
        if (col < ncols_y && row < nrows_dst) mmvq_store<OutT>(dst, col * nrows_dst + row, sums[c]);
    }
}
