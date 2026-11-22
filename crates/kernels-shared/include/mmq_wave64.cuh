#pragma once
// Templated single-pass tiled MMQ (prefill GEMM) skeleton for the simple
// 32-element-block quants (Q8_0/Q4_0/Q4_1/Q5_0/Q5_1). A `MMQ_Y`-thread block
// computes a `MMQ_Y` (rows) × `TILE_N` (cols) output tile: each thread owns one
// weight row, decodes its block ONCE per K-block (`quant_traits<Q>::mmq_decode`)
// and reuses it across the TILE_N columns (`mmq_partial` does the per-column
// 8-dp4a + scale/bias). No LDS, no cross-lane reduction → warp-size agnostic
// (the row stride uses `MMQ_Y`, the tile height / block thread count, not the
// wave size). `quant_traits<Q>` carries all per-dtype work; this skeleton is
// backend-neutral. `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_store.cuh"

template<class Q, int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_wave64_row(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    OutT* __restrict__ dst,
    int ncols_x, int nrows_x, int ncols_y, int nrows_y, int nrows_dst)
{
    using T = quant_traits<Q>;
    const int tile_m = blockIdx.x * MMQ_Y;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid    = threadIdx.x;

    const int row     = tile_m + tid;
    const bool row_ok = row < nrows_x;

    const typename T::block_t* x = (const typename T::block_t*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK8_1;   // 32-element blocks
    const int blocks_per_col_y = nrows_y / QK8_1;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int b = 0; b < blocks_per_row_x; ++b) {
        typename T::mmq_weight w = typename T::mmq_weight{};
        if (row_ok) w = T::mmq_decode(x[(size_t) row * blocks_per_row_x + b]);

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const int col = tile_n + c;
            if (col >= ncols_y) break;
            sums[c] += T::mmq_partial(w, y[(size_t) col * blocks_per_col_y + b]);
        }
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int col = tile_n + c;
        if (col < ncols_y && row < nrows_dst) mmvq_store<OutT>(dst, col * nrows_dst + row, sums[c]);
    }
}
