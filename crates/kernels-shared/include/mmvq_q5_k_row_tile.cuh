#pragma once
// Templated Q5_K row-tiled batched MMVQ (decode GEMV, N>1 slots). `ROWS =
// THREADS/32` output rows/block, each owned by a 32-lane group, share one
// LDS-staged Q8_1 super-block strip (8 sub-blocks) across N columns — the
// activation is fetched from HBM once per block instead of once per (row, sub,
// col). Warp-size agnostic (rule 5): the row group is `threadIdx.x/32` and the
// intra-row lane `threadIdx.x&31`, so 256 threads give 8 rows/block on both
// wave64 (sub-warp groups) and wave32 (full-warp groups) — no wave-size term.
// Backend-neutral: `flambeau_q4k_scale_min` + the r2 N-way reduce from the
// shared headers; activation int loads through `load_int_b2`. `OutT` is the
// destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<int THREADS, int N, class OutT = float>
static __device__ __forceinline__ void mmvq_q5_k_row_tile_batched_row(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_sb_per_row)
{
    constexpr int ROW_GROUP = 32;
    constexpr int ROWS      = THREADS / ROW_GROUP;
    constexpr int SUBS      = 8;

    const int tid       = threadIdx.x;
    const int row_group = tid / ROW_GROUP;
    const int lane_lo   = tid & (ROW_GROUP - 1);

    const int row     = blockIdx.x * ROWS + row_group;
    const bool row_ok = row < n_rows;
    const flambeau_block_q5_K* xrow =
        row_ok ? (x + (size_t) row * n_sb_per_row) : x;

    __shared__ int   s_y_qs[N][SUBS][8];
    __shared__ float s_y_d [N][SUBS];

    float acc[N];
    #pragma unroll
    for (int c = 0; c < N; ++c) acc[c] = 0.0f;

    const int act_blocks_per_row = n_sb_per_row * 8;

    for (int b = 0; b < n_sb_per_row; ++b) {
        if (tid < N * SUBS) {
            const int c   = tid >> 3;
            const int sub = tid & 7;
            const flambeau_block_q8_1* py = y + (size_t) c * act_blocks_per_row + b * 8 + sub;
            #pragma unroll
            for (int j = 0; j < 8; ++j) s_y_qs[c][sub][j] = load_int_b2(py->qs, j);
            s_y_d[c][sub] = (float) py->d;
        }
        __syncthreads();

        if (row_ok) {
            const flambeau_block_q5_K* bk = xrow + b;
            const float d_w       = (float) bk->d;
            const float dmin_w    = (float) bk->dmin;
            const uint8_t qh_byte = bk->qh[lane_lo];

            #pragma unroll
            for (int s = 0; s < SUBS; ++s) {
                uint8_t sc = 0, m_v = 0;
                flambeau_q4k_scale_min(s, bk->scales, &sc, &m_v);

                const int byte_v = (int) bk->qs[(s >> 1) * 32 + lane_lo];
                const int raw_q4 = (s & 1) ? (byte_v >> 4) : (byte_v & 0x0F);
                const int raw_q  = raw_q4 + ((qh_byte & (1 << s)) ? 16 : 0);
                const float x_val = d_w * (float) sc * (float) raw_q - dmin_w * (float) m_v;

                #pragma unroll
                for (int c = 0; c < N; ++c) {
                    const int qi = (int) ((const int8_t*) s_y_qs[c][s])[lane_lo];
                    acc[c] += x_val * s_y_d[c][s] * (float) qi;
                }
            }
        }
        __syncthreads();
    }

    if (!row_ok) return;
    mmvq_reduce_store_2row_batched<N, OutT>(acc, dst, row, n_rows, lane_lo);
}
