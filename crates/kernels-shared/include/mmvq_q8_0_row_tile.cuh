#pragma once
// Templated Q8_0 row-tiled batched MMVQ (decode GEMV, N>1 slots). Each block
// stages an 8-block Q8_1 activation strip (for all N columns) into LDS once per
// outer iteration and reuses it across `WARPS = THREADS/WARP_SIZE` output rows
// (one warp per row) — so the activation is fetched from HBM once per block
// instead of once per row, the row-tile amortization for the wide-`n_rows`
// (GDN alpha/beta) projections. Warp-size agnostic (rule 5): the per-warp
// block-coverage loop strides by `WARP_SIZE/INT32_PER_BLOCK` so all 8 staged
// blocks are covered on wave64 (one pass, 8 lanes/block) and wave32 (two
// passes); `WARPS` rows/block therefore differ per backend (4 on wave64, 8 on
// wave32 at THREADS=256), which the launcher accounts for. Backend-neutral:
// `dp4a` / `warp_reduce_sum` from the arch header, weight + activation int loads
// through `load_int_b2`. `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<int THREADS, int N, class OutT = float>
static __device__ __forceinline__ void mmvq_q8_0_row_tile_batched_row(
    const flambeau_block_q8_0* __restrict__ w,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_blocks_per_row)
{
    constexpr int WARPS          = THREADS / WARP_SIZE;       // output rows/block
    constexpr int OUTER_BLOCKS   = 8;
    constexpr int INT32_PER_BLOCK = 8;                        // Q8_0
    constexpr int BLOCKS_PER_PASS = WARP_SIZE / INT32_PER_BLOCK;

    const int tid   = threadIdx.x;
    const int warp  = tid / WARP_SIZE;
    const int lane  = tid & (WARP_SIZE - 1);
    const int lane8 = lane & (INT32_PER_BLOCK - 1);

    const int row     = blockIdx.x * WARPS + warp;
    const bool row_ok = row < n_rows;
    const flambeau_block_q8_0* w_row =
        row_ok ? (w + (size_t) row * n_blocks_per_row) : w;

    __shared__ int   s_y_qs[N][OUTER_BLOCKS][INT32_PER_BLOCK];
    __shared__ float s_y_d [N][OUTER_BLOCKS];

    float acc[N];
    #pragma unroll
    for (int c = 0; c < N; ++c) acc[c] = 0.0f;

    for (int b_outer = 0; b_outer < n_blocks_per_row; b_outer += OUTER_BLOCKS) {
        if (tid < N * OUTER_BLOCKS) {
            const int c  = tid / OUTER_BLOCKS;
            const int bi = tid % OUTER_BLOCKS;
            const int block_idx = b_outer + bi;
            if (block_idx < n_blocks_per_row) {
                const flambeau_block_q8_1* py = y + (size_t) c * n_blocks_per_row + block_idx;
                #pragma unroll
                for (int j = 0; j < INT32_PER_BLOCK; ++j) s_y_qs[c][bi][j] = load_int_b2(py->qs, j);
                s_y_d[c][bi] = (float) py->d;
            } else {
                #pragma unroll
                for (int j = 0; j < INT32_PER_BLOCK; ++j) s_y_qs[c][bi][j] = 0;
                s_y_d[c][bi] = 0.0f;
            }
        }
        __syncthreads();

        if (row_ok) {
            for (int bi = lane / INT32_PER_BLOCK; bi < OUTER_BLOCKS; bi += BLOCKS_PER_PASS) {
                const int block_idx = b_outer + bi;
                if (block_idx < n_blocks_per_row) {
                    const flambeau_block_q8_0* wbk = w_row + block_idx;
                    const int w_v    = load_int_b2(wbk->qs, lane8);
                    const float w_dx = (float) wbk->d;
                    #pragma unroll
                    for (int c = 0; c < N; ++c) {
                        const int sumi = dp4a(w_v, s_y_qs[c][bi][lane8], 0);
                        acc[c] += (float) sumi * w_dx * s_y_d[c][bi];
                    }
                }
            }
        }
        __syncthreads();
    }

    if (!row_ok) return;
    mmvq_reduce_store_rowtile_batched<N, OutT>(acc, dst, row, n_rows, lane);
}
