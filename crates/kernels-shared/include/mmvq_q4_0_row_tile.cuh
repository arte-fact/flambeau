#pragma once
// Templated Q4_0 row-tiled batched MMVQ (decode GEMV, N>1 slots). Like the Q8_0
// row-tile, `WARPS = THREADS/WARP_SIZE` rows/block share an LDS-staged Q8_1
// activation strip across N columns; here each Q4_0 block is covered by 4 weight
// int32 lanes (nibble lo/hi planes against the staged activation int32s `lane4`
// and `lane4+4`) and carries the `(q-8)` dp4a bias `−8·d_x·s_y` (split ×0.25
// across the 4 lanes). Warp-size agnostic (rule 5): the block-coverage loop
// strides `WARP_SIZE/4` so all 16 staged blocks are covered on wave64 (one pass)
// and wave32 (two). Backend-neutral: `dp4a` / `warp_reduce_sum` from the arch
// header, all int loads through `load_int_b2`. `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<int THREADS, int N, class OutT = float>
static __device__ __forceinline__ void mmvq_q4_0_row_tile_batched_row(
    const flambeau_block_q4_0* __restrict__ w,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_blocks_per_row)
{
    constexpr int WARPS           = THREADS / WARP_SIZE;     // output rows/block
    constexpr int OUTER_BLOCKS    = 16;
    constexpr int Y_INT32         = 8;                       // q8_1 block int32s
    constexpr int COVERAGE_LANES  = 4;                       // Q4_0 weight int32s/block
    constexpr int BLOCKS_PER_PASS = WARP_SIZE / COVERAGE_LANES;

    const int tid   = threadIdx.x;
    const int warp  = tid / WARP_SIZE;
    const int lane  = tid & (WARP_SIZE - 1);
    const int lane4 = lane & (COVERAGE_LANES - 1);

    const int row     = blockIdx.x * WARPS + warp;
    const bool row_ok = row < n_rows;
    const flambeau_block_q4_0* w_row =
        row_ok ? (w + (size_t) row * n_blocks_per_row) : w;

    __shared__ int   s_y_qs[N][OUTER_BLOCKS][Y_INT32];
    __shared__ float s_y_d [N][OUTER_BLOCKS];
    __shared__ float s_y_s [N][OUTER_BLOCKS];

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
                for (int j = 0; j < Y_INT32; ++j) s_y_qs[c][bi][j] = load_int_b2(py->qs, j);
                s_y_d[c][bi] = (float) py->d;
                s_y_s[c][bi] = (float) py->s;
            } else {
                #pragma unroll
                for (int j = 0; j < Y_INT32; ++j) s_y_qs[c][bi][j] = 0;
                s_y_d[c][bi] = 0.0f;
                s_y_s[c][bi] = 0.0f;
            }
        }
        __syncthreads();

        if (row_ok) {
            for (int bi = lane / COVERAGE_LANES; bi < OUTER_BLOCKS; bi += BLOCKS_PER_PASS) {
                const int block_idx = b_outer + bi;
                if (block_idx < n_blocks_per_row) {
                    const flambeau_block_q4_0* wbk = w_row + block_idx;
                    const int w_v    = load_int_b2((const int8_t*) wbk->qs, lane4);
                    const int w_vi_lo = w_v & 0x0F0F0F0F;
                    const int w_vi_hi = (w_v >> 4) & 0x0F0F0F0F;
                    const float w_dx = (float) wbk->d;

                    #pragma unroll
                    for (int c = 0; c < N; ++c) {
                        const int yu_lo = s_y_qs[c][bi][lane4];
                        const int yu_hi = s_y_qs[c][bi][lane4 + 4];
                        int sumi = dp4a(w_vi_lo, yu_lo, 0);
                        sumi     = dp4a(w_vi_hi, yu_hi, sumi);
                        acc[c] += sumi * (w_dx * s_y_d[c][bi]) - 8.0f * w_dx * s_y_s[c][bi] * 0.25f;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (!row_ok) return;
    mmvq_reduce_store_rowtile_batched<N, OutT>(acc, dst, row, n_rows, lane);
}
