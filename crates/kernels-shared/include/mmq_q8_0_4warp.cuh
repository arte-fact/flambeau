#pragma once
// Templated Q8_0 4-warp LDS-tiled MMQ (prefill GEMM). MMQ_Y×MMQ_X output tile
// staged through LDS: each K-iter loads MMQ_Y weight rows + MMQ_X activation
// cols into shared `x_qs`/`y_qs` (+ their scales), then every thread computes
// one `dst[batch,row]` via 8 chained dp4a over the staged ints. The hand kernel
// mapped threads to tile slots via the wave64 layout (`warp*8 + lane/8`,
// `if(warp==0)`); the output tile is exactly MMQ_Y*MMQ_X = block threads, so a
// tid-based mapping (`tid/MMQ_X`, `tid%MMQ_X`) is warp-size agnostic. Backend-
// neutral: `dp4a` from the shared headers, staged int loads through
// `load_int_b2` (Q8_0/Q8_1 qs sit at non-mul-4 block offsets). `OutT` is the
// destination dtype. Requires MMQ_X == 8 (8 int32 per QK8_0 weight block).
#include "quant_traits.cuh"
#include "mmvq_store.cuh"

template<int MMQ_Y, int MMQ_X, class OutT = float>
static __device__ __forceinline__ void mmq_q8_0_4warp_row(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    OutT* __restrict__ dst,
    int n_rows, int n_batches, int n_blocks_per_row)
{
    const flambeau_block_q8_0* x = (const flambeau_block_q8_0*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int row_base   = blockIdx.x * MMQ_Y;
    const int batch_base = blockIdx.y * MMQ_X;
    const int tid = threadIdx.x;

    const int row_in_tile = tid / MMQ_X;   // 0..MMQ_Y-1
    const int col_in_tile = tid % MMQ_X;   // 0..MMQ_X-1
    const int row   = row_base   + row_in_tile;
    const int batch = batch_base + col_in_tile;

    __shared__ int   x_qs[MMQ_Y * 8];
    __shared__ float x_df[MMQ_Y];
    __shared__ int   y_qs[MMQ_X * 8];
    __shared__ float y_df[MMQ_X];

    float acc = 0.0f;
    for (int kb = 0; kb < n_blocks_per_row; ++kb) {
        // X tile: MMQ_Y rows × 8 int32, loaded 1 int/thread by the first MMQ_Y*8.
        {
            const int row_tile = tid / 8;
            const int qs_idx   = tid & 7;
            if (row_tile < MMQ_Y) {
                const int row_abs = row_base + row_tile;
                if (row_abs < n_rows) {
                    const flambeau_block_q8_0* bk = x + (size_t) row_abs * n_blocks_per_row + kb;
                    x_qs[row_tile * 8 + qs_idx] = load_int_b2((const int8_t*) bk->qs, qs_idx);
                    if (qs_idx == 0) x_df[row_tile] = (float) bk->d;
                } else {
                    x_qs[row_tile * 8 + qs_idx] = 0;
                    if (qs_idx == 0) x_df[row_tile] = 0.0f;
                }
            }
        }
        // Y tile: MMQ_X cols × 8 int32, loaded by the first MMQ_X*8 threads.
        if (tid < MMQ_X * 8) {
            const int bcol   = tid / 8;
            const int qs_idx = tid & 7;
            const int batch_abs = batch_base + bcol;
            if (batch_abs < n_batches) {
                const flambeau_block_q8_1* by = y + (size_t) batch_abs * n_blocks_per_row + kb;
                y_qs[bcol * 8 + qs_idx] = load_int_b2(by->qs, qs_idx);
                if (qs_idx == 0) y_df[bcol] = (float) by->d;
            } else {
                y_qs[bcol * 8 + qs_idx] = 0;
                if (qs_idx == 0) y_df[bcol] = 0.0f;
            }
        }

        __syncthreads();

        if (row < n_rows && batch < n_batches) {
            int dot = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                dot = dp4a(x_qs[row_in_tile * 8 + i], y_qs[col_in_tile * 8 + i], dot);
            acc += (float) dot * (x_df[row_in_tile] * y_df[col_in_tile]);
        }

        __syncthreads();
    }

    if (row < n_rows && batch < n_batches)
        mmvq_store<OutT>(dst, (int) ((size_t) batch * n_rows + row), acc);
}
