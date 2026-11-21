#pragma once
// Templated Q5_K batched MMVQ (decode GEMV, N>1 slots), r2 multi-row: 64 threads
// = 2 output rows per block (a 32-lane group each), N activation columns. Each
// Q5_K super-block element is decoded ONCE (`x = d·sc·q − dmin·m`, q5 = 4-bit qs
// nibble | 5th bit from `qh`) and applied across all N columns (slot-major `y` /
// `dst`) — the weight read amortizes over the N slots. Backend-neutral:
// `flambeau_q4k_scale_min` + the r2 N-way reduce come from the shared headers;
// every load is a single byte or fp16 (no `load_int_b2`). `OutT` is the
// destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<int N, class OutT = float>
static __device__ __forceinline__ void mmvq_q5_k_r2_batched_row(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_superblocks_per_row)
{
    const int row_pair = blockIdx.x;
    const int lane     = threadIdx.x;
    const int row_hi   = lane >> 5;
    const int lane_lo  = lane & 31;

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const flambeau_block_q5_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc[N];
    #pragma unroll
    for (int c = 0; c < N; ++c) acc[c] = 0.0f;

    const int act_blocks_per_row = n_superblocks_per_row * 8;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q5_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;
        const uint8_t qh_byte = bk->qh[lane_lo];

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint8_t sc = 0, m = 0;
            flambeau_q4k_scale_min(s, bk->scales, &sc, &m);

            const int byte_v = (int) bk->qs[(s >> 1) * 32 + lane_lo];
            const int raw_q4 = (s & 1) ? (byte_v >> 4) : (byte_v & 0x0F);
            const int raw_q  = raw_q4 + ((qh_byte & (1 << s)) ? 16 : 0);
            const float x_val = d * (float) sc * (float) raw_q - dmin * (float) m;

            #pragma unroll
            for (int c = 0; c < N; ++c) {
                const flambeau_block_q8_1* ya =
                    y + (size_t) c * act_blocks_per_row + b * 8 + s;
                acc[c] += x_val * (float) ya->d * (float) (int) ya->qs[lane_lo];
            }
        }
    }

    mmvq_reduce_store_2row_batched<N, OutT>(acc, dst, row, n_rows, lane_lo);
}
