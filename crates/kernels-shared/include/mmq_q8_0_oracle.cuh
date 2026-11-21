#pragma once
// Templated Q8_0 MMQ correctness oracle: `dst[batch, row] = Σ_k x[row,k]·
// y[batch,k]` for Q8_0 weights × Q8_1 activation, one output element per block
// over a `{n_rows, n_batches}` grid. The K reduction spreads 256 threads as
// `sub = tid/32` (0..7) × `lane_lo = tid&31` — covering one byte of a block per
// lane, striding `THREADS/32` blocks — then a block-wide reduce. Warp-size
// agnostic: `sub`/`lane_lo` are thread-index derived (`sub = tid/32` equals the
// wave64 `warp*2+lane_hi`), and `mmvq_reduce_store<THREADS>` handles the
// warp-then-inter-warp sum on both wave64 (4 warps) and wave32 (8). Slow (no
// X reuse across batches) — the correctness baseline the tiled prefill kernels
// are checked against. `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<int THREADS, class OutT = float>
static __device__ __forceinline__ void mmq_q8_0_oracle_row(
    const flambeau_block_q8_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_batches, int n_blocks_per_row)
{
    const int row   = blockIdx.x;
    const int batch = blockIdx.y;
    if (row >= n_rows || batch >= n_batches) return;

    const int tid     = threadIdx.x;
    const int sub     = tid >> 5;          // 0..THREADS/32-1
    const int lane_lo = tid & 31;

    const flambeau_block_q8_0* xrow   = x + (size_t) row * n_blocks_per_row;
    const flambeau_block_q8_1* ybatch = y + (size_t) batch * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = sub; b < n_blocks_per_row; b += THREADS / 32) {
        const int xi = (int) xrow[b].qs[lane_lo];
        const int yi = (int) ybatch[b].qs[lane_lo];
        acc += (float) (xi * yi) * (float) xrow[b].d * (float) ybatch[b].d;
    }

    mmvq_reduce_store<THREADS, OutT>(acc, dst, batch * n_rows + row);
}
