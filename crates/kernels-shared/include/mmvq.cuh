#pragma once
// Templated single-row MMVQ (decode GEMV): quantized weight rows × one Q8_1
// activation column -> F32. One thread block per output row; each thread owns
// one int32 (4 packed quants) of a block; `dp4a` inner product; a warp reduce
// then an inter-warp reduce through shared memory. `Q` selects the weight dtype
// via quant_traits; `THREADS` is the block size; `OutT` is the destination
// dtype (`float` for the scratch-then-cast path, `fb_fp16_t` for direct F16-dst
// with saturation). The arch primitives it calls (`dp4a`, `warp_reduce_sum`,
// `__shfl_xor`) come from the backend's arch_primitives header, included before
// this one. Warp-size agnostic: the block stride and warp count derive from
// WARP_SIZE.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<class Q, int THREADS, class OutT = float>
static __device__ __forceinline__ void mmvq_dp4a_row(
    const typename quant_traits<Q>::block_t* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_blocks_per_row)
{
    using T = quant_traits<Q>;
    constexpr int IPB  = T::INT32_PER_BLOCK;      // int32s per block
    constexpr int STEP = THREADS / IPB;           // blocks advanced per iter

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid = threadIdx.x;
    const int j   = tid & (IPB - 1);              // int32 index within a block
    const int b0  = tid / IPB;                    // 0..STEP-1

    const typename T::block_t* xrow = x + (size_t) row * n_blocks_per_row;
    float acc = 0.0f;
    for (int b = b0; b < n_blocks_per_row; b += STEP) {
        acc += T::partial(xrow[b], y[b], j);
    }
    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}

// VDR variant of mmvq_dp4a_row: each thread owns `VDR` consecutive int32 lanes
// of a block (`IPB/VDR` threads per block), chaining VDR dp4a into one int sum
// before the block-scale multiply — fewer float ops + a tighter dp4a register
// chain (better ILP). `quant_traits<Q>::partial_vdr<VDR>` carries the per-dtype
// work. VDR=1 reproduces `mmvq_dp4a_row` exactly.
template<class Q, int THREADS, int VDR, class OutT = float>
static __device__ __forceinline__ void mmvq_dp4a_vdr_row(
    const typename quant_traits<Q>::block_t* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_blocks_per_row)
{
    using T = quant_traits<Q>;
    constexpr int TPB  = T::INT32_PER_BLOCK / VDR;   // threads per quant block
    constexpr int STEP = THREADS / TPB;              // blocks advanced per iter

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid = threadIdx.x;
    const int grp = tid & (TPB - 1);                 // int32-pair slot in a block
    const int b0  = tid / TPB;

    const typename T::block_t* xrow = x + (size_t) row * n_blocks_per_row;
    float acc = 0.0f;
    for (int b = b0; b < n_blocks_per_row; b += STEP) {
        acc += T::template partial_vdr<VDR>(xrow[b], y[b], grp);
    }
    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}

// Multi-row config variant of mmvq_dp4a_row: one block computes `ROWS` output
// rows (a `THREADS/ROWS`-lane group per row), so the y (activation) reads are
// shared across ROWS weight rows and the grid shrinks by ROWS. r2 = 2 rows × 32
// lanes, r4 = 4 rows × 16 lanes on a 64-thread (one gfx906 wave) block. The
// per-row group reduces with a width-(THREADS/ROWS) butterfly through
// `mmvq_reduce_store_group` (no LDS). `LANES = THREADS/ROWS` must be a power of
// two and a multiple of `INT32_PER_BLOCK`.
template<class Q, int THREADS, int ROWS, class OutT = float>
static __device__ __forceinline__ void mmvq_dp4a_nrow_row(
    const typename quant_traits<Q>::block_t* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_blocks_per_row)
{
    using T = quant_traits<Q>;
    constexpr int IPB   = T::INT32_PER_BLOCK;
    constexpr int LANES = THREADS / ROWS;        // lanes per output row
    constexpr int STEP  = LANES / IPB;           // blocks advanced per iter

    const int tid     = threadIdx.x;
    const int row_grp = tid / LANES;             // 0..ROWS-1
    const int lane_lo = tid & (LANES - 1);       // intra-row lane
    const int j       = lane_lo & (IPB - 1);     // int32 index within a block
    const int b0      = lane_lo / IPB;

    const int row = blockIdx.x * ROWS + row_grp;
    if (row >= n_rows) return;

    const typename T::block_t* xrow = x + (size_t) row * n_blocks_per_row;
    float acc = 0.0f;
    for (int b = b0; b < n_blocks_per_row; b += STEP) {
        acc += T::partial(xrow[b], y[b], j);
    }
    mmvq_reduce_store_group<LANES, OutT>(acc, dst, row, lane_lo);
}

// Batched config variant of mmvq_dp4a_row: one weight row is dotted against `N`
// Q8_1 activation columns (slot-major `y[N][n_blocks_per_row]`), producing the
// slot-major output `dst[N][n_rows]`. The weight block load is invariant in the
// slot index, so the unrolled inner loop hoists it (LICM) — the weight row is
// read once and reused across all N slots, the batched-decode amortization (the
// mirror of the gate+up fusion, which shares the activation across 2 weights).
template<class Q, int THREADS, int N, class OutT = float>
static __device__ __forceinline__ void mmvq_dp4a_batched_row(
    const typename quant_traits<Q>::block_t* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_blocks_per_row)
{
    using T = quant_traits<Q>;
    constexpr int IPB  = T::INT32_PER_BLOCK;
    constexpr int STEP = THREADS / IPB;

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid = threadIdx.x;
    const int j   = tid & (IPB - 1);
    const int b0  = tid / IPB;

    const typename T::block_t* xrow = x + (size_t) row * n_blocks_per_row;
    float acc[N];
    #pragma unroll
    for (int s = 0; s < N; ++s) acc[s] = 0.0f;

    for (int b = b0; b < n_blocks_per_row; b += STEP) {
        #pragma unroll
        for (int s = 0; s < N; ++s) {
            acc[s] += T::partial(xrow[b], y[(size_t) s * n_blocks_per_row + b], j);
        }
    }
    mmvq_reduce_store_batched<THREADS, N, OutT>(acc, dst, row, n_rows);
}
