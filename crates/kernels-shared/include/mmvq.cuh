#pragma once
// Templated single-row MMVQ (decode GEMV): quantized weight rows × one Q8_1
// activation column -> F32. One thread block per output row; each thread owns
// one int32 (4 packed quants) of a block; `dp4a` inner product; a warp reduce
// then an inter-warp reduce through shared memory. `Q` selects the weight dtype
// via quant_traits; `THREADS` is the block size. The arch primitives it calls
// (`dp4a`, `warp_reduce_sum`, `__shfl_xor`) come from the backend's
// arch_primitives header, included before this one. Warp-size agnostic: the
// block stride and warp count derive from WARP_SIZE.
#include "quant_traits.cuh"

template<class Q, int THREADS>
static __device__ __forceinline__ void mmvq_dp4a_row(
    const typename quant_traits<Q>::block_t* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    int n_rows, int n_blocks_per_row)
{
    using T = quant_traits<Q>;
    constexpr int IPB   = T::INT32_PER_BLOCK;     // int32s per block
    constexpr int STEP  = THREADS / IPB;          // blocks advanced per iter
    constexpr int WARPS = THREADS / WARP_SIZE;

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid & (WARP_SIZE - 1);
    const int j    = tid & (IPB - 1);             // int32 index within a block
    const int b0   = tid / IPB;                   // 0..STEP-1

    const typename T::block_t* xrow = x + (size_t) row * n_blocks_per_row;
    float acc = 0.0f;
    for (int b = b0; b < n_blocks_per_row; b += STEP) {
        acc += T::partial(xrow[b], y[b], j);
    }
    acc = warp_reduce_sum(acc);

    __shared__ float s_warp[WARPS];
    if (lane == 0) s_warp[warp] = acc;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) dst[row] = v;
    }
}
