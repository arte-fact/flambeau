#pragma once
// Block-wide reduce-and-store tail shared by the MMVQ kernels. Each thread
// arrives with its partial `acc`; the block sums all `THREADS` lanes and the
// surviving lane writes `dst[row]` via mmvq_store. Warp-size agnostic: with one
// warp per block (`WARPS == 1`, e.g. gfx906 wave64 at THREADS=64) it is a plain
// warp reduce; with several warps it adds an inter-warp LDS rendezvous (CUDA
// wave32, or the 256-thread skeleton). The arch primitives it calls
// (`warp_reduce_sum`, `__shfl_xor`) come from the backend's arch_primitives
// header, included before this one.
#include "mmvq_store.cuh"

template<int THREADS, class OutT>
static __device__ __forceinline__ void mmvq_reduce_store(float acc, OutT* dst, int row) {
    constexpr int WARPS = THREADS / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    acc = warp_reduce_sum(acc);
    if constexpr (WARPS > 1) {
        const int warp = threadIdx.x / WARP_SIZE;
        __shared__ float s_warp[WARPS];
        if (lane == 0) s_warp[warp] = acc;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < WARPS) ? s_warp[lane] : 0.0f;
            #pragma unroll
            for (int off = WARPS / 2; off > 0; off >>= 1) {
                v += __shfl_xor(v, off, WARP_SIZE);
            }
            if (lane == 0) mmvq_store<OutT>(dst, row, v);
        }
    } else {
        if (lane == 0) mmvq_store<OutT>(dst, row, acc);
    }
}
