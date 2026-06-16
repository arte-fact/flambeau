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

// Reduce-and-store for the r2 (2 output rows per 64-thread block) MMVQ kernels.
// Each row is owned by a contiguous 32-lane group — a sub-warp on wave64 (two
// groups share one warp) or a full warp on wave32 (the two rows are two warps).
// A width-32 `__shfl_xor` butterfly reduces within the group on both, with no
// LDS / cross-warp traffic. `32` here is the row-group width (THREADS/ROWS =
// 64/2), not a wave-size assumption. The ascending offset order (1,2,4,8,16)
// matches gfx906's DPP `half_warp_reduce_sum` add order so the HIP result stays
// bit-identical to the hand-written kernel. `lane_lo` is the caller's intra-row
// lane (`threadIdx.x & 31`).
template<class OutT>
static __device__ __forceinline__ void mmvq_reduce_store_2row(
    float acc, OutT* dst, int row, int lane_lo) {
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        acc += __shfl_xor(acc, off, 32);
    }
    if (lane_lo == 0) mmvq_store<OutT>(dst, row, acc);
}

// Width-GROUP generalisation of the 2-row reduce for the multi-row skeleton: a
// contiguous GROUP-lane group owns one output row and reduces with a width-GROUP
// `__shfl_xor` butterfly (no LDS). GROUP = THREADS/ROWS (32 for r2, 16 for r4).
// On wave64 each group is a sub-warp; on wave32 a group is one warp (r2) or a
// half-warp (r4) — the explicit width keeps the reduce inside the group either
// way. Ascending offsets (1,2,…,GROUP/2) match gfx906's DPP add order. `lane_lo`
// is the caller's intra-group lane (`threadIdx.x & (GROUP-1)`).
template<int GROUP, class OutT>
static __device__ __forceinline__ void mmvq_reduce_store_group(
    float acc, OutT* dst, int row, int lane_lo) {
    #pragma unroll
    for (int off = 1; off < GROUP; off <<= 1) {
        acc += __shfl_xor(acc, off, GROUP);
    }
    if (lane_lo == 0) mmvq_store<OutT>(dst, row, acc);
}

// Two-output reduce-and-store for the fused gate+up MMVQ kernel: each thread
// arrives with its gate and up partials; the block reduces both through one
// barrier (separate LDS arrays) and the surviving lane writes `gate_out[row]`
// / `up_out[row]`. `do_gate`/`do_up` gate the asymmetric-row tail (an FFN where
// the gate and up projections differ in row count). Same warp-then-inter-warp
// structure as `mmvq_reduce_store`.
template<int THREADS, class OutT>
static __device__ __forceinline__ void mmvq_reduce_store_gate_up(
    float acc_g, float acc_u, OutT* gate_out, OutT* up_out, int row,
    bool do_gate, bool do_up) {
    constexpr int WARPS = THREADS / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    acc_g = warp_reduce_sum(acc_g);
    acc_u = warp_reduce_sum(acc_u);
    if constexpr (WARPS > 1) {
        const int warp = threadIdx.x / WARP_SIZE;
        __shared__ float s_g[WARPS];
        __shared__ float s_u[WARPS];
        if (lane == 0) { s_g[warp] = acc_g; s_u[warp] = acc_u; }
        __syncthreads();
        if (warp == 0) {
            float g = (lane < WARPS) ? s_g[lane] : 0.0f;
            float u = (lane < WARPS) ? s_u[lane] : 0.0f;
            #pragma unroll
            for (int off = WARPS / 2; off > 0; off >>= 1) {
                g += __shfl_xor(g, off, WARP_SIZE);
                u += __shfl_xor(u, off, WARP_SIZE);
            }
            if (lane == 0) {
                if (do_gate) mmvq_store<OutT>(gate_out, row, g);
                if (do_up)   mmvq_store<OutT>(up_out, row, u);
            }
        }
    } else {
        if (lane == 0) {
            if (do_gate) mmvq_store<OutT>(gate_out, row, acc_g);
            if (do_up)   mmvq_store<OutT>(up_out, row, acc_u);
        }
    }
}

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
