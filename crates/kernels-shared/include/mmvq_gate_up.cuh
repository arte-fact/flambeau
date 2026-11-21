#pragma once
// Templated fused gate+up MMVQ (decode GEMV): two weight matrices (gate_w,
// up_w) share one Q8_1 activation column, producing two outputs. One thread
// block per output row; the activation int32s are read once per block and
// reused for both the gate and up dot products — the bandwidth saving over two
// separate MMVQ launches is the point of the fusion. `Q` selects the weight
// dtype via quant_traits; `THREADS` is the block size, `VDR` the dp4a chain
// length per thread, `OutT` the destination dtype. The arch primitives (`dp4a`,
// `warp_reduce_sum`, `__shfl_xor`) come from the backend's arch_primitives
// header, included before this one.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<class Q, int THREADS, int VDR, class OutT = float>
static __device__ __forceinline__ void mmvq_gate_up_dp4a_vdr_row(
    const typename quant_traits<Q>::block_t* __restrict__ gate_w,
    const typename quant_traits<Q>::block_t* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ gate_out,
    OutT* __restrict__ up_out,
    int n_rows_gate, int n_rows_up, int n_blocks_per_row)
{
    using T = quant_traits<Q>;
    constexpr int TPB  = T::INT32_PER_BLOCK / VDR;   // threads per quant block
    constexpr int STEP = THREADS / TPB;              // blocks advanced per iter

    const int row = blockIdx.x;
    const bool do_gate = row < n_rows_gate;
    const bool do_up   = row < n_rows_up;
    if (!do_gate && !do_up) return;

    const int tid = threadIdx.x;
    const int grp = tid & (TPB - 1);
    const int b0  = tid / TPB;

    const typename T::block_t* g_row = gate_w + (size_t) row * n_blocks_per_row;
    const typename T::block_t* u_row = up_w   + (size_t) row * n_blocks_per_row;

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int b = b0; b < n_blocks_per_row; b += STEP) {
        int yi[VDR];
        load_q8_1_ints<VDR>(y[b], grp, yi);
        const float d_y = (float) y[b].d;
        if (do_gate) acc_g += T::template partial_vdr_y<VDR>(g_row[b], yi, d_y, grp);
        if (do_up)   acc_u += T::template partial_vdr_y<VDR>(u_row[b], yi, d_y, grp);
    }
    mmvq_reduce_store_gate_up<THREADS, OutT>(acc_g, acc_u, gate_out, up_out, row, do_gate, do_up);
}
