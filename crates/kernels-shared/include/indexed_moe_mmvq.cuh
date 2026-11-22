#pragma once
// Templated single-row indexed-MoE MMVQ (decode GEMV with per-token expert
// routing). Sibling of `mmvq.cuh`: the inner product is identical
// (`quant_traits<Q>::partial` / `partial_vdr`), only the addressing differs —
// the weight row is selected by `expert_ids[token, slot]`, the activation is the
// per-token column `y[token, :]` shared across all top_k slots, and the output
// lands at the combine-kernel layout `dst[token, slot, row]`. Grid is
// (n_rows, n_tokens*top_k, 1); each block owns one output row. The block reduce
// (`mmvq_reduce_store`) is warp-size agnostic, so these are backend-neutral. `Q`
// selects the weight dtype via quant_traits; `THREADS` is the block size; `OutT`
// is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<class Q, int THREADS, class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_row(
    const typename quant_traits<Q>::block_t* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int top_k, int n_blocks_per_row)
{
    using T = quant_traits<Q>;
    constexpr int IPB  = T::INT32_PER_BLOCK;
    constexpr int STEP = THREADS / IPB;

    const int row      = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;
    if (row >= n_rows || token >= n_tokens) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const int tid = threadIdx.x;
    const int j   = tid & (IPB - 1);
    const int b0  = tid / IPB;

    const typename T::block_t* xrow =
        x + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q8_1* yrow = y + (size_t) token * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = b0; b < n_blocks_per_row; b += STEP) {
        acc += T::partial(xrow[b], yrow[b], j);
    }
    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store<THREADS, OutT>(acc, dst, dst_idx);
}

// VDR variant: each thread owns `VDR` consecutive int32 lanes of a block,
// chaining VDR dp4a into one int sum before the block-scale multiply (the
// indexed-MoE mirror of `mmvq_dp4a_vdr_row`). VDR=1 reproduces
// `indexed_moe_mmvq_row`.
template<class Q, int THREADS, int VDR, class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_vdr_row(
    const typename quant_traits<Q>::block_t* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int top_k, int n_blocks_per_row)
{
    using T = quant_traits<Q>;
    constexpr int TPB  = T::INT32_PER_BLOCK / VDR;
    constexpr int STEP = THREADS / TPB;

    const int row      = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;
    if (row >= n_rows || token >= n_tokens) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const int tid = threadIdx.x;
    const int grp = tid & (TPB - 1);
    const int b0  = tid / TPB;

    const typename T::block_t* xrow =
        x + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q8_1* yrow = y + (size_t) token * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = b0; b < n_blocks_per_row; b += STEP) {
        acc += T::template partial_vdr<VDR>(xrow[b], yrow[b], grp);
    }
    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store<THREADS, OutT>(acc, dst, dst_idx);
}

// Fused gate+up variant: two expert weight matrices (gate_w, up_w) share one
// per-token Q8_1 activation, producing two outputs — the MoE sibling of
// `mmvq_gate_up_dp4a_vdr_row`. The activation int32s are loaded once per block
// and reused for both dots (the bandwidth saving is the point of the fusion).
// `do_gate`/`do_up` carry the asymmetric-row tail; the two-output reduce is
// `mmvq_reduce_store_gate_up`.
template<class Q, int THREADS, int VDR, class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_gate_up_vdr_row(
    const typename quant_traits<Q>::block_t* __restrict__ gate_w,
    const typename quant_traits<Q>::block_t* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    OutT* __restrict__ gate_out,
    OutT* __restrict__ up_out,
    int n_rows, int n_tokens, int top_k, int n_blocks_per_row)
{
    using T = quant_traits<Q>;
    constexpr int TPB  = T::INT32_PER_BLOCK / VDR;
    constexpr int STEP = THREADS / TPB;

    const int row      = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;
    if (row >= n_rows || token >= n_tokens) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const int tid = threadIdx.x;
    const int grp = tid & (TPB - 1);
    const int b0  = tid / TPB;

    const typename T::block_t* g_row =
        gate_w + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const typename T::block_t* u_row =
        up_w + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q8_1* yrow = y + (size_t) token * n_blocks_per_row;

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int b = b0; b < n_blocks_per_row; b += STEP) {
        int yi[VDR];
        load_q8_1_ints<VDR>(yrow[b], grp, yi);
        const float d_y = (float) yrow[b].d;
        acc_g += T::template partial_vdr_y<VDR>(g_row[b], yi, d_y, grp);
        acc_u += T::template partial_vdr_y<VDR>(u_row[b], yi, d_y, grp);
    }
    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_gate_up<THREADS, OutT>(acc_g, acc_u, gate_out, up_out, dst_idx, true, true);
}

// Multi-row config variant: one block computes `ROWS` output rows (a
// `THREADS/ROWS`-lane group per row), so the per-token activation reads are
// shared across ROWS weight rows and the grid shrinks by ROWS (grid.x =
// ceil(n_rows/ROWS)). The indexed-MoE mirror of `mmvq_dp4a_nrow_row`; the
// per-row group reduces with a width-(THREADS/ROWS) butterfly through
// `mmvq_reduce_store_group` (no LDS).
template<class Q, int THREADS, int ROWS, class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_nrow_row(
    const typename quant_traits<Q>::block_t* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int top_k, int n_blocks_per_row)
{
    using T = quant_traits<Q>;
    constexpr int IPB   = T::INT32_PER_BLOCK;
    constexpr int LANES = THREADS / ROWS;
    constexpr int STEP  = LANES / IPB;

    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;

    const int tid     = threadIdx.x;
    const int row_grp = tid / LANES;
    const int lane_lo = tid & (LANES - 1);
    const int j       = lane_lo & (IPB - 1);
    const int b0      = lane_lo / IPB;

    const int row = blockIdx.x * ROWS + row_grp;
    if (row >= n_rows || token >= n_tokens) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const typename T::block_t* xrow =
        x + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q8_1* yrow = y + (size_t) token * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = b0; b < n_blocks_per_row; b += STEP) {
        acc += T::partial(xrow[b], yrow[b], j);
    }
    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_group<LANES, OutT>(acc, dst, dst_idx, lane_lo);
}
