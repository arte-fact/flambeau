#pragma once
// Templated Q4_K indexed-MoE MMVQ, r2 (2 rows/block, 32-lane groups) and r4
// (4 rows/block, 16-lane groups). Same per-super-block dot as the dense
// `mmvq_q4_k_r2_dp4a_row` (llama.cpp's `vec_dot_q4_K_q8_1_impl_vmmq`: two dp4a —
// value + broadcast-min — per sub-block, y = d·sc·q − dmin·m), shared here as
// `flambeau_q4k_subblock_dot` and reused by both row counts. MoE addressing:
// weight row by `expert_ids[token,slot]`, per-token activation `y[token,:]`
// (8 Q8_1 sub-blocks per super-block), output `dst[token,slot,row]`. The
// per-row group reduces through `mmvq_reduce_store_group` (width 32 / 16,
// ascending offsets = the gfx906 half/quarter-warp DPP add order), so
// backend-neutral. `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

// Forward: the fused gate+up r4 sorted-reorder variant lives at the bottom of
// this header (it reuses flambeau_q4k_subblock_dot defined just below).

// One Q4_K super-block × one int32 slice (`iqs`) of its `pair_idx` sub-block
// pair → the (low, high)-nibble contribution to the row dot.
static __device__ __forceinline__ float flambeau_q4k_subblock_dot(
    const flambeau_block_q4_K* __restrict__ bk,
    const flambeau_block_q8_1* __restrict__ y_row,
    int b, int pair_idx, int iqs, int sub_lo, int sub_hi)
{
    const float d    = (float) bk->d;
    const float dmin = (float) bk->dmin;

    const int qs_int = load_int_b2((const int8_t*) bk->qs, pair_idx * 8 + iqs);
    const int q_lo = qs_int & 0x0F0F0F0F;
    const int q_hi = (qs_int >> 4) & 0x0F0F0F0F;

    uint8_t sc_lo = 0, m_lo = 0, sc_hi = 0, m_hi = 0;
    flambeau_q4k_scale_min(sub_lo, bk->scales, &sc_lo, &m_lo);
    flambeau_q4k_scale_min(sub_hi, bk->scales, &sc_hi, &m_hi);

    const flambeau_block_q8_1* ya_lo = y_row + (b * 8 + sub_lo);
    const flambeau_block_q8_1* ya_hi = y_row + (b * 8 + sub_hi);
    const int u_lo = load_int_b2(ya_lo->qs, iqs);
    const int u_hi = load_int_b2(ya_hi->qs, iqs);

    const int sumi_lo = dp4a(q_lo, u_lo, 0);
    const int sumi_hi = dp4a(q_hi, u_hi, 0);
    const int summ_lo = dp4a(0x01010101, u_lo, 0);
    const int summ_hi = dp4a(0x01010101, u_hi, 0);

    return (float) ya_lo->d * (d * (float) sc_lo * (float) sumi_lo - dmin * (float) m_lo * (float) summ_lo)
         + (float) ya_hi->d * (d * (float) sc_hi * (float) sumi_hi - dmin * (float) m_hi * (float) summ_hi);
}

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_q4_k_r2_dp4a_row(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int top_k, int n_sb_per_row)
{
    const int row_pair = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;
    if (token >= n_tokens) return;

    const int lane    = threadIdx.x;       // 0..63
    const int row_hi  = lane >> 5;         // 0/1 — row in the pair
    const int lane_lo = lane & 31;         // 0..31 — intra-row lane

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];
    const flambeau_block_q4_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_sb_per_row * 8;

    const int pair_idx = lane_lo >> 3;     // 0..3
    const int iqs      = lane_lo & 7;      // 0..7
    const int sub_lo   = pair_idx * 2;
    const int sub_hi   = sub_lo + 1;

    float acc = 0.0f;
    for (int b = 0; b < n_sb_per_row; ++b) {
        acc += flambeau_q4k_subblock_dot(xrow + b, y_row, b, pair_idx, iqs, sub_lo, sub_hi);
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_group<32, OutT>(acc, dst, dst_idx, lane_lo);
}

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_q4_k_r4_dp4a_row(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int top_k, int n_sb_per_row)
{
    const int row_quad = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;
    if (token >= n_tokens) return;

    const int lane    = threadIdx.x;       // 0..63
    const int row_idx = lane >> 4;         // 0..3 — row in the quad
    const int lane_lo = lane & 15;         // 0..15 — intra-row lane

    const int row = row_quad * 4 + row_idx;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];
    const flambeau_block_q4_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_sb_per_row * 8;

    const int pair_idx = lane_lo >> 2;     // 0..3
    const int iqs_base = lane_lo & 3;      // 0..3 — each lane covers iqs_base, +4
    const int sub_lo   = pair_idx * 2;
    const int sub_hi   = sub_lo + 1;

    float acc = 0.0f;
    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q4_K* bk = xrow + b;
        #pragma unroll
        for (int ipair = 0; ipair < 2; ++ipair) {
            acc += flambeau_q4k_subblock_dot(bk, y_row, b, pair_idx, iqs_base + ipair * 4, sub_lo, sub_hi);
        }
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_group<16, OutT>(acc, dst, dst_idx, lane_lo);
}

// Fused gate+up Q4_K MoE MMVQ, r4 (4 rows/16-lane groups) with a sorted-reorder
// of blockIdx.y: `sorted_pair_idx[block.y]` → the original (token, slot) pair,
// so blocks adjacent in grid.y share weight-tile L2 residency (the experts are
// pre-sorted). Reuses `flambeau_q4k_subblock_dot` for both weights; the two
// outputs reduce through two width-16 group butterflies.
template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_q4_k_gate_up_r4_sorted_dp4a_row(
    const flambeau_block_q4_K* __restrict__ gate_w,
    const flambeau_block_q4_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx,
    OutT* __restrict__ gate_out,
    OutT* __restrict__ up_out,
    int n_rows, int n_tokens, int top_k, int n_sb_per_row)
{
    const int row_quad     = blockIdx.x;
    const int slot_ordered = blockIdx.y;
    const int slot         = sorted_pair_idx[slot_ordered];
    const int token        = slot / top_k;
    const int slot_idx      = slot - token * top_k;
    if (token >= n_tokens) return;

    const int lane    = threadIdx.x;       // 0..63
    const int row_idx = lane >> 4;         // 0..3
    const int lane_lo = lane & 15;         // 0..15

    const int row = row_quad * 4 + row_idx;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];
    const flambeau_block_q4_K* gate_row =
        gate_w + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q4_K* up_row =
        up_w + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_sb_per_row * 8;

    const int pair_idx = lane_lo >> 2;     // 0..3
    const int iqs_base = lane_lo & 3;      // 0..3
    const int sub_lo   = pair_idx * 2;
    const int sub_hi   = sub_lo + 1;

    float acc_gate = 0.0f, acc_up = 0.0f;
    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q4_K* gbk = gate_row + b;
        const flambeau_block_q4_K* ubk = up_row + b;
        #pragma unroll
        for (int ipair = 0; ipair < 2; ++ipair) {
            const int iqs = iqs_base + ipair * 4;
            acc_gate += flambeau_q4k_subblock_dot(gbk, y_row, b, pair_idx, iqs, sub_lo, sub_hi);
            acc_up   += flambeau_q4k_subblock_dot(ubk, y_row, b, pair_idx, iqs, sub_lo, sub_hi);
        }
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_group<16, OutT>(acc_gate, gate_out, dst_idx, lane_lo);
    mmvq_reduce_store_group<16, OutT>(acc_up, up_out, dst_idx, lane_lo);
}

// Fused gate+up Q4_K MoE MMVQ, r4 (4 rows/16-lane groups), unsorted: `blockIdx.y`
// is the (token, slot) pair directly. Same per-row dot as the sorted variant
// above, dropping the `sorted_pair_idx` indirection; the two width-16 group
// butterflies reduce the gate and up partials. Backend-neutral via
// `flambeau_q4k_subblock_dot` + `mmvq_reduce_store_group`.
template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_q4_k_gate_up_r4_dp4a_row(
    const flambeau_block_q4_K* __restrict__ gate_w,
    const flambeau_block_q4_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    OutT* __restrict__ gate_out,
    OutT* __restrict__ up_out,
    int n_rows, int n_tokens, int top_k, int n_sb_per_row)
{
    const int row_quad = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;
    if (token >= n_tokens) return;

    const int lane    = threadIdx.x;       // 0..63
    const int row_idx = lane >> 4;         // 0..3
    const int lane_lo = lane & 15;         // 0..15

    const int row = row_quad * 4 + row_idx;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];
    const flambeau_block_q4_K* gate_row =
        gate_w + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q4_K* up_row =
        up_w + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_sb_per_row * 8;

    const int pair_idx = lane_lo >> 2;     // 0..3
    const int iqs_base = lane_lo & 3;      // 0..3
    const int sub_lo   = pair_idx * 2;
    const int sub_hi   = sub_lo + 1;

    float acc_gate = 0.0f, acc_up = 0.0f;
    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q4_K* gbk = gate_row + b;
        const flambeau_block_q4_K* ubk = up_row + b;
        #pragma unroll
        for (int ipair = 0; ipair < 2; ++ipair) {
            const int iqs = iqs_base + ipair * 4;
            acc_gate += flambeau_q4k_subblock_dot(gbk, y_row, b, pair_idx, iqs, sub_lo, sub_hi);
            acc_up   += flambeau_q4k_subblock_dot(ubk, y_row, b, pair_idx, iqs, sub_lo, sub_hi);
        }
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_group<16, OutT>(acc_gate, gate_out, dst_idx, lane_lo);
    mmvq_reduce_store_group<16, OutT>(acc_up, up_out, dst_idx, lane_lo);
}
