#pragma once
// Templated IQ1_S indexed-MoE MMVQ, r2 (2 rows/block, 32-lane groups). IQ1_S:
// an 11-bit index into `IQ1S_GRID` (uint64 of 8 signed-i8 magnitudes), with a
// per-sub-block 3-bit scale and a ±IQ1_DELTA bias (delta contributes via a
// dp4a vs broadcast 0x01010101 — no sign mask, the codebook is signed). Each
// 32-lane group owns one output row and covers a super-block as (iqs ∈ 0..7) ×
// (l_grp ∈ 0..3). MoE addressing: weight row by `expert_ids[token,slot]`,
// per-token activation `y[token,:]`, output `dst[token,slot,row]`. Backend-
// neutral: `IQ1S_GRID`/`IQ1_DELTA`/`dp4a` from the shared+arch headers; Q8_1
// int loads through `load_int_b2`; group reduce from the arch header.
#include "mmvq_iq1_s.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_iq1_s_r2_dp4a_row(
    const flambeau_block_iq1_s* __restrict__ x,
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

    const int lane    = threadIdx.x;
    const int row_hi  = lane >> 5;
    const int lane_lo = lane & 31;

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];
    const flambeau_block_iq1_s* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_sb_per_row * 8;

    const int iqs   = lane_lo >> 2;        // 0..7 — sub-block (no x2)
    const int l_grp = lane_lo & 3;         // 0..3
    const int l0    = 2 * l_grp;

    float acc = 0.0f;
    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        const flambeau_block_iq1_s* bk = xrow + sb;
        const float d_sb = (float) bk->d;

        const int qh_u16 = (int) bk->qh[2 * iqs] | ((int) bk->qh[2 * iqs + 1] << 8);

        const float dl    = d_sb * (2.0f * (float) ((qh_u16 >> 12) & 7) + 1.0f);
        const float delta = (qh_u16 & 0x8000) ? -IQ1_DELTA : IQ1_DELTA;

        const int idx_lo = (int) bk->qs[4 * iqs + l_grp];
        const int high3  = (qh_u16 >> (3 * l_grp)) & 7;
        const int idx    = idx_lo | (high3 << 8);

        const uint64_t grid_u64 = IQ1S_GRID[idx];
        const int grid_lo = (int) (uint32_t) grid_u64;
        const int grid_hi = (int) (uint32_t) (grid_u64 >> 32);

        const flambeau_block_q8_1* ya = y_row + sb * 8 + iqs;
        const int u0 = load_int_b2(ya->qs, l0 + 0);
        const int u1 = load_int_b2(ya->qs, l0 + 1);
        const float d_y = (float) ya->d;

        int sumi = dp4a(grid_lo, u0, 0);
        sumi     = dp4a(grid_hi, u1, sumi);

        int sum_q8 = dp4a(0x01010101, u0, 0);
        sum_q8     = dp4a(0x01010101, u1, sum_q8);

        acc += dl * d_y * ((float) sumi + delta * (float) sum_q8);
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_group<32, OutT>(acc, dst, dst_idx, lane_lo);
}
