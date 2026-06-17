#pragma once
// Templated IQ1_M indexed-MoE MMVQ, r2 (2 rows/block, 32-lane groups). IQ1_M
// extends IQ1_S: the super-block `d` is reassembled from four 16-bit `scales`
// words (`iq1_m_reassemble_d`), with per-half-sub-block 3-bit scales and a
// per-l_grp ±IQ1_DELTA sign. Each 32-lane group owns one output row and covers
// a super-block as (iqs ∈ 0..7) × (l_grp ∈ 0..3). MoE addressing: weight row by
// `expert_ids[token,slot]`, per-token activation `y[token,:]`, output
// `dst[token,slot,row]`. Backend-neutral: helpers from the dense header +
// `load_int_b2`; group reduce from the arch header.
#include "mmvq_iq1_m.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_iq1_m_r2_dp4a_row(
    const flambeau_block_iq1_m* __restrict__ x,
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
    const flambeau_block_iq1_m* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_sb_per_row * 8;

    const int iqs   = lane_lo >> 2;        // 0..7
    const int l_grp = lane_lo & 3;         // 0..3
    const int l0    = 2 * l_grp;

    float acc = 0.0f;
    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        const flambeau_block_iq1_m* bk = xrow + sb;

        const float d_super = iq1_m_reassemble_d(bk->scales);

        const int sc_word = (int) bk->scales[2 * (iqs >> 1)]
                          | ((int) bk->scales[2 * (iqs >> 1) + 1] << 8);
        const int shift1 = 6 * (iqs & 1);
        const int dl_scale = (l_grp < 2)
            ? ((sc_word >> shift1) & 7)
            : ((sc_word >> (shift1 + 3)) & 7);
        const float dl = d_super * (2.0f * (float) dl_scale + 1.0f);

        const int qh_pick = (l_grp < 2) ? (2 * iqs) : (2 * iqs + 1);
        const uint8_t qh_byte = bk->qh[qh_pick];
        const int shift_idx = 8 - 4 * (l_grp & 1);
        const int idx = (int) bk->qs[4 * iqs + l_grp]
                      | (((int) qh_byte << shift_idx) & 0x700);
        const int delta_bit = (l_grp & 1) == 0 ? 0x08 : 0x80;
        const float delta = (qh_byte & delta_bit) ? -IQ1_DELTA : IQ1_DELTA;

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
