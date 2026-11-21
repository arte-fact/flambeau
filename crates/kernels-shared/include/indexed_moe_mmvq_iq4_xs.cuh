#pragma once
// Templated IQ4_XS indexed-MoE MMVQ, r2 (2 rows/block, 32-lane groups). IQ4_XS
// is a 256-element super-block: the same 16-entry non-linear codebook as IQ4_NL
// (`iq4_nl_table_4`) with a 6-bit per-sub-block scale (`flambeau_iq4_xs_scale`
// from scales_h/scales_l). 4 lanes per sub-block × 8 sub-blocks = 32 lanes cover
// one super-block. MoE addressing: weight row by `expert_ids[token,slot]`,
// per-token activation `y[token,:]` (8 Q8_1 sub-blocks/super-block), output
// `dst[token,slot,row]`. Backend-neutral: helpers from the dense header +
// `load_int_b2`; group reduce from the arch header.
#include "mmvq_iq4_xs.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_iq4_xs_r2_dp4a_row(
    const flambeau_block_iq4_xs* __restrict__ x,
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
    const flambeau_block_iq4_xs* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_sb_per_row * 8;

    const int sub   = lane_lo >> 2;        // 0..7 — sub-block in super
    const int lane4 = lane_lo & 3;         // 0..3 — int32 in sub-block qs

    float acc = 0.0f;
    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        const flambeau_block_iq4_xs* bk = xrow + sb;

        const float d  = (float) bk->d;
        const int   ls = flambeau_iq4_xs_scale(sub, bk->scales_h, bk->scales_l);

        const int aux_q4 = load_int_b2((const int8_t*) bk->qs, sub * 4 + lane4);
        const unsigned int q_low  = (unsigned int) aux_q4 & 0x0F0F0F0Fu;
        const unsigned int q_high = ((unsigned int) aux_q4 >> 4) & 0x0F0F0F0Fu;

        const unsigned int v_lo = iq4_nl_table_4(q_low);
        const unsigned int v_hi = iq4_nl_table_4(q_high);

        const flambeau_block_q8_1* ya = y_row + sb * 8 + sub;
        const int u_lo = load_int_b2(ya->qs, lane4);
        const int u_hi = load_int_b2(ya->qs, lane4 + 4);
        const float d_y = (float) ya->d;

        int sumi = dp4a((int) v_lo, u_lo, 0);
        sumi     = dp4a((int) v_hi, u_hi, sumi);

        acc += d * (float) ls * d_y * (float) sumi;
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_group<32, OutT>(acc, dst, dst_idx, lane_lo);
}
