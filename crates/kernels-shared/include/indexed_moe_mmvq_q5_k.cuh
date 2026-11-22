#pragma once
// Templated Q5_K indexed-MoE MMVQ, r2 (2 rows/block, 32-lane groups). Q5_K is a
// 256-element super-block: 4-bit `qs` + 1-bit `qh` reassemble a 5-bit weight,
// with the same packed 6-bit (scale, min) pairs as Q4_K (`flambeau_q4k_scale_min`)
// and fp16 `d`/`dmin`; y = d·sc·q − dmin·m. Each lane owns one int32 of a
// sub-block pair (4 low nibbles → sub_lo, 4 high → sub_hi) + the matching qh
// int32. MoE addressing: weight row by `expert_ids[token,slot]`, per-token
// activation `y[token,:]`, output `dst[token,slot,row]`. Backend-neutral: `dp4a`
// + the group reduce from the arch header + shared helpers; int loads through
// `load_int_b2`. `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_q5_k_r2_dp4a_row(
    const flambeau_block_q5_K* __restrict__ x,
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
    const flambeau_block_q5_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_sb_per_row * 8;

    const int pair_idx = lane_lo >> 3;
    const int iqs      = lane_lo & 7;
    const int sub_lo   = pair_idx * 2;
    const int sub_hi   = sub_lo + 1;

    float acc = 0.0f;
    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q5_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        const int qs_int = load_int_b2((const int8_t*) bk->qs, pair_idx * 8 + iqs);
        const int vlq_lo = qs_int & 0x0F0F0F0F;
        const int vlq_hi = (qs_int >> 4) & 0x0F0F0F0F;

        const int vh = load_int_b2((const int8_t*) bk->qh, iqs);
        const int vhq_lo = ((vh >> sub_lo) << 4) & 0x10101010;
        const int vhq_hi = ((vh >> sub_hi) << 4) & 0x10101010;

        const int v_lo = vlq_lo | vhq_lo;
        const int v_hi = vlq_hi | vhq_hi;

        uint8_t sc_lo = 0, m_lo = 0, sc_hi = 0, m_hi = 0;
        flambeau_q4k_scale_min(sub_lo, bk->scales, &sc_lo, &m_lo);
        flambeau_q4k_scale_min(sub_hi, bk->scales, &sc_hi, &m_hi);

        const flambeau_block_q8_1* ya_lo = y_row + (b * 8 + sub_lo);
        const flambeau_block_q8_1* ya_hi = y_row + (b * 8 + sub_hi);
        const int u_lo = load_int_b2(ya_lo->qs, iqs);
        const int u_hi = load_int_b2(ya_hi->qs, iqs);

        const int sumi_lo = dp4a(v_lo, u_lo, 0);
        const int sumi_hi = dp4a(v_hi, u_hi, 0);
        const int summ_lo = dp4a(0x01010101, u_lo, 0);
        const int summ_hi = dp4a(0x01010101, u_hi, 0);

        const float d_y_lo = (float) ya_lo->d;
        const float d_y_hi = (float) ya_hi->d;

        acc += d_y_lo * (d * (float) sc_lo * (float) sumi_lo - dmin * (float) m_lo * (float) summ_lo);
        acc += d_y_hi * (d * (float) sc_hi * (float) sumi_hi - dmin * (float) m_hi * (float) summ_hi);
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_group<32, OutT>(acc, dst, dst_idx, lane_lo);
}
