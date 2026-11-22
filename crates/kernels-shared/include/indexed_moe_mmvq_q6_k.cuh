#pragma once
// Templated Q6_K indexed-MoE MMVQ, r2 (2 rows/block, 32-lane groups). Q6_K is a
// 256-element super-block: 4-bit `ql` + 2-bit `qh` reassemble a 6-bit weight
// biased by -32, with 16 signed int8 sub-block scales + one fp16 super-block
// scale. Each 32-lane group owns one output row and covers all 32 `iqs` of a
// super-block; the (raw-32)·y bias uses the DP4A identity raw·y - 32·Σy. MoE
// addressing: weight row by `expert_ids[token,slot]`, per-token activation
// `y[token,:]`, output `dst[token,slot,row]`. Backend-neutral: `dp4a` + the
// group reduce from the arch header + shared helpers; misaligned super-block
// int loads go through `load_int_b2` (a 4-byte ql/qh read faults on NVIDIA for
// odd super-blocks). `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_q6_k_r2_dp4a_row(
    const flambeau_block_q6_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int top_k, int n_sb_per_row)
{
    constexpr int QR = 2;
    constexpr int QI = 32;

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
    const flambeau_block_q6_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_sb_per_row * 8;

    const int iqs = lane_lo;
    const int bq8_offset   = QR * (iqs / (QI / 2)) * 2 + (iqs % (QI / 2)) / (QI / 4);
    const int scale_offset = (QI / 4) * (iqs / (QI / 2)) + (iqs % (QI / 2)) / (QI / 8);
    const int vh_shift     = 2 * ((iqs % (QI / 2)) / (QI / 4));
    const int qh_int_idx   = (QI / 4) * (iqs / (QI / 2)) + iqs % (QI / 4);

    float acc = 0.0f;
    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q6_K* bk = xrow + b;

        const int vl = load_int_b2((const int8_t*) bk->ql, iqs);
        const int vh = load_int_b2((const int8_t*) bk->qh, qh_int_idx) >> vh_shift;
        const int8_t* sc_ptr = bk->scales + scale_offset;

        const flambeau_block_q8_1* ya0 = y_row + (b * 8 + bq8_offset + 0);
        const flambeau_block_q8_1* ya1 = y_row + (b * 8 + bq8_offset + 2);
        const int u0 = load_int_b2(ya0->qs, iqs % 8);
        const int u1 = load_int_b2(ya1->qs, iqs % 8);
        const float d8_0 = (float) ya0->d;
        const float d8_1 = (float) ya1->d;

        float sumf = 0.0f;
        {
            const int sc   = (int) sc_ptr[0];
            const int vil  = (vl >> 0) & 0x0F0F0F0F;
            const int vih  = ((vh >> 0) << 4) & 0x30303030;
            const int raw  = vil | vih;
            const int dot  = dp4a(raw, u0, 0);
            const int sumy = dp4a(0x01010101, u0, 0);
            sumf += d8_0 * ((float) (dot - 32 * sumy) * (float) sc);
        }
        {
            const int sc   = (int) sc_ptr[4];
            const int vil  = (vl >> 4) & 0x0F0F0F0F;
            const int vih  = ((vh >> 4) << 4) & 0x30303030;
            const int raw  = vil | vih;
            const int dot  = dp4a(raw, u1, 0);
            const int sumy = dp4a(0x01010101, u1, 0);
            sumf += d8_1 * ((float) (dot - 32 * sumy) * (float) sc);
        }

        acc += (float) bk->d * sumf;
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_group<32, OutT>(acc, dst, dst_idx, lane_lo);
}
