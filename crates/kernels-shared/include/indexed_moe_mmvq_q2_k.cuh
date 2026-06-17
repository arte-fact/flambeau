#pragma once
// Templated Q2_K indexed-MoE MMVQ, r2 (2 rows/block, 32-lane groups). Q2_K is a
// 256-element super-block: 2-bit `qs`, 4-bit (scale, min) packed in `scales`,
// fp16 `d`/`dmin`; y = d·sc·q − dmin·m. Each lane processes 2 (iqs, i) positions
// per super-block via the j ∈ {0,1} loop (j=0 → iqs 0..7, j=1 → iqs 8..15). MoE
// addressing: weight row by `expert_ids[token,slot]`, per-token activation
// `y[token,:]`, output `dst[token,slot,row]`. Backend-neutral: `dp4a` + the group
// reduce from the arch header; int loads through `load_int_b2`. `OutT` is the
// destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_q2_k_r2_dp4a_row(
    const flambeau_block_q2_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int top_k, int n_sb_per_row)
{
    constexpr int QI8_1 = 8;
    constexpr int QR2_K = 4;

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
    const flambeau_block_q2_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_sb_per_row * 8;

    const int iqs_lo = lane_lo >> 2;       // 0..7
    const int i      = lane_lo & 3;        // 0..3

    float acc = 0.0f;
    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        const flambeau_block_q2_K* bk = xrow + sb;
        const float d_sb    = (float) bk->d;
        const float dmin_sb = (float) bk->dmin;

        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int iqs          = iqs_lo + 8 * j;       // 0..15
            const int bq8_offset   = QR2_K * (iqs / QI8_1);
            const int scale_offset = iqs - (iqs & (QI8_1 - 1))
                                   + ((iqs & (QI8_1 - 1)) / (QI8_1 / 2));

            const int sc_byte  = bk->scales[scale_offset + 2 * i];
            const int sc_lo    = sc_byte & 0x0F;
            const int m_4bit   = (sc_byte >> 4) & 0x0F;
            const int m_packed = m_4bit * 0x01010101;

            const int v  = load_int_b2((const int8_t*) bk->qs, iqs);
            const int vi = (v >> (2 * i)) & 0x03030303;

            const flambeau_block_q8_1* ya = y_row + (size_t) sb * 8 + bq8_offset + i;
            const int u_i = load_int_b2(ya->qs, iqs & (QI8_1 - 1));
            const float d_y = (float) ya->d;

            const int sumi_d = dp4a(vi, u_i, 0);
            const int sumi_m = dp4a(m_packed, u_i, 0);

            acc += d_y * (d_sb * (float) sc_lo * (float) sumi_d - dmin_sb * (float) sumi_m);
        }
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_group<32, OutT>(acc, dst, dst_idx, lane_lo);
}
