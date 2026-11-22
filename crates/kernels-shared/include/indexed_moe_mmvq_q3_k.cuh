#pragma once
// Templated Q3_K indexed-MoE MMVQ, r2 (2 rows/block, 32-lane groups). Q3_K is a
// 256-element super-block: 2-bit `qs` + 1-bit `hmask` reassemble a signed 3-bit
// weight, with 6-bit sub-block scales packed across `scales[12]` (biased -32)
// and fp16 `d`. The signed dot is `sumi_l - sumi_h` (qs value minus the
// hmask-cleared high bit). Each lane processes 2 (iqs, i) positions per
// super-block via the j ∈ {0,1} loop. MoE addressing: weight row by
// `expert_ids[token,slot]`, per-token activation `y[token,:]`, output
// `dst[token,slot,row]`. Backend-neutral: `dp4a` + the group reduce from the
// arch header; int loads through `load_int_b2`. `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_q3_k_r2_dp4a_row(
    const flambeau_block_q3_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int top_k, int n_sb_per_row)
{
    constexpr int QI3_K = 16;
    constexpr int QR3_K = 4;
    constexpr int QI8_1 = 8;

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
    const flambeau_block_q3_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_sb_per_row * 8;

    const int iqs_lo = lane_lo >> 2;       // 0..7
    const int i      = lane_lo & 3;        // 0..3

    float acc = 0.0f;
    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        const flambeau_block_q3_K* bk = xrow + sb;
        const float d = (float) bk->d;

        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int iqs          = iqs_lo + 8 * j;       // 0..15
            const int bq8_offset   = QR3_K * (iqs / (QI3_K / 2));
            const int scale_offset = iqs - (iqs & (QI8_1 - 1))
                                   + ((iqs & (QI8_1 - 1)) / (QI8_1 / 2));
            const int isc           = scale_offset + 2 * i;
            const int isc_low       = isc & 7;
            const int sc_shift_low  = 4 * (isc >> 3);
            const int isc_high      = isc & 3;
            const int sc_shift_high = 2 * (isc >> 2);

            const int sc_low  = (bk->scales[isc_low] >> sc_shift_low) & 0xF;
            const int sc_high = ((bk->scales[(QI3_K / 2) + isc_high] >> sc_shift_high) & 3) << 4;
            const int sc      = (sc_low | sc_high) - 32;

            const int vl      = load_int_b2((const int8_t*) bk->qs, iqs);
            const int vh_full = ~load_int_b2((const int8_t*) bk->hmask, iqs & ((QI3_K / 2) - 1));
            const int vh      = vh_full >> bq8_offset;

            const int vil = (vl >> (2 * i)) & 0x03030303;
            const int vih = ((vh >> i) << 2) & 0x04040404;

            const flambeau_block_q8_1* ya = y_row + sb * 8 + bq8_offset + i;
            const int u_i = load_int_b2(ya->qs, iqs & (QI8_1 - 1));
            const float d_y = (float) ya->d;

            const int sumi_l = dp4a(vil, u_i, 0);
            const int sumi_h = dp4a(vih, u_i, 0);
            const int sumi   = sumi_l - sumi_h;

            acc += d * (float) sc * d_y * (float) sumi;
        }
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_group<32, OutT>(acc, dst, dst_idx, lane_lo);
}
