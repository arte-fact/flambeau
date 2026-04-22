// indexed_moe_mmvq_q4_k_r4_sorted_dp4a — V2.5.b sorted-reorder variant of
// V2.4.b's r4 down-projection kernel.
//
// Same arithmetic / launch shape as `indexed_moe_mmvq_q4_k_r4_dp4a`, plus
// a `sorted_pair_idx[total]` lookup that remaps `blockIdx.y` to the
// original (token, slot) pair. Adjacent grid.y blocks share an expert
// when pairs are sorted → L2 cache reuse on weight tiles.

#include "block_quant.cuh"
#include "gfx906.cuh"

static __device__ __forceinline__ int flambeau_dp4a_q4k_r4s(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q4_k_r4_sorted_dp4a_q8_1(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx,
    float* __restrict__ dst,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row
) {
    const int row_quad = blockIdx.x;
    const int slot_ordered = blockIdx.y;
    const int slot = sorted_pair_idx[slot_ordered];
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;

    if (token >= n_tokens) return;

    const int lane    = threadIdx.x;
    const int row_idx = lane >> 4;
    const int lane_lo = lane & 15;

    const int row = row_quad * 4 + row_idx;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const flambeau_block_q4_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    const int pair_idx = lane_lo >> 2;
    const int iqs_base = lane_lo & 3;
    const int sub_lo   = pair_idx * 2;
    const int sub_hi   = sub_lo + 1;

    float acc = 0.0f;

    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q4_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        uint8_t sc_lo = 0, m_lo = 0, sc_hi = 0, m_hi = 0;
        flambeau_q4k_scale_min(sub_lo, bk->scales, &sc_lo, &m_lo);
        flambeau_q4k_scale_min(sub_hi, bk->scales, &sc_hi, &m_hi);

        const flambeau_block_q8_1* ya_lo = y_row + (b * 8 + sub_lo);
        const flambeau_block_q8_1* ya_hi = y_row + (b * 8 + sub_hi);
        const float d_y_lo = (float) ya_lo->d;
        const float d_y_hi = (float) ya_hi->d;

        #pragma unroll
        for (int ipair = 0; ipair < 2; ++ipair) {
            const int iqs = iqs_base + ipair * 4;

            const int qs_int = ((const int*) bk->qs)[pair_idx * 8 + iqs];
            const int q_lo = qs_int & 0x0F0F0F0F;
            const int q_hi = (qs_int >> 4) & 0x0F0F0F0F;

            const int u_lo = ((const int*) ya_lo->qs)[iqs];
            const int u_hi = ((const int*) ya_hi->qs)[iqs];

            const int sumi_lo = flambeau_dp4a_q4k_r4s(q_lo, u_lo, 0);
            const int sumi_hi = flambeau_dp4a_q4k_r4s(q_hi, u_hi, 0);
            const int summ_lo = flambeau_dp4a_q4k_r4s(0x01010101, u_lo, 0);
            const int summ_hi = flambeau_dp4a_q4k_r4s(0x01010101, u_hi, 0);

            acc += d_y_lo *
                   (d * (float) sc_lo * (float) sumi_lo - dmin * (float) m_lo * (float) summ_lo);
            acc += d_y_hi *
                   (d * (float) sc_hi * (float) sumi_hi - dmin * (float) m_hi * (float) summ_hi);
        }
    }

    acc = gfx906_quarter_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
