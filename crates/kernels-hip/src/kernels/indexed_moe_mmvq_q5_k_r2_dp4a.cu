// indexed_moe_mmvq_q5_k_r2_dp4a — Q5_K MoE MMVQ with DP4A inner loop.
// Drop-in replacement for `indexed_moe_mmvq_q5_k.cu` (scalar FP32). Same
// per-element math as dense `mmvq_q5_k_dp4a` plus Q4_K MoE r2 wiring
// (expert_ids indirection, 2 output rows per block, half-warp DPP reduce).
//
// Launch: blockDim = { 64 } (wave64), gridDim = { n_row_pairs,
// n_tokens * top_k, 1 }. 2 output rows per block (lanes 0..31 own row R,
// lanes 32..63 own row R+1). Each lane covers ONE int32 of pair-qs (4
// low nibbles → sub_lo, 4 high nibbles → sub_hi) + matching qh int32 for
// the 5th bit.
//
// Per (pair, iqs): two dp4as for the 4-element products (sub_lo, sub_hi)
// + two dp4as for the sum-of-y correction. Pair has 8 int32 of qs, so
// 8 lanes per pair × 4 pairs = 32 lanes per row.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"

static __device__ __forceinline__ int flambeau_indexed_moe_q5_k_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q5_k_r2_dp4a_q8_1(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    float* __restrict__ dst,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row
) {
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
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    const int pair_idx = lane_lo >> 3;
    const int iqs      = lane_lo & 7;
    const int sub_lo   = pair_idx * 2;
    const int sub_hi   = sub_lo + 1;

    float acc = 0.0f;

    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q5_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        const int* qs_pair = (const int*) &bk->qs[pair_idx * 32];
        const int qs_int   = qs_pair[iqs];
        const int vlq_lo   = qs_int & 0x0F0F0F0F;
        const int vlq_hi   = (qs_int >> 4) & 0x0F0F0F0F;

        const int vh = *((const int*) &bk->qh[iqs * 4]);
        const int vhq_lo = ((vh >> sub_lo) << 4) & 0x10101010;
        const int vhq_hi = ((vh >> sub_hi) << 4) & 0x10101010;

        const int v_lo = vlq_lo | vhq_lo;
        const int v_hi = vlq_hi | vhq_hi;

        uint8_t sc_lo = 0, m_lo = 0, sc_hi = 0, m_hi = 0;
        flambeau_q4k_scale_min(sub_lo, bk->scales, &sc_lo, &m_lo);
        flambeau_q4k_scale_min(sub_hi, bk->scales, &sc_hi, &m_hi);

        const flambeau_block_q8_1* ya_lo = y_row + (b * 8 + sub_lo);
        const flambeau_block_q8_1* ya_hi = y_row + (b * 8 + sub_hi);
        const int u_lo = ((const int*) ya_lo->qs)[iqs];
        const int u_hi = ((const int*) ya_hi->qs)[iqs];

        const int sumi_lo = flambeau_indexed_moe_q5_k_dp4a(v_lo, u_lo, 0);
        const int sumi_hi = flambeau_indexed_moe_q5_k_dp4a(v_hi, u_hi, 0);
        const int summ_lo = flambeau_indexed_moe_q5_k_dp4a(0x01010101, u_lo, 0);
        const int summ_hi = flambeau_indexed_moe_q5_k_dp4a(0x01010101, u_hi, 0);

        const float d_y_lo = (float) ya_lo->d;
        const float d_y_hi = (float) ya_hi->d;

        acc += d_y_lo *
               (d * (float) sc_lo * (float) sumi_lo - dmin * (float) m_lo * (float) summ_lo);
        acc += d_y_hi *
               (d * (float) sc_hi * (float) sumi_hi - dmin * (float) m_hi * (float) summ_hi);
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
