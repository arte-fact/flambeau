// indexed_moe_mmvq_q4_k_gate_up_r4_sorted_dp4a — sorted-reorder
// variant of the r4 gate_up kernel.
// Identical arithmetic / launch shape to
// `indexed_moe_mmvq_q4_k_gate_up_r4_dp4a`, but takes an additional
// `sorted_pair_idx[total]` lookup array that remaps `blockIdx.y` to an
// original (token, slot) pair. produces this array grouping pairs
// by expert.
// Why: in the unsorted baseline, adjacent blocks along grid.y touch
// DIFFERENT experts → each block fetches its own weight slab cold from
// HBM. With the remap, adjacent blocks along grid.y touch the SAME
// expert → the next block's weights are warm in L1/L2 from the previous
// block's fetch. For Qwen3.6-35B: ~16 tokens per expert typical, so 16
// consecutive blocks share the expert-weight working set.
// One-line change vs r4_dp4a: the first two lines of kernel body.
// Output layout unchanged — still writes via the original token/slot_idx.

#include "block_quant.cuh"
#include "gfx906.cuh"

static __device__ __forceinline__ int flambeau_dp4a_q4k_gu_r4s(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q4_k_gate_up_r4_sorted_dp4a_q8_1(
    const flambeau_block_q4_K* __restrict__ gate_w,
    const flambeau_block_q4_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx,  // output: permutes block.y → original pair_idx
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row
) {
    const int row_quad = blockIdx.x;
    const int slot_ordered = blockIdx.y;     // sorted position
    // Remap to original (token, slot_idx) — blocks sorted by expert, so
    // adjacent blocks share weight-tile L2 residency.
    const int slot = sorted_pair_idx[slot_ordered];
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;

    if (token >= n_tokens) return;

    const int lane    = threadIdx.x;              // 0..63
    const int row_idx = lane >> 4;                // 0..3
    const int lane_lo = lane & 15;                // 0..15

    const int row = row_quad * 4 + row_idx;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const flambeau_block_q4_K* gate_row =
        gate_w + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q4_K* up_row =
        up_w   + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    const int pair_idx = lane_lo >> 2;             // 0..3
    const int iqs_base = lane_lo & 3;              // 0..3
    const int sub_lo   = pair_idx * 2;
    const int sub_hi   = sub_lo + 1;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q4_K* gbk = gate_row + b;
        const flambeau_block_q4_K* ubk = up_row + b;
        const float g_d    = (float) gbk->d;
        const float g_dmin = (float) gbk->dmin;
        const float u_d    = (float) ubk->d;
        const float u_dmin = (float) ubk->dmin;

        const flambeau_block_q8_1* y_sb = y_row + b * 8;
        const flambeau_block_q8_1* ya_lo = y_sb + sub_lo;
        const flambeau_block_q8_1* ya_hi = y_sb + sub_hi;
        const float d_y_lo = (float) ya_lo->d;
        const float d_y_hi = (float) ya_hi->d;

        uint8_t sc_lo_g = 0, m_lo_g = 0, sc_hi_g = 0, m_hi_g = 0;
        uint8_t sc_lo_u = 0, m_lo_u = 0, sc_hi_u = 0, m_hi_u = 0;
        flambeau_q4k_scale_min(sub_lo, gbk->scales, &sc_lo_g, &m_lo_g);
        flambeau_q4k_scale_min(sub_hi, gbk->scales, &sc_hi_g, &m_hi_g);
        flambeau_q4k_scale_min(sub_lo, ubk->scales, &sc_lo_u, &m_lo_u);
        flambeau_q4k_scale_min(sub_hi, ubk->scales, &sc_hi_u, &m_hi_u);

        #pragma unroll
        for (int ipair = 0; ipair < 2; ++ipair) {
            const int iqs = iqs_base + ipair * 4;

            const int u_lo = ((const int*) ya_lo->qs)[iqs];
            const int u_hi = ((const int*) ya_hi->qs)[iqs];
            const int summ_lo = flambeau_dp4a_q4k_gu_r4s(0x01010101, u_lo, 0);
            const int summ_hi = flambeau_dp4a_q4k_gu_r4s(0x01010101, u_hi, 0);

            {
                const int gs = ((const int*) gbk->qs)[pair_idx * 8 + iqs];
                const int q_lo = gs & 0x0F0F0F0F;
                const int q_hi = (gs >> 4) & 0x0F0F0F0F;
                const int sumi_lo = flambeau_dp4a_q4k_gu_r4s(q_lo, u_lo, 0);
                const int sumi_hi = flambeau_dp4a_q4k_gu_r4s(q_hi, u_hi, 0);
                acc_gate += d_y_lo *
                    (g_d * (float) sc_lo_g * (float) sumi_lo - g_dmin * (float) m_lo_g * (float) summ_lo);
                acc_gate += d_y_hi *
                    (g_d * (float) sc_hi_g * (float) sumi_hi - g_dmin * (float) m_hi_g * (float) summ_hi);
            }
            {
                const int us = ((const int*) ubk->qs)[pair_idx * 8 + iqs];
                const int q_lo = us & 0x0F0F0F0F;
                const int q_hi = (us >> 4) & 0x0F0F0F0F;
                const int sumi_lo = flambeau_dp4a_q4k_gu_r4s(q_lo, u_lo, 0);
                const int sumi_hi = flambeau_dp4a_q4k_gu_r4s(q_hi, u_hi, 0);
                acc_up += d_y_lo *
                    (u_d * (float) sc_lo_u * (float) sumi_lo - u_dmin * (float) m_lo_u * (float) summ_lo);
                acc_up += d_y_hi *
                    (u_d * (float) sc_hi_u * (float) sumi_hi - u_dmin * (float) m_hi_u * (float) summ_hi);
            }
        }
    }

    acc_gate = gfx906_quarter_warp_reduce_sum(acc_gate);
    acc_up   = gfx906_quarter_warp_reduce_sum(acc_up);

    if (lane_lo == 0) {
        const size_t out_idx =
            ((size_t) token * top_k + slot_idx) * n_rows + row;
        gate_out[out_idx] = acc_gate;
        up_out[out_idx]   = acc_up;
    }
}
