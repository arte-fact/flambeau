// indexed_moe_mmvq_q4_k_gate_up_dp4a — fused gate+up Q4_K MoE MMVQ, DP4A.
// Drop-in replacement for `indexed_moe_mmvq_q4_k_gate_up.cu`. Same block/grid
// shape (64 threads = 1 wave64, one output row per block), same output layout.
// Fusion: reads the shared activation ONCE per super-block-per-sub-block and
// uses it for both gate and up matmuls (P30 pattern).
// DP4A inner loop: same pattern as `indexed_moe_mmvq_q4_k_r2_dp4a.cu` but
// scaled to 64 lanes full wave coverage. 32 lanes handle 1 super-block's
// worth of int32s (128 bytes = 32 × 4). With 64 lanes, outer loop strides
// super-blocks by 2 so both halves contribute. Per lane:
// - ONE qs int32 load per super-block (own position)
// - ONE Q8_1 int32 load per sub-block (shared across gate+up)
// - FOUR dp4a calls: 2 for gate (raw_q×y + sum_u), 2 for up
// Lane layout (within wave64):
// super_hi = lane >> 5 — 0 or 1: which of the two super-blocks in the stride-2 iter
// lane_lo = lane & 31 — 0..31: position within a super-block's 32 int32s
// pair_idx = lane_lo >> 3 — 0..3: which pair of sub-blocks
// iqs = lane_lo & 7 — 0..7: which int32 of the pair's slice

#include "block_quant.cuh"
#include "gfx906.cuh"

static __device__ __forceinline__ int flambeau_dp4a_q4k_gu(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Tried __launch_bounds__(64, 8) — forced VGPR 38→32 but introduced 4 spills;
// measured -5.7% decode (53.7 → 50.6 tok/s). gfx906 has no AGPR backup so
// VGPR spills go straight to scratch/HBM — any gain from 6→8 waves/SIMD was
// dwarfed. Kept at default (no explicit launch_bounds) which yields VGPR=38,
// 6 waves/SIMD, zero spills.
extern "C" __global__ void flambeau_indexed_moe_mmvq_q4_k_gate_up_dp4a_q8_1(
    const flambeau_block_q4_K* __restrict__ gate_w,
    const flambeau_block_q4_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row
) {
    const int row      = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;

    if (row >= n_rows || token >= n_tokens) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const int lane     = threadIdx.x;              // 0..63
    const int super_hi = lane >> 5;                // 0 or 1 — which super-block in stride-2 pair
    const int lane_lo  = lane & 31;                // 0..31
    const int pair_idx = lane_lo >> 3;             // 0..3
    const int iqs      = lane_lo & 7;              // 0..7
    const int sub_lo   = pair_idx * 2;             // 0, 2, 4, 6
    const int sub_hi   = sub_lo + 1;               // 1, 3, 5, 7

    const flambeau_block_q4_K* gate_row =
        gate_w + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q4_K* up_row =
        up_w   + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    // Stride-2 outer loop: lanes 0-31 handle super-block b, lanes 32-63 handle b+1.
    for (int b_base = 0; b_base < n_sb_per_row; b_base += 2) {
        const int b = b_base + super_hi;
        if (b >= n_sb_per_row) continue;

        const flambeau_block_q4_K* gbk = gate_row + b;
        const flambeau_block_q4_K* ubk = up_row + b;
        const float g_d    = (float) gbk->d;
        const float g_dmin = (float) gbk->dmin;
        const float u_d    = (float) ubk->d;
        const float u_dmin = (float) ubk->dmin;

        const flambeau_block_q8_1* y_sb = y_row + b * 8;

        // Load shared activation int32s once — one per sub-block of the pair.
        const flambeau_block_q8_1* ya_lo = y_sb + sub_lo;
        const flambeau_block_q8_1* ya_hi = y_sb + sub_hi;
        const int u_lo = ((const int*) ya_lo->qs)[iqs];
        const int u_hi = ((const int*) ya_hi->qs)[iqs];
        const float d_y_lo = (float) ya_lo->d;
        const float d_y_hi = (float) ya_hi->d;

        // sum_u: constant-part for dmin subtraction, shared across gate and up.
        const int summ_lo = flambeau_dp4a_q4k_gu(0x01010101, u_lo, 0);
        const int summ_hi = flambeau_dp4a_q4k_gu(0x01010101, u_hi, 0);

        uint8_t sc_lo_g = 0, m_lo_g = 0, sc_hi_g = 0, m_hi_g = 0;
        uint8_t sc_lo_u = 0, m_lo_u = 0, sc_hi_u = 0, m_hi_u = 0;
        flambeau_q4k_scale_min(sub_lo, gbk->scales, &sc_lo_g, &m_lo_g);
        flambeau_q4k_scale_min(sub_hi, gbk->scales, &sc_hi_g, &m_hi_g);
        flambeau_q4k_scale_min(sub_lo, ubk->scales, &sc_lo_u, &m_lo_u);
        flambeau_q4k_scale_min(sub_hi, ubk->scales, &sc_hi_u, &m_hi_u);

        // Gate: int32 of qs, split low/high nibbles, dp4a both.
        {
            const int gs = ((const int*) gbk->qs)[pair_idx * 8 + iqs];
            const int q_lo = gs & 0x0F0F0F0F;
            const int q_hi = (gs >> 4) & 0x0F0F0F0F;
            const int sumi_lo = flambeau_dp4a_q4k_gu(q_lo, u_lo, 0);
            const int sumi_hi = flambeau_dp4a_q4k_gu(q_hi, u_hi, 0);
            acc_gate += d_y_lo *
                (g_d * (float) sc_lo_g * (float) sumi_lo - g_dmin * (float) m_lo_g * (float) summ_lo);
            acc_gate += d_y_hi *
                (g_d * (float) sc_hi_g * (float) sumi_hi - g_dmin * (float) m_hi_g * (float) summ_hi);
        }
        // Up: same pattern, different weights, SAME activation.
        {
            const int us = ((const int*) ubk->qs)[pair_idx * 8 + iqs];
            const int q_lo = us & 0x0F0F0F0F;
            const int q_hi = (us >> 4) & 0x0F0F0F0F;
            const int sumi_lo = flambeau_dp4a_q4k_gu(q_lo, u_lo, 0);
            const int sumi_hi = flambeau_dp4a_q4k_gu(q_hi, u_hi, 0);
            acc_up += d_y_lo *
                (u_d * (float) sc_lo_u * (float) sumi_lo - u_dmin * (float) m_lo_u * (float) summ_lo);
            acc_up += d_y_hi *
                (u_d * (float) sc_hi_u * (float) sumi_hi - u_dmin * (float) m_hi_u * (float) summ_hi);
        }
    }

    acc_gate = gfx906_warp_reduce_sum(acc_gate);
    acc_up   = gfx906_warp_reduce_sum(acc_up);

    if (lane == 0) {
        const size_t out_idx =
            ((size_t) token * top_k + slot_idx) * n_rows + row;
        gate_out[out_idx] = acc_gate;
        up_out[out_idx]   = acc_up;
    }
}
