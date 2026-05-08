// indexed_moe_mmvq_q4_k_gate_up_r2_dp4a — MoE Q4_K fused gate+up, 2 rows per block.
// Same output layout as `indexed_moe_mmvq_q4_k_gate_up_dp4a.cu` but block/grid
// shape halves block count in the row dimension:
// - Original: grid = (n_rows, n_tokens * top_k), block = (64, 1, 1)
// - This: grid = (n_rows / 2, n_tokens * top_k), block = (64, 1, 1)
// Why: at prefill L=512, n_slots = n_tokens * top_k = 4096 and n_rows ≈ 2816
// → ~11.5M blocks per call. Each block produces 1 output using a full wave64
// → tiny work per block, launch-overhead-dominated at 11.46 ms/call (profile).
// Adaptation from `indexed_moe_mmvq_q4_k_r2_dp4a.cu`'s r2 pattern (used for
// the down projection) but with the `gate_up_dp4a` kernel's fusion semantics:
// one activation load per (slot, super-block) shared between gate and up.
// Lane layout (wave64, handles 2 rows simultaneously):
// row_hi = lane >> 5 — 0 or 1: selects row_pair*2 or row_pair*2+1
// lane_lo = lane & 31 — 0..31: half-warp position within a row
// pair_idx = lane_lo >> 3 — 0..3: which sub-block pair
// iqs = lane_lo & 7 — 0..7: which int32 of the pair's slice
// Each half-warp (32 lanes) sequentially iterates super-blocks for ITS row.
// Both halves iterate the same super-block index in lockstep → activation
// reads happen at the same iteration → shared HBM fetches via L1 cache
// (gfx906 has no cooperative shared LDS load here, but L1 naturally coalesces
// because both halves read the same y address).

#include "block_quant.cuh"
#include "gfx906.cuh"

static __device__ __forceinline__ int flambeau_dp4a_q4k_gu_r2(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q4_k_gate_up_r2_dp4a_q8_1(
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
    const int row_pair = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;

    if (token >= n_tokens) return;

    const int lane    = threadIdx.x;              // 0..63
    const int row_hi  = lane >> 5;                // 0 or 1 — which row in the pair
    const int lane_lo = lane & 31;                // 0..31 — half-warp position

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const flambeau_block_q4_K* gate_row =
        gate_w + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q4_K* up_row =
        up_w   + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    const int pair_idx = lane_lo >> 3;             // 0..3
    const int iqs      = lane_lo & 7;              // 0..7
    const int sub_lo   = pair_idx * 2;             // 0, 2, 4, 6
    const int sub_hi   = sub_lo + 1;               // 1, 3, 5, 7

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

        // Activation: shared between gate and up for this super-block.
        // Both row halves read the same address → L1 coalesces the fetch.
        const int u_lo = ((const int*) ya_lo->qs)[iqs];
        const int u_hi = ((const int*) ya_hi->qs)[iqs];
        const float d_y_lo = (float) ya_lo->d;
        const float d_y_hi = (float) ya_hi->d;

        // sum_u: constant-part for dmin subtraction, shared across gate and up.
        const int summ_lo = flambeau_dp4a_q4k_gu_r2(0x01010101, u_lo, 0);
        const int summ_hi = flambeau_dp4a_q4k_gu_r2(0x01010101, u_hi, 0);

        uint8_t sc_lo_g = 0, m_lo_g = 0, sc_hi_g = 0, m_hi_g = 0;
        uint8_t sc_lo_u = 0, m_lo_u = 0, sc_hi_u = 0, m_hi_u = 0;
        flambeau_q4k_scale_min(sub_lo, gbk->scales, &sc_lo_g, &m_lo_g);
        flambeau_q4k_scale_min(sub_hi, gbk->scales, &sc_hi_g, &m_hi_g);
        flambeau_q4k_scale_min(sub_lo, ubk->scales, &sc_lo_u, &m_lo_u);
        flambeau_q4k_scale_min(sub_hi, ubk->scales, &sc_hi_u, &m_hi_u);

        // Gate qs int32: 4 low nibbles → sub_lo, 4 high nibbles → sub_hi.
        {
            const int gs = ((const int*) gbk->qs)[pair_idx * 8 + iqs];
            const int q_lo = gs & 0x0F0F0F0F;
            const int q_hi = (gs >> 4) & 0x0F0F0F0F;
            const int sumi_lo = flambeau_dp4a_q4k_gu_r2(q_lo, u_lo, 0);
            const int sumi_hi = flambeau_dp4a_q4k_gu_r2(q_hi, u_hi, 0);
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
            const int sumi_lo = flambeau_dp4a_q4k_gu_r2(q_lo, u_lo, 0);
            const int sumi_hi = flambeau_dp4a_q4k_gu_r2(q_hi, u_hi, 0);
            acc_up += d_y_lo *
                (u_d * (float) sc_lo_u * (float) sumi_lo - u_dmin * (float) m_lo_u * (float) summ_lo);
            acc_up += d_y_hi *
                (u_d * (float) sc_hi_u * (float) sumi_hi - u_dmin * (float) m_hi_u * (float) summ_hi);
        }
    }

    // Per-row half-warp reduce (32 lanes of each row_hi half).
    acc_gate = gfx906_half_warp_reduce_sum(acc_gate);
    acc_up   = gfx906_half_warp_reduce_sum(acc_up);

    if (lane_lo == 0) {
        const size_t out_idx =
            ((size_t) token * top_k + slot_idx) * n_rows + row;
        gate_out[out_idx] = acc_gate;
        up_out[out_idx]   = acc_up;
    }
}
