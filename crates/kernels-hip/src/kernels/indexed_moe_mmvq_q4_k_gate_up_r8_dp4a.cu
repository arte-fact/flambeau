// indexed_moe_mmvq_q4_k_gate_up_r8_dp4a — MoE Q4_K fused gate+up, 8 rows per block.
// r8 extension of r4. Halves block count once more:
// - r1 baseline: grid = (n_rows, n_tokens * top_k)
// - r2: grid = (n_rows/2, n_tokens * top_k)
// - r4: grid = (n_rows/4, n_tokens * top_k)
// - r8: grid = (n_rows/8, n_tokens * top_k)
// Lane layout (wave64, 8 rows simultaneously):
// row_idx = lane >> 3 — 0..7: which row within the octet
// lane_lo = lane & 7 — 0..7: eighth-warp position
// Each 8-lane eighth-warp handles 1 row. Q4_K super-block has 32 int32s of
// qs (128 bytes); 8 lanes cover them with 4 ints/lane per super-block →
// inner loop does 4 dp4a chains per lane (vs 2 for r4, 1 for r2).
// Reduce via `gfx906_eighth_warp_reduce_sum` (xor-1, xor-2, xor-4 DPP
// sequence within a 16-lane bank — no cross-bank shuffle needed).
// Trade-off: 2× per-lane work vs r4, but 2× fewer blocks. Net gain
// depends on whether launch overhead still dominates per-block cost at
// r4 scale. /b measured r1→r2 (+6 %), r2→r4 (+12 %); r4→r8 is the
// "is launch overhead still the bottleneck?" test.

#include "block_quant.cuh"
#include "gfx906.cuh"

static __device__ __forceinline__ int flambeau_dp4a_q4k_gu_r8(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q4_k_gate_up_r8_dp4a_q8_1(
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
    const int row_oct = blockIdx.x;
    const int slot    = blockIdx.y;
    const int token   = slot / top_k;
    const int slot_idx = slot - token * top_k;

    if (token >= n_tokens) return;

    const int lane    = threadIdx.x;               // 0..63
    const int row_idx = lane >> 3;                 // 0..7
    const int lane_lo = lane & 7;                  // 0..7

    const int row = row_oct * 8 + row_idx;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const flambeau_block_q4_K* gate_row =
        gate_w + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q4_K* up_row =
        up_w   + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    // 8 lanes cover 32 int32s of qs → 4 ints/lane/super-block.
    // lane_lo ∈ [0,8): pair_idx picks sub-block pair (0..3), iqs_base picks
    // which two iqs positions (0 and 4) within the pair's 8-int slice.
    // Each lane iterates 4 iqs positions: iqs_base, iqs_base+2, iqs_base+4, iqs_base+6
    // ... wait, lane_lo=0..7 and pair_idx should be 0..3. Use pair_idx = lane_lo >> 1,
    // iqs_base = lane_lo & 1. Each lane handles iqs in {iqs_base, iqs_base+2,
    // iqs_base+4, iqs_base+6} (4 ints).
    const int pair_idx = lane_lo >> 1;             // 0..3
    const int iqs_base = lane_lo & 1;              // 0..1
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
        const float d_y_lo = (float) ya_lo->d;
        const float d_y_hi = (float) ya_hi->d;

        uint8_t sc_lo_g = 0, m_lo_g = 0, sc_hi_g = 0, m_hi_g = 0;
        uint8_t sc_lo_u = 0, m_lo_u = 0, sc_hi_u = 0, m_hi_u = 0;
        flambeau_q4k_scale_min(sub_lo, gbk->scales, &sc_lo_g, &m_lo_g);
        flambeau_q4k_scale_min(sub_hi, gbk->scales, &sc_hi_g, &m_hi_g);
        flambeau_q4k_scale_min(sub_lo, ubk->scales, &sc_lo_u, &m_lo_u);
        flambeau_q4k_scale_min(sub_hi, ubk->scales, &sc_hi_u, &m_hi_u);

        // Each lane covers 4 iqs positions: iqs_base, iqs_base+2, +4, +6.
        #pragma unroll
        for (int iquad = 0; iquad < 4; ++iquad) {
            const int iqs = iqs_base + iquad * 2;

            const int u_lo = ((const int*) ya_lo->qs)[iqs];
            const int u_hi = ((const int*) ya_hi->qs)[iqs];
            const int summ_lo = flambeau_dp4a_q4k_gu_r8(0x01010101, u_lo, 0);
            const int summ_hi = flambeau_dp4a_q4k_gu_r8(0x01010101, u_hi, 0);

            {
                const int gs = ((const int*) gbk->qs)[pair_idx * 8 + iqs];
                const int q_lo = gs & 0x0F0F0F0F;
                const int q_hi = (gs >> 4) & 0x0F0F0F0F;
                const int sumi_lo = flambeau_dp4a_q4k_gu_r8(q_lo, u_lo, 0);
                const int sumi_hi = flambeau_dp4a_q4k_gu_r8(q_hi, u_hi, 0);
                acc_gate += d_y_lo *
                    (g_d * (float) sc_lo_g * (float) sumi_lo - g_dmin * (float) m_lo_g * (float) summ_lo);
                acc_gate += d_y_hi *
                    (g_d * (float) sc_hi_g * (float) sumi_hi - g_dmin * (float) m_hi_g * (float) summ_hi);
            }
            {
                const int us = ((const int*) ubk->qs)[pair_idx * 8 + iqs];
                const int q_lo = us & 0x0F0F0F0F;
                const int q_hi = (us >> 4) & 0x0F0F0F0F;
                const int sumi_lo = flambeau_dp4a_q4k_gu_r8(q_lo, u_lo, 0);
                const int sumi_hi = flambeau_dp4a_q4k_gu_r8(q_hi, u_hi, 0);
                acc_up += d_y_lo *
                    (u_d * (float) sc_lo_u * (float) sumi_lo - u_dmin * (float) m_lo_u * (float) summ_lo);
                acc_up += d_y_hi *
                    (u_d * (float) sc_hi_u * (float) sumi_hi - u_dmin * (float) m_hi_u * (float) summ_hi);
            }
        }
    }

    acc_gate = gfx906_eighth_warp_reduce_sum(acc_gate);
    acc_up   = gfx906_eighth_warp_reduce_sum(acc_up);

    if (lane_lo == 0) {
        const size_t out_idx =
            ((size_t) token * top_k + slot_idx) * n_rows + row;
        gate_out[out_idx] = acc_gate;
        up_out[out_idx]   = acc_up;
    }
}
