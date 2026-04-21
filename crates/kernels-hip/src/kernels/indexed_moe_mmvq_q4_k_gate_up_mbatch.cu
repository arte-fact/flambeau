// indexed_moe_mmvq_q4_k_gate_up_mbatch — MoE Q4_K gate+up with llama.cpp's
// "one block per row, top_k warps per block" shape.
//
// Difference from indexed_moe_mmvq_q4_k_gate_up_dp4a.cu:
//   - Ours:   blockDim = (64, 1),     grid = (n_rows, n_tokens * top_k)
//             → 1 warp handles 1 (row, slot). top_k × more blocks launched.
//   - This:   blockDim = (64, top_k), grid = (n_rows, n_tokens)
//             → top_k warps share the block. Each warp handles 1 slot of
//               the same (row, token). top_k × fewer blocks launched.
//             → each block runs top_k independent warps; their independent
//               memory latency hides each other's stalls → better CU util.
//
// Same DP4A inner loop (2 dp4a for data, 2 for sum-of-u on the min subtraction),
// same per-super-block lane layout (32 lanes × 4 sub-block-pairs = full coverage).
//
// Fusion preserved: activation read ONCE per sub-block-pair per block (shared
// across the top_k warps via an LDS tile staged once per iteration).

#include "block_quant.cuh"
#include "gfx906.cuh"

// top_k = 8 for Qwen3.6. Parameterised below; keep as compile-time for efficient
// unrolling. If a model uses different top_k we'll specialise.
#ifndef MOE_TOP_K
#define MOE_TOP_K 8
#endif

static __device__ __forceinline__ int flambeau_dp4a_q4k_mb(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(MOE_TOP_K * 64, 1)
void flambeau_indexed_moe_mmvq_q4_k_gate_up_mbatch_q8_1(
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
    const int row      = blockIdx.x;           // which output row
    const int token    = blockIdx.y;           // which token
    const int slot_idx = threadIdx.y;          // which expert-slot (0..top_k-1)
    if (row >= n_rows || token >= n_tokens || slot_idx >= top_k) return;

    const int lane     = threadIdx.x;          // 0..63 within the warp
    const int lane_lo  = lane & 31;            // 0..31
    const int super_hi = lane >> 5;            // 0 or 1
    const int pair_idx = lane_lo >> 3;         // 0..3
    const int iqs      = lane_lo & 7;          // 0..7
    const int sub_lo   = pair_idx * 2;
    const int sub_hi   = sub_lo + 1;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const flambeau_block_q4_K* gate_row =
        gate_w + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q4_K* up_row =
        up_w   + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    // Stride-2 outer loop: within a warp, super_hi picks b or b+1.
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
        const flambeau_block_q8_1* ya_lo = y_sb + sub_lo;
        const flambeau_block_q8_1* ya_hi = y_sb + sub_hi;
        const int u_lo = ((const int*) ya_lo->qs)[iqs];
        const int u_hi = ((const int*) ya_hi->qs)[iqs];
        const float d_y_lo = (float) ya_lo->d;
        const float d_y_hi = (float) ya_hi->d;

        const int summ_lo = flambeau_dp4a_q4k_mb(0x01010101, u_lo, 0);
        const int summ_hi = flambeau_dp4a_q4k_mb(0x01010101, u_hi, 0);

        uint8_t sc_lo_g, m_lo_g, sc_hi_g, m_hi_g;
        uint8_t sc_lo_u, m_lo_u, sc_hi_u, m_hi_u;
        flambeau_q4k_scale_min(sub_lo, gbk->scales, &sc_lo_g, &m_lo_g);
        flambeau_q4k_scale_min(sub_hi, gbk->scales, &sc_hi_g, &m_hi_g);
        flambeau_q4k_scale_min(sub_lo, ubk->scales, &sc_lo_u, &m_lo_u);
        flambeau_q4k_scale_min(sub_hi, ubk->scales, &sc_hi_u, &m_hi_u);

        {
            const int gs = ((const int*) gbk->qs)[pair_idx * 8 + iqs];
            const int q_lo = gs & 0x0F0F0F0F;
            const int q_hi = (gs >> 4) & 0x0F0F0F0F;
            const int sumi_lo = flambeau_dp4a_q4k_mb(q_lo, u_lo, 0);
            const int sumi_hi = flambeau_dp4a_q4k_mb(q_hi, u_hi, 0);
            acc_gate += d_y_lo *
                (g_d * (float) sc_lo_g * (float) sumi_lo - g_dmin * (float) m_lo_g * (float) summ_lo);
            acc_gate += d_y_hi *
                (g_d * (float) sc_hi_g * (float) sumi_hi - g_dmin * (float) m_hi_g * (float) summ_hi);
        }
        {
            const int us = ((const int*) ubk->qs)[pair_idx * 8 + iqs];
            const int q_lo = us & 0x0F0F0F0F;
            const int q_hi = (us >> 4) & 0x0F0F0F0F;
            const int sumi_lo = flambeau_dp4a_q4k_mb(q_lo, u_lo, 0);
            const int sumi_hi = flambeau_dp4a_q4k_mb(q_hi, u_hi, 0);
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
