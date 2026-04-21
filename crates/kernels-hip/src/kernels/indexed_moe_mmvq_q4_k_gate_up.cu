// indexed_moe_mmvq_q4_k_gate_up — fused gate+up Q4_K MoE MMVQ (candle P30).
//
// In the MoE FFN the gate and up projections share the same activation
// `y[token]`. A naive implementation launches two separate
// `indexed_moe_mmvq_q4_k` kernels — one for gate, one for up — which
// reads the activation from HBM twice. Candle's P30 fuses them into a
// single kernel that reads the activation once and computes both
// matmuls in the same inner loop, halving activation HBM traffic.
//
// Layout (same as the non-fused kernel, doubled on the weight side):
//   gate_w    [n_experts, n_rows, n_sb_per_row]   Q4_K
//   up_w      [n_experts, n_rows, n_sb_per_row]   Q4_K
//   y         [n_tokens, n_sb_per_row * 8]        Q8_1
//   expert_id [n_tokens, top_k]                   i32
//   gate_out  [n_tokens, top_k, n_rows]           F32
//   up_out    [n_tokens, top_k, n_rows]           F32
//
// Launch + inner arithmetic identical to `indexed_moe_mmvq_q4_k.cu`, just
// carries two accumulators through the inner loop. Register usage grows
// modestly (extra accumulator + pointer) but stays well clear of the
// 256-VGPR wave-ceiling.

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_indexed_moe_mmvq_q4_k_gate_up_q8_1(
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

    const int lane     = threadIdx.x;
    const int byte_off = lane & 31;
    const int hi_half  = lane >> 5;

    const flambeau_block_q4_K* gate_row =
        gate_w + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q4_K* up_row =
        up_w   + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

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

        #pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const int sub = 2 * grp + hi_half;

            // Activation read — ONCE per (super-block, sub-block) — this is
            // the optimisation P30 buys: the non-fused kernel does this
            // read twice (once in the gate pass, once in the up pass).
            const flambeau_block_q8_1* ya = y_sb + sub;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[byte_off];
            const float y_val = d_y * (float) qi;

            // Gate weight read + dequant.
            uint8_t g_sc = 0, g_m = 0;
            flambeau_q4k_scale_min(sub, gbk->scales, &g_sc, &g_m);
            const int g_byte = (int) gbk->qs[grp * 32 + byte_off];
            const int g_raw  = hi_half ? (g_byte >> 4) : (g_byte & 0x0F);
            const float g_x =
                g_d * (float) g_sc * (float) g_raw - g_dmin * (float) g_m;
            acc_gate += g_x * y_val;

            // Up weight read + dequant, SAME activation value.
            uint8_t u_sc = 0, u_m = 0;
            flambeau_q4k_scale_min(sub, ubk->scales, &u_sc, &u_m);
            const int u_byte = (int) ubk->qs[grp * 32 + byte_off];
            const int u_raw  = hi_half ? (u_byte >> 4) : (u_byte & 0x0F);
            const float u_x =
                u_d * (float) u_sc * (float) u_raw - u_dmin * (float) u_m;
            acc_up += u_x * y_val;
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
