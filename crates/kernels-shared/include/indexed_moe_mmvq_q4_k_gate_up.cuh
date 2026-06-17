#pragma once
// Templated fused gate+up Q4_K indexed-MoE MMVQ (scalar-FP32 dequant path). The
// gate and up projections share the per-token activation `y[token]`, read once
// per (super-block, sub-block) and reused for both dots — halving activation HBM
// traffic vs two separate launches. One output row per block over 64 lanes
// (byte_off ∈ 0..31 × hi_half ∈ 0..1 = the 64 elements of a sub-block pair). MoE
// addressing: weight rows by `expert_ids[token,slot]`, output
// `gate_out`/`up_out`[token,slot,row]. Backend-neutral: all byte reads; the
// two-output block reduce is `mmvq_reduce_store_gate_up<64>` (warp + inter-warp
// LDS — correct on wave64 single-warp AND wave32 two-warp). `OutT` is the dest.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_q4_k_gate_up_row(
    const flambeau_block_q4_K* __restrict__ gate_w,
    const flambeau_block_q4_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    OutT* __restrict__ gate_out,
    OutT* __restrict__ up_out,
    int n_rows, int n_tokens, int top_k, int n_sb_per_row)
{
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
        up_w + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_sb_per_row * 8;

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

            const flambeau_block_q8_1* ya = y_sb + sub;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[byte_off];
            const float y_val = d_y * (float) qi;

            uint8_t g_sc = 0, g_m = 0;
            flambeau_q4k_scale_min(sub, gbk->scales, &g_sc, &g_m);
            const int g_byte = (int) gbk->qs[grp * 32 + byte_off];
            const int g_raw  = hi_half ? (g_byte >> 4) : (g_byte & 0x0F);
            const float g_x = g_d * (float) g_sc * (float) g_raw - g_dmin * (float) g_m;
            acc_gate += g_x * y_val;

            uint8_t u_sc = 0, u_m = 0;
            flambeau_q4k_scale_min(sub, ubk->scales, &u_sc, &u_m);
            const int u_byte = (int) ubk->qs[grp * 32 + byte_off];
            const int u_raw  = hi_half ? (u_byte >> 4) : (u_byte & 0x0F);
            const float u_x = u_d * (float) u_sc * (float) u_raw - u_dmin * (float) u_m;
            acc_up += u_x * y_val;
        }
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_gate_up<64, OutT>(acc_gate, acc_up, gate_out, up_out, dst_idx, true, true);
}
