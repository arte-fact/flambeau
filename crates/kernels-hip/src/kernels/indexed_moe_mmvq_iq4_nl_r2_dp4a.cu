// indexed_moe_mmvq_iq4_nl_r2_dp4a — IQ4_NL MoE MMVQ with DP4A inner loop.
// Drop-in replacement for `indexed_moe_mmvq_iq4_nl.cu` (scalar FP32). Same
// per-element math as dense `mmvq_iq4_nl_dp4a.cu` (32-element plain quant,
// 16-entry kvalues_iq4nl codebook resolved via gfx906 __builtin_amdgcn_perm)
// plus the standard MoE wiring.
//
// Launch: blockDim = { 64 } (wave64), gridDim = { n_row_pairs,
// n_tokens * top_k, 1 }. 2 rows per block; 32 lanes per row, 4 lanes per
// IQ4_NL block (one int32 of qs = 8 nibbles) → 8 blocks parallel per
// row per outer-iter.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"

#define MOE_IQ4_NL_LANES_PER_BLOCK 4
#define MOE_IQ4_NL_BLOCKS_PER_ITER 8

static __device__ __forceinline__ int flambeau_indexed_moe_iq4_nl_dp4a(
    int a, int b, int c
) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ uint32_t flambeau_indexed_moe_iq4_nl_table_4(
    uint32_t q4_nibbles
) {
    constexpr uint32_t kv0 = 0xBFAD9881u;
    constexpr uint32_t kv1 = 0xF6EADDCFu;
    constexpr uint32_t kv2 = 0x26190D01u;
    constexpr uint32_t kv3 = 0x71594535u;

    const uint32_t v_low  = __builtin_amdgcn_perm(kv1, kv0, q4_nibbles & 0x07070707);
    const uint32_t v_high = __builtin_amdgcn_perm(kv3, kv2, q4_nibbles & 0x07070707);
    const uint32_t mask   = 0x03020100u | ((q4_nibbles & 0x08080808u) >> 1);
    return __builtin_amdgcn_perm(v_high, v_low, mask);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_iq4_nl_r2_dp4a_q8_1(
    const flambeau_block_iq4_nl* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    float* __restrict__ dst,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_blocks_per_row
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

    const flambeau_block_iq4_nl* xrow =
        x + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_blocks_per_row;

    const int lane4     = lane_lo & 3;
    const int block_idx = lane_lo >> 2;

    float acc = 0.0f;

    for (int b = block_idx; b < n_blocks_per_row; b += MOE_IQ4_NL_BLOCKS_PER_ITER) {
        const flambeau_block_iq4_nl* bx = xrow + b;
        const flambeau_block_q8_1*   by = y_row + b;

        const int aux_q4 = ((const int*) bx->qs)[lane4];
        const uint32_t q_low  = (uint32_t) aux_q4 & 0x0F0F0F0Fu;
        const uint32_t q_high = ((uint32_t) aux_q4 >> 4) & 0x0F0F0F0Fu;

        const uint32_t v_lo = flambeau_indexed_moe_iq4_nl_table_4(q_low);
        const uint32_t v_hi = flambeau_indexed_moe_iq4_nl_table_4(q_high);

        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];

        int sumi = flambeau_indexed_moe_iq4_nl_dp4a((int) v_lo, u_lo, 0);
        sumi     = flambeau_indexed_moe_iq4_nl_dp4a((int) v_hi, u_hi, sumi);

        const float d_x = (float) bx->d;
        const float d_y = (float) by->d;
        acc += d_x * d_y * (float) sumi;
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
