#pragma once
// Templated IQ4_NL indexed-MoE MMVQ, r2 (2 rows/block, 32-lane groups). IQ4_NL
// is a 32-element block (fp16 `d` + 16 nibble bytes) whose 4-bit indices address
// the 16-entry non-linear codebook; reuses the dense `iq4_nl_table_4` LUT
// (neutral `byte_perm`). 4 lanes per block (one int32 = 8 nibbles), 8 blocks in
// flight per 32-lane row group. MoE addressing: weight row by
// `expert_ids[token,slot]`, per-token activation `y[token,:]`, output
// `dst[token,slot,row]`. Backend-neutral: `byte_perm`/`dp4a`/the group reduce
// from the arch header + shared helpers; int loads through `load_int_b2`.
#include "mmvq_iq4_nl.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_iq4_nl_r2_dp4a_row(
    const flambeau_block_iq4_nl* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int top_k, int n_blocks_per_row)
{
    constexpr int BLOCKS_PER_ITER = 8;     // 32 lanes / 4 lanes-per-block

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
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_blocks_per_row;

    const int lane4     = lane_lo & 3;
    const int block_idx = lane_lo >> 2;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += BLOCKS_PER_ITER) {
        const flambeau_block_iq4_nl* bx = xrow + b;
        const flambeau_block_q8_1*   by = y_row + b;

        const int aux_q4 = load_int_b2((const int8_t*) bx->qs, lane4);
        const unsigned int q_low  = (unsigned int) aux_q4 & 0x0F0F0F0Fu;
        const unsigned int q_high = ((unsigned int) aux_q4 >> 4) & 0x0F0F0F0Fu;

        const unsigned int v_lo = iq4_nl_table_4(q_low);
        const unsigned int v_hi = iq4_nl_table_4(q_high);

        const int u_lo = load_int_b2(by->qs, lane4);
        const int u_hi = load_int_b2(by->qs, lane4 + 4);

        int sumi = dp4a((int) v_lo, u_lo, 0);
        sumi     = dp4a((int) v_hi, u_hi, sumi);

        acc += (float) bx->d * (float) by->d * (float) sumi;
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_group<32, OutT>(acc, dst, dst_idx, lane_lo);
}
