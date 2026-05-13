// indexed_moe_mmvq_q2_k — Q2_K MMVQ with per-token expert routing.
// Single-row MMVQ pattern (64-thread wave64, 1 row/block) extended for MoE.
// Decode logic identical to mmvq_q2_k.cu.

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_indexed_moe_mmvq_q2_K_q8_1(
    const flambeau_block_q2_K* __restrict__ x,     // [n_experts, n_rows, n_sb]
    const flambeau_block_q8_1* __restrict__ y,     // [n_tokens, n_sb * 8]
    const int* __restrict__ expert_ids,            // [n_tokens, top_k]
    float* __restrict__ dst,                       // [n_tokens, top_k, n_rows]
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

    const int lane = threadIdx.x;

    const flambeau_block_q2_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    float acc = 0.0f;

    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q2_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        const flambeau_block_q8_1* y_sb = y_row + b * 8;

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int y_idx      = lane + k * 64;
            const int chunk_idx  = y_idx >> 7;
            const int within_128 = y_idx & 127;
            const int shift_iter = within_128 >> 5;
            const int within_32  = within_128 & 31;
            const int scale_idx  = within_32 >> 4;
            const int l          = within_32 & 15;
            const int qi         = l + 16 * scale_idx;
            const int is         = chunk_idx * 8 + 2 * shift_iter + scale_idx;
            const int shift      = 2 * shift_iter;

            const int qs_byte = bk->qs[chunk_idx * 32 + qi];
            const int q       = (qs_byte >> shift) & 3;
            const int sc_byte = bk->scales[is];
            const int sc      = sc_byte & 0xF;
            const int mn      = sc_byte >> 4;

            const float x_val = d * (float) sc * (float) q - dmin * (float) mn;

            const int y_block = y_idx >> 5;
            const int y_off   = y_idx & 31;
            const flambeau_block_q8_1* ya = y_sb + y_block;
            const float d_y = (float) ya->d;
            const int   qi8 = (int) ya->qs[y_off];
            acc += x_val * d_y * (float) qi8;
        }
    }

    acc = gfx906_warp_reduce_sum(acc);

    if (lane == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
