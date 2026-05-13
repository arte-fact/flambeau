// indexed_moe_mmvq_iq4_nl — IQ4_NL MMVQ with per-token expert routing.
// IQ4_NL = 32-elem block, no super-block. n_sb_per_row carries the
// per-row block count (k / 32). Phase 4 Slice B.

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "iq_grid.cuh"

extern "C" __global__ void flambeau_indexed_moe_mmvq_iq4_nl_q8_1(
    const flambeau_block_iq4_nl* __restrict__ x,
    const flambeau_block_q8_1*   __restrict__ y,
    const int* __restrict__ expert_ids,
    float* __restrict__ dst,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_blocks_per_row
) {
    const int row       = blockIdx.x;
    const int slot      = blockIdx.y;
    const int token     = slot / top_k;
    const int slot_idx  = slot - token * top_k;
    if (row >= n_rows || token >= n_tokens) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const int lane      = threadIdx.x;
    const int lane_lo   = lane & 31;
    const int byte_off  = lane_lo & 15;
    const int nibble_hi = lane_lo >> 4;
    const int block_hi  = lane >> 5;
    const int elem_in_block = byte_off + nibble_hi * 16;

    const flambeau_block_iq4_nl* xrow =
        x + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_blocks_per_row;

    float acc = 0.0f;

    for (int b = block_hi; b < n_blocks_per_row; b += 2) {
        const flambeau_block_iq4_nl* bk = xrow + b;
        const flambeau_block_q8_1*   by = y_row + b;

        const float d_x = (float) bk->d;
        const int byte_v = (int) bk->qs[byte_off];
        const int code = (byte_v >> (nibble_hi * 4)) & 0x0F;
        const float x_val = d_x * (float) flambeau_iq4nl_lut(code);

        const float d_y = (float) by->d;
        const int qi = (int) by->qs[elem_in_block];
        acc += x_val * d_y * (float) qi;
    }

    acc = gfx906_warp_reduce_sum(acc);
    if (lane == 0) {
        dst[(size_t)(token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
