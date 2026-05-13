// indexed_moe_mmvq_iq4_xs — IQ4_XS MMVQ with per-token expert routing.
// Mirrors `indexed_moe_mmvq_q4_k.cu` structurally; inner arithmetic is
// byte-identical to `mmvq_iq4_xs.cu` (single-row scalar variant). Per
// (token, slot) pair: read expert id, fetch its weight slice, compute
// MMVQ row × Q8_1 activation row, write the f32 scalar.
// Phase 4 Slice B.

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "iq_grid.cuh"

extern "C" __global__ void flambeau_indexed_moe_mmvq_iq4_xs_q8_1(
    const flambeau_block_iq4_xs* __restrict__ x,
    const flambeau_block_q8_1*   __restrict__ y,
    const int* __restrict__ expert_ids,
    float* __restrict__ dst,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row
) {
    const int row       = blockIdx.x;
    const int slot      = blockIdx.y;
    const int token     = slot / top_k;
    const int slot_idx  = slot - token * top_k;
    if (row >= n_rows || token >= n_tokens) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const int lane      = threadIdx.x;
    const int sub_hi    = lane >> 5;
    const int lane_lo   = lane & 31;
    const int byte_off  = lane_lo & 15;
    const int nibble_hi = lane_lo >> 4;
    const int elem_in_sub = byte_off + nibble_hi * 16;

    const flambeau_block_iq4_xs* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    float acc = 0.0f;

    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_iq4_xs* bk = xrow + b;
        const float    d        = (float) bk->d;
        const uint16_t scales_h = bk->scales_h;
        const flambeau_block_q8_1* y_sb = y_row + (size_t) b * 8;

        #pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const int sub = 2 * grp + sub_hi;
            const int ls  = flambeau_iq4_xs_scale(sub, scales_h, bk->scales_l);

            const int byte_v = (int) bk->qs[sub * 16 + byte_off];
            const int code   = (byte_v >> (nibble_hi * 4)) & 0x0F;
            const float x_val = d * (float) ls * (float) flambeau_iq4nl_lut(code);

            const flambeau_block_q8_1* ya = y_sb + sub;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[elem_in_sub];
            acc += x_val * d_y * (float) qi;
        }
    }

    acc = gfx906_warp_reduce_sum(acc);

    if (lane == 0) {
        dst[(size_t)(token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
