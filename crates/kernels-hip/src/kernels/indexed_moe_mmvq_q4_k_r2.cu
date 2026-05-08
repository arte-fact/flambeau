// indexed_moe_mmvq_q4_k_r2 — Q4_K MoE MMVQ with multi-row r2 DPP reduce.
// Candle P29 `_nw1_r2` pattern adapted for MoE: same 64-thread wave64,
// but each block computes TWO output rows of the expert weight matrix
// for one (token, slot). Lanes 0..31 own row R+0, lanes 32..63 own R+1.
// Half-warp DPP reduce finishes each row's dot product independently.
// Dispatch drop-in for `indexed_moe_mmvq_q4_k.cu`:
// - Grid: { ceil(n_rows / 2), n_tokens * top_k, 1 }
// - Output layout unchanged: [n_tokens, top_k, n_rows]
// Win vs the single-row MoE MMVQ: half the kernel launches for the same
// work, same activation HBM traffic, half the Q4_K-scale-unpack work
// (the block's Q4_K scales are read once per 32 lanes instead of once
// per 32 lanes × 2 launches).

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_indexed_moe_mmvq_q4_k_r2_q8_1(
    const flambeau_block_q4_K* __restrict__ x,     // [n_experts, n_rows, n_sb]
    const flambeau_block_q8_1* __restrict__ y,     // [n_tokens, n_sb * 8]
    const int* __restrict__ expert_ids,            // [n_tokens, top_k]
    float* __restrict__ dst,                       // [n_tokens, top_k, n_rows]
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

    const int lane     = threadIdx.x;              // 0..63
    const int row_hi   = lane >> 5;                // 0 → row R, 1 → row R+1
    const int lane_lo  = lane & 31;                // 0..31 — sub-block position

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;   // odd-row-count boundary

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const flambeau_block_q4_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    float acc = 0.0f;

    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q4_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        const flambeau_block_q8_1* y_sb = y_row + b * 8;

        // Same half-warp / 8-sub-block walk as `mmvq_q4_k_r2.cu`:
        // each lane visits all 8 sub-blocks, reading one quant nibble +
        // one Q8_1 quant byte per iteration.
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint8_t sc = 0, m = 0;
            flambeau_q4k_scale_min(s, bk->scales, &sc, &m);

            const int byte_idx = (s >> 1) * 32 + lane_lo;
            const int byte_v = (int) bk->qs[byte_idx];
            const int raw_q = (s & 1) ? (byte_v >> 4) : (byte_v & 0x0F);

            const float x_val =
                d * (float) sc * (float) raw_q - dmin * (float) m;

            const flambeau_block_q8_1* ya = y_sb + s;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[lane_lo];
            const float y_val = d_y * (float) qi;

            acc += x_val * y_val;
        }
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
