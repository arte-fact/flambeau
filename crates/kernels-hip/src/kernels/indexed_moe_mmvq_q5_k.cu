// indexed_moe_mmvq_q5_k — Q5_K MMVQ with per-token expert routing.
//
// Q5_K sibling of `indexed_moe_mmvq_q4_k.cu` / `indexed_moe_mmvq_q6_k.cu`.
// Inner arithmetic is byte-identical to `mmvq_q5_k.cu` (5-bit = 4-bit nibble
// + high bit from `qh`, mask `1<<(2*grp)` for low-nibble, `2<<(2*grp)` for
// high-nibble). Needed because Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL
// promotes 13/48 `ffn_down_exps` from Q4_K to Q5_K for quality.
//
// Layout:
//   weights       [n_experts, n_rows, n_sb_per_row]   Q5_K
//   activations   [n_tokens, n_sb_per_row * 8]        Q8_1 (8 blocks/SB)
//   expert_ids    [n_tokens, top_k]                   i32
//   output        [n_tokens, top_k, n_rows]           F32
//
// Launch:
//   blockDim  = { 64 }                (one wave64)
//   gridDim   = { n_rows, n_tokens * top_k, 1 }

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_indexed_moe_mmvq_q5_k_q8_1(
    const flambeau_block_q5_K* __restrict__ x,     // [n_experts, n_rows, n_sb]
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

    const int lane     = threadIdx.x;   // 0..63
    const int byte_off = lane & 31;     // 0..31
    const int hi_half  = lane >> 5;     // 0 or 1

    const flambeau_block_q5_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    float acc = 0.0f;

    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q5_K* bk = xrow + b;

        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;
        const flambeau_block_q8_1* y_sb = y_row + b * 8;

        #pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const int sub = 2 * grp + hi_half;
            uint8_t sc = 0, m = 0;
            flambeau_q4k_scale_min(sub, bk->scales, &sc, &m);

            const int byte_v = (int) bk->qs[grp * 32 + byte_off];
            const int raw_q4 = hi_half ? (byte_v >> 4) : (byte_v & 0x0F);

            const uint8_t qh_byte = bk->qh[byte_off];
            const int mask   = (hi_half ? 2 : 1) << (2 * grp);
            const int raw_q  = raw_q4 + ((qh_byte & mask) ? 16 : 0);

            const float x_val = d * (float) sc * (float) raw_q - dmin * (float) m;

            const flambeau_block_q8_1* ya = y_sb + sub;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[byte_off];
            const float y_val = d_y * (float) qi;

            acc += x_val * y_val;
        }
    }

    acc = gfx906_warp_reduce_sum(acc);

    if (lane == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
