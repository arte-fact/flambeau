// indexed_moe_mmvq_q6_k — Q6_K MMVQ with per-token expert routing.
// Q6_K sibling of `indexed_moe_mmvq_q4_k.cu`. Same indexing /
// launch-shape contract; inner arithmetic is byte-identical to
// `mmvq_q6_k.cu`. Needed because UD-Q4_K_S and similar mixed-quant
// GGUFs promote some `ffn_down_exps` from Q4_K to Q6_K for quality.
// Layout:
// weights [n_experts, n_rows, n_sb_per_row] Q6_K
// activations [n_tokens, n_sb_per_row * 8] Q8_1 (8 Q8_1 blocks per super-block)
// expert_ids [n_tokens, top_k] i32
// output [n_tokens, top_k, n_rows] F32
// Launch:
// blockDim = { 64 } (one wave64, matches mmvq_q6_k.cu)
// gridDim = { n_rows, n_tokens * top_k, 1 }
// Lane mapping (lane ∈ 0..63), mirroring mmvq_q6_k.cu:
// h = lane / 32 {0, 1} — which 128-element half of the super-block
// pos = lane % 32 0..31 — position within that half
// lsub = pos / 16 {0, 1} — which of the two per-8-scales pairs
// Each lane processes 4 elements per super-block (q_idx ∈ 0..3) →
// 8 lanes × 4 × 2 halves = 64 elements/lane group, × 32 = one super-block.

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_indexed_moe_mmvq_q6_k_q8_1(
    const flambeau_block_q6_K* __restrict__ x,     // [n_experts, n_rows, n_sb]
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

    const int lane = threadIdx.x;    // 0..63
    const int h    = lane >> 5;      // 0 or 1
    const int pos  = lane & 31;      // 0..31
    const int lsub = pos >> 4;       // 0 or 1

    const flambeau_block_q6_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    float acc = 0.0f;

    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q6_K* bk = xrow + b;

        const float d = (float) bk->d;
        const flambeau_block_q8_1* y_sb = y_row + b * 8;

        const uint8_t qh_byte = bk->qh[32 * h + pos];

        #pragma unroll
        for (int q_idx = 0; q_idx < 4; ++q_idx) {
            const int ql_off = 64 * h + ((q_idx & 1) ? pos + 32 : pos);
            const int ql_byte = (int) bk->ql[ql_off];
            const int nibble = (q_idx < 2) ? (ql_byte & 0x0F) : (ql_byte >> 4);
            const int qh_bits = (qh_byte >> (2 * q_idx)) & 0x3;
            const int raw_q = (nibble | (qh_bits << 4)) - 32;

            const int scale_idx = 8 * h + 2 * q_idx + lsub;
            const int sc = (int) bk->scales[scale_idx];

            const float x_val = d * (float) sc * (float) raw_q;

            const int y_block = h * 4 + q_idx;
            const flambeau_block_q8_1* ya = y_sb + y_block;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[pos];
            const float y_val = d_y * (float) qi;

            acc += x_val * y_val;
        }
    }

    acc = gfx906_warp_reduce_sum(acc);

    if (lane == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
