// mmvq_q6_k — Q6_K weight × Q8_1 activation → F32 dst.
//
// Layout diverges from Q4_K/Q5_K: no dmin, per-16-element i8 scales are
// multiplied by a single super-block `d`, and each 6-bit value is 4 low bits
// from `ql` + 2 high bits from `qh`. Elements are organised as two halves
// of 128 elements, each half using 64 ql bytes + 32 qh bytes + 8 of the 16
// super-block scales.
//
// Lane mapping (lane ∈ 0..63):
//   h     = lane / 32               {0, 1}   — which 128-element half
//   pos   = lane % 32                0..31   — position within a half
//   lsub  = pos / 16                 {0, 1}  — which of the two per-8-scales pairs
// Each lane processes 4 elements per super-block (q_idx ∈ 0..3):
//   scale_idx = 8*h + 2*q_idx + lsub
//   ql_byte   = ql[64*h + ((q_idx & 1) ? pos + 32 : pos)]
//   nibble    = q_idx < 2 ? (ql_byte & 0xF) : (ql_byte >> 4)
//   qh_byte   = qh[32*h + pos]
//   qh_bits   = (qh_byte >> (2 * q_idx)) & 0x3
//   raw_q     = (nibble | (qh_bits << 4)) - 32
//   y_block   = h*4 + q_idx
//   y_pos     = pos

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_mmvq_q6_k_q8_1(
    const flambeau_block_q6_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane = threadIdx.x;    // 0..63
    const int h    = lane >> 5;      // 0 or 1
    const int pos  = lane & 31;      // 0..31
    const int lsub = pos >> 4;       // 0 or 1

    const flambeau_block_q6_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q6_K* bk = xrow + b;

        const float d = (float) bk->d;
        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

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
        dst[row] = acc;
    }
}
