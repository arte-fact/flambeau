// mmvq_iq4_xs — IQ4_XS weight × Q8_1 activation → F32 dst.
// cert-grade single-row reference (scalar inner loop).
//
// IQ4_XS is the 4-bit non-linear K-quant: 256-element super-block with 8
// sub-blocks of 32. Per-sub-block signed 6-bit scale split into
// `scales_l` (low 4 bits × 8 → 4 bytes) and `scales_h` (high 2 bits × 8
// → 16 bits). Per element:
//   y = d * (scale_signed) * KVALUES_IQ4NL[code]
// with `scale_signed ∈ [-32, 31]` (biased). The 4-bit codes map to the
// same signed-i8 LUT as IQ4_NL.
//
// qs layout: sub-block ib owns bytes qs[ib*16 .. ib*16+16] — low nibble →
// elem 0..15 of sub-block, high nibble → elem 16..31.
//
// Thread layout (mirrors mmvq_q4_K.cu):
//   blockIdx.x → output row.
//   threadIdx.x ∈ [0, 64). One wave64 per row.
//   sub_hi   = lane >> 5: which sub-block of the current group (0 or 1).
//   lane_lo  = lane & 31 → byte_off ∈ [0, 16), nibble_hi ∈ [0, 2).
// Each group covers 64 elements (= 2 adjacent sub-blocks).
// 4 groups × 64 elements = one full 256-elem super-block per outer iter.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_IQ4_XS_THREADS 64

extern "C" __global__ void flambeau_mmvq_iq4_xs_q8_1(
    const flambeau_block_iq4_xs* __restrict__ x,
    const flambeau_block_q8_1*   __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane      = threadIdx.x;        // 0..63
    const int sub_hi    = lane >> 5;          // 0 or 1
    const int lane_lo   = lane & 31;          // 0..31
    const int byte_off  = lane_lo & 15;       // 0..15
    const int nibble_hi = lane_lo >> 4;       // 0 (low) or 1 (high)
    const int elem_in_sub = byte_off + nibble_hi * 16;  // 0..31

    const flambeau_block_iq4_xs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_iq4_xs* bk = xrow + b;

        const float    d         = (float) bk->d;
        const uint16_t scales_h  = bk->scales_h;
        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

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
            const float y_val = d_y * (float) qi;

            acc += x_val * y_val;
        }
    }

    acc = gfx906_warp_reduce_sum(acc);

    if (lane == 0) {
        dst[row] = acc;
    }
}
