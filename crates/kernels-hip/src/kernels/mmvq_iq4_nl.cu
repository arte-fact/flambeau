// mmvq_iq4_nl — IQ4_NL weight × Q8_1 activation → F32 dst.
// cert-grade single-row reference (scalar inner loop; dp4a-with-LUT
// variant deferred to Phase 3a-perf).
//
// IQ4_NL is the 4-bit non-linear quant: 32-element block, f16 d, 16 bytes
// of nibble-packed LUT indices. The codes map to signed i8 values via
// `kvalues_iq4nl[16]`. Reconstruction:
//   y_i = d * KVALUES_IQ4NL[code_i]
// Low nibble of byte j → elem j; high nibble → elem j + 16.
//
// Thread layout (mirrors mmvq_q4_K's wave64 / sub-block split):
//   blockIdx.x → output row.
//   threadIdx.x ∈ [0, 64). Each wave64 owns one row.
//   lane_lo = lane & 31 (0..31): position within a 32-elem block.
//   byte_off = lane_lo & 15 (0..15): which byte in the 16-byte qs[].
//   nibble_hi = lane_lo >> 4 (0..1): low (0..15) or high (16..31) nibble.
//   block_hi  = lane >> 5: which of two consecutive blocks this lane owns.
// Each iteration of the outer K-loop covers 2 IQ4_NL blocks (64 elems).
// Lanes 0..31 process block (b + 0), lanes 32..63 process block (b + 1).

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_IQ4_NL_THREADS 64

extern "C" __global__ void flambeau_mmvq_iq4_nl_q8_1(
    const flambeau_block_iq4_nl* __restrict__ x,
    const flambeau_block_q8_1*    __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane      = threadIdx.x;        // 0..63
    const int lane_lo   = lane & 31;          // 0..31
    const int byte_off  = lane_lo & 15;       // 0..15
    const int nibble_hi = lane_lo >> 4;       // 0 (low nibble) or 1 (high nibble)
    const int block_hi  = lane >> 5;          // 0 or 1
    const int elem_in_block = byte_off + nibble_hi * 16;  // 0..31

    const flambeau_block_iq4_nl* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;

    for (int b = block_hi; b < n_blocks_per_row; b += 2) {
        const flambeau_block_iq4_nl* bk = xrow + b;
        const flambeau_block_q8_1*   by = y + b;

        const float d_x = (float) bk->d;
        const int   byte_v = (int) bk->qs[byte_off];
        const int   code   = (byte_v >> (nibble_hi * 4)) & 0x0F;
        const float x_val  = d_x * (float) flambeau_iq4nl_lut(code);

        const float d_y = (float) by->d;
        const int   qi  = (int) by->qs[elem_in_block];
        const float y_val = d_y * (float) qi;

        acc += x_val * y_val;
    }

    // Both block_hi halves contribute to the same row; full-wave reduce.
    acc = gfx906_warp_reduce_sum(acc);

    if (lane == 0) {
        dst[row] = acc;
    }
}
