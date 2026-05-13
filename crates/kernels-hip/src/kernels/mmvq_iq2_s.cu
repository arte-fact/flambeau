// mmvq_iq2_s — IQ2_S weight × Q8_1 activation → F32 dst.
// cert-grade single-row (1024×u64 codebook, 10-bit index).
//
// IQ2_S layout per 256-elem super-block (82 bytes):
//   f16 d
//   u8  qs[64]      — qs[0..32]: low 8 of 10-bit codebook idx (one byte per
//                                4-elem-group); qs[32..64]: per-8-elem sign masks
//   u8  qh[8]       — high 2 bits of codebook idx; 4 lanes per qh byte
//   u8  scales[8]   — 4-bit nibble pairs (one byte per pair of sub-blocks)
//
// Per element of sub-block ib32, element offset (8*l + j_in_8):
//   sc_byte = scales[ib32 >> 1]
//   db      = d * (0.5 + ((ib32 & 1) ? (sc_byte >> 4) : (sc_byte & 0xF))) * 0.25
//   idx_lo  = qs[4*ib32 + l]
//   high2   = (qh[ib32] << (8 - 2*l)) & 0x300        // bits 8..9
//   idx     = idx_lo | high2                          // 10-bit
//   grid    = IQ2S_GRID[idx]
//   signs   = qs[32 + 4*ib32 + l]
//   y_w     = db * (uint8)(grid >> (8*j_in_8)) * (signs & KMASK_IQ2XS[j_in_8] ? -1 : 1)
//
// Same wave64 thread layout as mmvq_iq3_s.cu.

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "iq_grid.cuh"

extern "C" __global__ void flambeau_mmvq_iq2_s_q8_1(
    const flambeau_block_iq2_s*  __restrict__ x,
    const flambeau_block_q8_1*   __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane    = threadIdx.x;
    const int sub_hi  = lane >> 5;
    const int lane_lo = lane & 31;
    const int l       = lane_lo >> 3;
    const int j_in_8  = lane_lo & 7;

    const flambeau_block_iq2_s* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_iq2_s* bk = xrow + b;
        const float d = (float) bk->d;
        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const int ib32 = 2 * grp + sub_hi;
            const uint8_t sc_byte = bk->scales[ib32 >> 1];
            const int sc_nibble = (ib32 & 1) ? (sc_byte >> 4) : (sc_byte & 0x0F);
            const float db = d * (0.5f + (float) sc_nibble) * 0.25f;

            const uint8_t qh_byte = bk->qh[ib32];
            const int idx_lo = (int) bk->qs[4 * ib32 + l];
            const int high2  = ((int) qh_byte << (8 - 2 * l)) & 0x300;
            const int idx    = idx_lo | high2;
            const uint64_t g_u64 = IQ2S_GRID[idx];
            const int lane_val = (int)((g_u64 >> (8 * j_in_8)) & 0xFF);

            const uint8_t signs = bk->qs[32 + 4 * ib32 + l];
            const float sign = (signs & KMASK_IQ2XS[j_in_8]) ? -1.0f : 1.0f;
            const float x_val = db * (float) lane_val * sign;

            const flambeau_block_q8_1* ya = y_sb + ib32;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[8 * l + j_in_8];
            const float y_val = d_y * (float) qi;

            acc += x_val * y_val;
        }
    }

    acc = gfx906_warp_reduce_sum(acc);
    if (lane == 0) dst[row] = acc;
}
