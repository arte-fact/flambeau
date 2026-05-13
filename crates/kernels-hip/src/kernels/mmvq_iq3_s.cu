// mmvq_iq3_s — IQ3_S weight × Q8_1 activation → F32 dst.
// cert-grade single-row reference (scalar inner loop with 9-bit codebook).
//
// IQ3_S layout per 256-elem super-block (110 bytes):
//   f16 d
//   u8  qs[64]      — codebook low 8 bits
//   u8  qh[8]       — codebook 9th bits (one bit per qs byte)
//   u8  signs[32]   — per-byte 8-bit sign masks
//   u8  scales[4]   — 4-bit nibbles, two per byte, 8 sub-block scales total
//
// Per element of sub-block ib32, element offset (8*l + j_in_8):
//   sc_byte   = scales[ib32 >> 1]
//   sc_nibble = (ib32 & 1) ? (sc_byte >> 4) : (sc_byte & 0xF)
//   db        = d * (1.0 + 2.0 * sc_nibble)
//   qh_byte   = qh[ib32]
//   qh_shift  = g_choice == 0 ? (8 - 2*l) : (7 - 2*l)
//   g_idx     = qs[ib32*8 + 2*l + g_choice] | ((qh_byte << qh_shift) & 256)
//   g_u32     = IQ3S_GRID[g_idx]
//   lane_val  = (uint8_t)(g_u32 >> (8 * (j_in_8 & 3)))
//   sign      = (signs[ib32*4 + l] & KMASK_IQ2XS[j_in_8]) ? -1.0f : 1.0f
//   y_w       = db * (float)lane_val * sign
//
// Thread layout matches mmvq_iq3_xxs.cu; only the per-element compute body
// differs.

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "iq_grid.cuh"

extern "C" __global__ void flambeau_mmvq_iq3_s_q8_1(
    const flambeau_block_iq3_s*  __restrict__ x,
    const flambeau_block_q8_1*   __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane      = threadIdx.x;
    const int sub_hi    = lane >> 5;
    const int lane_lo   = lane & 31;
    const int l         = lane_lo >> 3;
    const int j_in_8    = lane_lo & 7;
    const int j_lane    = j_in_8 & 3;
    const int g_choice  = (j_in_8 >> 2);

    const flambeau_block_iq3_s* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_iq3_s* bk = xrow + b;
        const float d = (float) bk->d;
        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const int ib32 = 2 * grp + sub_hi;

            const uint8_t sc_byte = bk->scales[ib32 >> 1];
            const int sc_nibble = (ib32 & 1) ? (sc_byte >> 4) : (sc_byte & 0x0F);
            const float db = d * (1.0f + 2.0f * (float) sc_nibble);

            const uint8_t qh_byte = bk->qh[ib32];
            const int qh_shift = g_choice ? (7 - 2 * l) : (8 - 2 * l);
            const int g_idx = (int) bk->qs[ib32 * 8 + 2 * l + g_choice]
                            | (((int) qh_byte << qh_shift) & 256);
            const uint32_t g_u32 = IQ3S_GRID[g_idx];
            const int lane_val = (int)((g_u32 >> (8 * j_lane)) & 0xFF);

            const uint8_t signs = bk->signs[ib32 * 4 + l];
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

    if (lane == 0) {
        dst[row] = acc;
    }
}
