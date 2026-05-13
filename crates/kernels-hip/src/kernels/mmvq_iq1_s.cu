// mmvq_iq1_s — IQ1_S weight × Q8_1 activation → F32 dst.
// cert-grade single-row (2048×u64 codebook, 11-bit index, signed-i8 entries
// + a per-sub-block ±IQ1_DELTA offset).
//
// IQ1_S layout per 256-elem super-block (50 bytes):
//   f16 d
//   u8  qs[32]    — low 8 bits of 11-bit codebook idx (one byte per 8-elem grp)
//   u16 qh[8]     — per-sub-block packed:
//                     bits  0..2  = high-3-bits of idx for l=0
//                     bits  3..5  = high-3-bits of idx for l=1
//                     bits  6..8  = high-3-bits of idx for l=2
//                     bits  9..11 = high-3-bits of idx for l=3
//                     bits 12..14 = 3-bit sub-block scale (dl = d * (2*scale + 1))
//                     bit  15     = delta-sign (0 → +IQ1_DELTA, 1 → -IQ1_DELTA)
//
// Per element of sub-block ib32, element offset (8*l + j_in_8):
//   qh_u16   = u16 little-endian at qh[2*ib32 .. 2*ib32+2]
//   dl       = d * (2 * ((qh_u16 >> 12) & 7) + 1)
//   delta    = (qh_u16 & 0x8000) ? -IQ1_DELTA : +IQ1_DELTA
//   idx      = qs[4*ib32 + l] | (((qh_u16 >> (3*l)) & 7) << 8)
//   g_i8     = (int8_t)(IQ1S_GRID[idx] >> (8 * j_in_8))          // SIGNED
//   y_w      = dl * ((float)g_i8 + delta)
//
// Unlike IQ2/IQ3 there is no separate sign mask — the grid entries are
// already signed magnitudes; delta is a uniform per-sub-block shift.

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "iq_grid.cuh"

extern "C" __global__ void flambeau_mmvq_iq1_s_q8_1(
    const flambeau_block_iq1_s*  __restrict__ x,
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

    const flambeau_block_iq1_s* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_iq1_s* bk = xrow + b;
        const float d = (float) bk->d;
        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const int ib32 = 2 * grp + sub_hi;
            const int qh_u16 = (int) bk->qh[2 * ib32]
                             | ((int) bk->qh[2 * ib32 + 1] << 8);
            const float dl    = d * (2.0f * (float)((qh_u16 >> 12) & 7) + 1.0f);
            const float delta = (qh_u16 & 0x8000) ? -IQ1_DELTA : IQ1_DELTA;

            const int idx_lo = (int) bk->qs[4 * ib32 + l];
            const int high3  = (qh_u16 >> (3 * l)) & 7;
            const int idx    = idx_lo | (high3 << 8);
            const uint64_t g_u64 = IQ1S_GRID[idx];
            const int8_t g_i8 = (int8_t)((g_u64 >> (8 * j_in_8)) & 0xFF);
            const float x_val = dl * ((float) g_i8 + delta);

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
