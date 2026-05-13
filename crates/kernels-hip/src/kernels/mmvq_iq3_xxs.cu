// mmvq_iq3_xxs — IQ3_XXS weight × Q8_1 activation → F32 dst.
// cert-grade single-row reference (scalar inner loop with codebook lookup).
//
// IQ3_XXS layout per 256-elem super-block (98 bytes):
//   f16 d
//   u8  qs[64]   — codebook indices (2 indices per (ib32, l) pair)
//   u8  scs[32]  — packed u32-per-ib32 with 4-bit scale + 4 × 7-bit sign idx
//
// Per element of sub-block ib32, element offset (8*l + j_in_8):
//   aux32     = scs_u32[ib32]
//   db        = d * (0.5 + (aux32 >> 28)) * 0.5
//   sign_byte = KSIGNS_IQ2XS[(aux32 >> (7*l)) & 0x7F]
//   grid_idx  = qs[ib32*8 + 2*l + (j_in_8 >= 4 ? 1 : 0)]
//   g_u32     = IQ3XXS_GRID[grid_idx]
//   lane_val  = (uint8_t)(g_u32 >> (8 * (j_in_8 & 3)))        // 0..127 magnitude
//   sign      = (sign_byte & (1 << j_in_8)) ? -1.0f : 1.0f
//   y_w       = db * (float)lane_val * sign
//
// Thread layout (mirrors mmvq_q4_K_q8_1 / mmvq_iq4_xs_q8_1):
//   blockIdx.x  → output row.
//   threadIdx.x ∈ [0, 64). One wave64 per row.
//   sub_hi  = lane >> 5 → owns sub-block 2*grp + sub_hi within group.
//   lane_lo = lane & 31 → 8*l + j_in_8 element offset inside sub-block.
//   4 groups × 64 lanes × 1 elem = 256 elems per super-block.

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "iq_grid.cuh"

extern "C" __global__ void flambeau_mmvq_iq3_xxs_q8_1(
    const flambeau_block_iq3_xxs* __restrict__ x,
    const flambeau_block_q8_1*    __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane      = threadIdx.x;
    const int sub_hi    = lane >> 5;
    const int lane_lo   = lane & 31;
    const int l         = lane_lo >> 3;       // 0..3
    const int j_in_8    = lane_lo & 7;        // 0..7
    const int j_lane    = j_in_8 & 3;         // 0..3
    const int g_choice  = (j_in_8 >> 2);      // 0 → g1, 1 → g2

    const flambeau_block_iq3_xxs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_iq3_xxs* bk = xrow + b;
        const float d = (float) bk->d;
        const uint8_t* scs = bk->qs + QK_K / 4;   // 32-byte tail
        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const int ib32 = 2 * grp + sub_hi;

            // 4 bytes → aux32 for this ib32. The struct's qs[] is u8 so build
            // the u32 byte-wise to avoid misalignment UB across blocks.
            const uint8_t* sp = scs + 4 * ib32;
            const uint32_t aux32 =
                  (uint32_t) sp[0]
                | ((uint32_t) sp[1] << 8)
                | ((uint32_t) sp[2] << 16)
                | ((uint32_t) sp[3] << 24);

            const float    db        = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            const uint8_t  sign_byte = KSIGNS_IQ2XS[(aux32 >> (7 * l)) & 0x7F];
            const int      grid_idx  = (int) bk->qs[ib32 * 8 + 2 * l + g_choice];
            const uint32_t g_u32     = IQ3XXS_GRID[grid_idx];
            const int      lane_val  = (int)((g_u32 >> (8 * j_lane)) & 0xFF);
            const float    sign      = (sign_byte & (1u << j_in_8)) ? -1.0f : 1.0f;
            const float    x_val     = db * (float) lane_val * sign;

            // Activation: ib32 == Q8_1 block index within the super-block,
            // element offset within that block = 8*l + j_in_8.
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
