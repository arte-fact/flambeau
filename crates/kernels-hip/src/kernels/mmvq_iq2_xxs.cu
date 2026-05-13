// mmvq_iq2_xxs — IQ2_XXS weight × Q8_1 activation → F32 dst.
// cert-grade single-row reference (scalar inner loop with 256×u64 codebook).
//
// IQ2_XXS layout per 256-elem super-block (66 bytes):
//   f16 d
//   u8  qs[64]   — viewed as 8 × (u32 idx-tuple, u32 sign+scale-tuple)
//
// Per ib32 (0..8) we read 8 qs bytes as two u32:
//   aux0 = qs[ib32*8 + 0..4]    → 4 × 8-bit codebook indices
//   aux1 = qs[ib32*8 + 4..8]    → 4 × 7-bit sign-LUT indices [bits 0..28]
//                                 + 4-bit scale [bits 28..32]
//   db   = d * (0.5 + (aux1 >> 28)) * 0.25
//   For l ∈ [0,4):
//     idx_byte  = (aux0 >> (8*l)) & 0xFF
//     g_u64     = IQ2XXS_GRID[idx_byte]
//     sign_byte = KSIGNS_IQ2XS[(aux1 >> (7*l)) & 0x7F]
//     For j ∈ [0,8):
//       y_w[8*l + j] = db * (uint8)(g_u64 >> (8*j)) * sign(bit j)
//
// Each codebook lookup yields a full 8-element group; one lane handles
// one element. Same wave64 layout as mmvq_iq3_xxs.cu (lane_lo = 8*l + j_in_8).

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "iq_grid.cuh"

extern "C" __global__ void flambeau_mmvq_iq2_xxs_q8_1(
    const flambeau_block_iq2_xxs* __restrict__ x,
    const flambeau_block_q8_1*    __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane    = threadIdx.x;
    const int sub_hi  = lane >> 5;
    const int lane_lo = lane & 31;
    const int l       = lane_lo >> 3;      // 0..3
    const int j_in_8  = lane_lo & 7;       // 0..7

    const flambeau_block_iq2_xxs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_iq2_xxs* bk = xrow + b;
        const float d = (float) bk->d;
        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;

        #pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const int ib32 = 2 * grp + sub_hi;
            const uint8_t* sp = bk->qs + 8 * ib32;
            const uint32_t aux0 =
                  (uint32_t) sp[0] | ((uint32_t) sp[1] << 8)
                | ((uint32_t) sp[2] << 16) | ((uint32_t) sp[3] << 24);
            const uint32_t aux1 =
                  (uint32_t) sp[4] | ((uint32_t) sp[5] << 8)
                | ((uint32_t) sp[6] << 16) | ((uint32_t) sp[7] << 24);

            const float    db        = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            const int      idx       = (int)((aux0 >> (8 * l)) & 0xFF);
            const uint64_t g_u64     = IQ2XXS_GRID[idx];
            const int      lane_val  = (int)((g_u64 >> (8 * j_in_8)) & 0xFF);
            const uint8_t  sign_byte = KSIGNS_IQ2XS[(aux1 >> (7 * l)) & 0x7F];
            const float    sign      = (sign_byte & (1u << j_in_8)) ? -1.0f : 1.0f;
            const float    x_val     = db * (float) lane_val * sign;

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
