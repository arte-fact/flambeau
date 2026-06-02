// mmvq_iq3_xxs_dp4a — IQ3_XXS weight × Q8_1 → {F32, F16}, DP4A.
//
// Ported from llama.cpp's `vec_dot_iq3_xxs_q8_1` (vecdotq.cuh:1109).
// Reuses the IQ3_S apply-signs trick and the IQ3XXS_GRID codebook
// from `iq_grid.cuh`.
//
// IQ3_XXS block layout (256 elements per super-block):
//   d (F16)
//   qs[96]: bytes [0..64) hold 64 codebook indices (8-bit, one per
//           group-of-4 elements), bytes [64..96) hold 8 uint32 aux
//           words — one per ib32 sub-block — packing a 4-bit ls in
//           bits [28..32) and four 7-bit sign-LUT indices in the low
//           28 bits.
//
// Algorithm per (iqs, l_grp) lane:
//   q3 = 8 bytes from qs[4*iqs..4*iqs+8) — 8 codebook indices.
//   aux32 = qs[64 + 2*iqs ..]  (4 bytes)
//   ls = aux32 >> 28           (per-sub-block scale 0..15)
//   signs_in = aux32 >> (7 * l_grp)
//   ksigns = unpack_ksigns(signs_in & 0x7F) — popcount parity for 8th bit
//   signs0 byte k = (ksigns_byte_k AND { 0x01,0x02,0x04,0x08 }) != 0 ? 0xFF : 0
//   signs1 byte k = (ksigns_byte_k AND { 0x10,0x20,0x40,0x80 }) != 0 ? 0xFF : 0
//   grid_l = IQ3XXS_GRID[q3[2*l_grp + 0]]
//   grid_h = IQ3XXS_GRID[q3[2*l_grp + 1]]
//   apply per-byte sign, dp4a vs u0/u1.
//   sumi = (ls * sumi + sumi/2) / 2   (integer; matches reference rounding)
//   contrib = d_super · d_y · sumi.
//
// Launch shape: 256 threads/block, single row, 32 threads per super-
// block (8 iqs × 4 l_grp), BLOCKS_PER_ITER = 8 — same as IQ3_S.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "iq_grid.cuh"
#include "mmvq_store.cuh"

#define MMVQ_IQ3_XXS_THREADS 256
#define MMVQ_IQ3_XXS_WARPS (MMVQ_IQ3_XXS_THREADS / WARP_SIZE)
#define MMVQ_IQ3_XXS_THREADS_PER_BLOCK 32
#define MMVQ_IQ3_XXS_BLOCKS_PER_ITER (MMVQ_IQ3_XXS_THREADS / MMVQ_IQ3_XXS_THREADS_PER_BLOCK)

static __device__ __forceinline__ int flambeau_iq3_xxs_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Per-byte conditional negate. See mmvq_iq3_s_dp4a.cu for the why —
// gfx906 has no __vsub4 and a direct int32 ((grid ^ signs) - signs)
// borrows across byte lanes when a grid byte is 0.
static __device__ __forceinline__ int flambeau_iq3_xxs_apply_signs(
    uint32_t grid, uint32_t signs
) {
    const int8_t g0 = (int8_t)(uint8_t)(grid & 0xFF);
    const int8_t g1 = (int8_t)(uint8_t)((grid >> 8) & 0xFF);
    const int8_t g2 = (int8_t)(uint8_t)((grid >> 16) & 0xFF);
    const int8_t g3 = (int8_t)(uint8_t)((grid >> 24) & 0xFF);
    const int8_t r0 = (signs & 0xFF) ? (int8_t)(-g0) : g0;
    const int8_t r1 = ((signs >> 8) & 0xFF) ? (int8_t)(-g1) : g1;
    const int8_t r2 = ((signs >> 16) & 0xFF) ? (int8_t)(-g2) : g2;
    const int8_t r3 = ((signs >> 24) & 0xFF) ? (int8_t)(-g3) : g3;
    return (int)((uint8_t)r0)
         | ((int)((uint8_t)r1) << 8)
         | ((int)((uint8_t)r2) << 16)
         | ((int)((uint8_t)r3) << 24);
}

// llama.cpp `unpack_ksigns`: extend 7-bit sign field to 8 bits via
// popcount-parity, then broadcast over all 4 bytes.
static __device__ __forceinline__ uint32_t flambeau_iq3_xxs_unpack_ksigns(
    uint32_t v_7bit
) {
    const uint32_t v = v_7bit & 0x7F;
    const uint32_t p = (uint32_t) __builtin_popcount(v) & 1u;
    const uint32_t s = v ^ (p << 7);
    return s * 0x01010101u;
}

template<typename OutT>
__device__ void mmvq_iq3_xxs_dp4a_body(
    const flambeau_block_iq3_xxs* __restrict__ x,
    const flambeau_block_q8_1*    __restrict__ y,
    OutT* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid           = threadIdx.x;
    const int warp          = tid / WARP_SIZE;
    const int lane          = tid & (WARP_SIZE - 1);
    const int sb_idx_in_iter = tid / MMVQ_IQ3_XXS_THREADS_PER_BLOCK;
    const int lane_in_sb    = tid & (MMVQ_IQ3_XXS_THREADS_PER_BLOCK - 1);
    const int iqs_idx       = lane_in_sb >> 2;          // 0..7
    const int l_grp         = lane_in_sb & 3;           // 0..3

    const int iqs           = 2 * iqs_idx;
    const int l0            = 2 * l_grp;

    const flambeau_block_iq3_xxs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int sb = sb_idx_in_iter; sb < n_superblocks_per_row;
         sb += MMVQ_IQ3_XXS_BLOCKS_PER_ITER) {
        const flambeau_block_iq3_xxs* bk = xrow + sb;

        const float d_sb = (float) bk->d;

        // q3 is 8 bytes from qs[4*iqs..4*iqs+8).
        const uint8_t* q3_base = bk->qs + 4 * iqs;
        const uint8_t q3_0 = q3_base[l0 + 0];
        const uint8_t q3_1 = q3_base[l0 + 1];

        // aux32 from qs[QK_K/16 + iqs/2] as get_int_b2 (4 bytes at byte 4*idx).
        // QK_K/16 = 16, so byte offset = 4*16 + 2*iqs = 64 + 2*iqs.
        const uint32_t aux32 =
            *((const uint32_t*) (bk->qs + 64 + 2 * iqs));

        const int ls = (int)(aux32 >> 28);

        const uint32_t signs_full =
            flambeau_iq3_xxs_unpack_ksigns(aux32 >> (7 * l_grp));

        // Byte-select 4 specific bits per (low/high) half via AND masks.
        const uint32_t sel_lo = signs_full & 0x08040201u;
        const uint32_t sel_hi = signs_full & 0x80402010u;

        uint32_t signs0_pack = 0;
        signs0_pack |= ((sel_lo >>  0) & 0xFFu) ? 0x000000FFu : 0;
        signs0_pack |= ((sel_lo >>  8) & 0xFFu) ? 0x0000FF00u : 0;
        signs0_pack |= ((sel_lo >> 16) & 0xFFu) ? 0x00FF0000u : 0;
        signs0_pack |= ((sel_lo >> 24) & 0xFFu) ? 0xFF000000u : 0;

        uint32_t signs1_pack = 0;
        signs1_pack |= ((sel_hi >>  0) & 0xFFu) ? 0x000000FFu : 0;
        signs1_pack |= ((sel_hi >>  8) & 0xFFu) ? 0x0000FF00u : 0;
        signs1_pack |= ((sel_hi >> 16) & 0xFFu) ? 0x00FF0000u : 0;
        signs1_pack |= ((sel_hi >> 24) & 0xFFu) ? 0xFF000000u : 0;

        const uint32_t grid0 = IQ3XXS_GRID[q3_0];
        const uint32_t grid1 = IQ3XXS_GRID[q3_1];

        const int grid_signed0 = flambeau_iq3_xxs_apply_signs(grid0, signs0_pack);
        const int grid_signed1 = flambeau_iq3_xxs_apply_signs(grid1, signs1_pack);

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + (iqs >> 1);
        const int u0 = ((const int*) ya->qs)[l0 + 0];
        const int u1 = ((const int*) ya->qs)[l0 + 1];
        const float d_y = (float) ya->d;

        int sumi = flambeau_iq3_xxs_dp4a(grid_signed0, u0, 0);
        sumi     = flambeau_iq3_xxs_dp4a(grid_signed1, u1, sumi);

        // Reference rounding: sumi = (ls * sumi + sumi/2) / 2.
        sumi = (ls * sumi + sumi / 2) / 2;

        acc += d_sb * d_y * (float) sumi;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_IQ3_XXS_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_IQ3_XXS_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_IQ3_XXS_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_iq3_xxs_dp4a_q8_1(
    const flambeau_block_iq3_xxs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq3_xxs_dp4a_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_iq3_xxs_dp4a_q8_1_f16(
    const flambeau_block_iq3_xxs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq3_xxs_dp4a_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
