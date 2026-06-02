// mmvq_iq2_xxs_dp4a — IQ2_XXS weight × Q8_1 → {F32, F16}, DP4A.
//
// Ported from llama.cpp's `vec_dot_iq2_xxs_q8_1` (vecdotq.cuh:985).
// IQ2_XXS packs (4 codebook indices + 4 × 7-bit signs + 4-bit scale)
// into 8 qs bytes per sub-block.
//
// Block layout (256 elems, 66 bytes):
//   d (F16)
//   qs[64] — 8 sub-blocks × 8 bytes each:
//     [0..4)  : 4 codebook indices (one byte each, indexing IQ2XXS_GRID)
//     [4..8)  : aux32 packing 4 × 7-bit sign-LUT indices (bits 0..28) +
//               4-bit super-block-scale-nibble (bits 28..32)
//
// Algorithm per (iqs, l_grp) lane:
//   ib32 = iqs/2     — sub-block index
//   qs region = qs[8*ib32..8*ib32 + 8) = qs[4*iqs..4*iqs + 8)
//   idx_byte = qs[4*iqs + l_grp]                       — codebook index
//   aux32 = qs[4*iqs + 4 .. 4*iqs + 8)                 — same for all 4 l_grp
//   sign_field_7bit = (aux32 >> (7 * l_grp)) & 0x7F
//   grid_pos = IQ2XXS_GRID[idx_byte]                   — uint64 → 2 int32s
//   signs = unpack_ksigns(sign_field_7bit)             — broadcasts s × 0x01010101
//   sign-apply byte-wise to grid_lo/grid_hi via flambeau_iq2_xxs_apply_signs
//   dp4a vs u0/u1, sum into sumi_lane.
//   scale_nibble = aux32 >> 28
//   contrib = d_super · d_y · sumi_lane · (scale + 0.5) · 0.25
//
// One scale per sub-block; all 4 l_grp lanes for one (sb, ib32) share it,
// so distributively each lane multiplies its sumi_lane by the same scale.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "iq_grid.cuh"
#include "mmvq_store.cuh"

#define MMVQ_IQ2_XXS_THREADS 256
#define MMVQ_IQ2_XXS_WARPS (MMVQ_IQ2_XXS_THREADS / WARP_SIZE)
#define MMVQ_IQ2_XXS_THREADS_PER_BLOCK 32
#define MMVQ_IQ2_XXS_BLOCKS_PER_ITER (MMVQ_IQ2_XXS_THREADS / MMVQ_IQ2_XXS_THREADS_PER_BLOCK)

static __device__ __forceinline__ int flambeau_iq2_xxs_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ int flambeau_iq2_xxs_apply_signs(
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

static __device__ __forceinline__ uint32_t flambeau_iq2_xxs_unpack_ksigns(
    uint32_t v_7bit
) {
    const uint32_t v = v_7bit & 0x7F;
    const uint32_t p = (uint32_t) __builtin_popcount(v) & 1u;
    const uint32_t s = v ^ (p << 7);
    return s * 0x01010101u;
}

template<typename OutT>
__device__ void mmvq_iq2_xxs_dp4a_body(
    const flambeau_block_iq2_xxs* __restrict__ x,
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
    const int sb_idx_in_iter = tid / MMVQ_IQ2_XXS_THREADS_PER_BLOCK;
    const int lane_in_sb    = tid & (MMVQ_IQ2_XXS_THREADS_PER_BLOCK - 1);
    const int iqs_idx       = lane_in_sb >> 2;          // 0..7
    const int l_grp         = lane_in_sb & 3;           // 0..3

    const int iqs           = 2 * iqs_idx;
    const int l0            = 2 * l_grp;

    const flambeau_block_iq2_xxs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int sb = sb_idx_in_iter; sb < n_superblocks_per_row;
         sb += MMVQ_IQ2_XXS_BLOCKS_PER_ITER) {
        const flambeau_block_iq2_xxs* bk = xrow + sb;

        const float d_sb = (float) bk->d;

        // qs[4*iqs + l_grp] is the codebook index byte (one of 4 per sub-block).
        const uint8_t* qs_base = bk->qs + 4 * iqs;
        const uint8_t idx_byte = qs_base[l_grp];

        // aux32 lives at qs[4*iqs + 4 .. 4*iqs + 8). All four l_grp lanes for
        // this sub-block read the same aux32; only the sign-field shift differs.
        const uint32_t aux32 = *((const uint32_t*) (qs_base + 4));

        const int scale_nibble = (int) (aux32 >> 28);

        const uint32_t signs_full =
            flambeau_iq2_xxs_unpack_ksigns(aux32 >> (7 * l_grp));

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

        const uint64_t grid_u64 = IQ2XXS_GRID[idx_byte];
        const uint32_t grid0 = (uint32_t) grid_u64;
        const uint32_t grid1 = (uint32_t) (grid_u64 >> 32);

        const int grid_signed0 = flambeau_iq2_xxs_apply_signs(grid0, signs0_pack);
        const int grid_signed1 = flambeau_iq2_xxs_apply_signs(grid1, signs1_pack);

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + (iqs >> 1);
        const int u0 = ((const int*) ya->qs)[l0 + 0];
        const int u1 = ((const int*) ya->qs)[l0 + 1];
        const float d_y = (float) ya->d;

        int sumi = flambeau_iq2_xxs_dp4a(grid_signed0, u0, 0);
        sumi     = flambeau_iq2_xxs_dp4a(grid_signed1, u1, sumi);

        // (2*scale + 1) / 8 == (scale + 0.5) * 0.25.
        const float scale_factor = ((float) scale_nibble + 0.5f) * 0.25f;
        acc += d_sb * d_y * (float) sumi * scale_factor;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_IQ2_XXS_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_IQ2_XXS_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_IQ2_XXS_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_iq2_xxs_dp4a_q8_1(
    const flambeau_block_iq2_xxs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq2_xxs_dp4a_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_iq2_xxs_dp4a_q8_1_f16(
    const flambeau_block_iq2_xxs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq2_xxs_dp4a_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
