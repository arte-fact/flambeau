// mmvq_iq2_s_dp4a — IQ2_S weight × Q8_1 → {F32, F16}, DP4A.
//
// Ported from llama.cpp's `vec_dot_iq2_s_q8_1` (vecdotq.cuh:1061).
// Reuses the apply_signs helper pattern from IQ3_S / IQ3_XXS.
//
// IQ2_S block (256 elems, 82 bytes):
//   d (F16)
//   qs[64]   — bytes [0..32) are codebook indices (low 8 bits of 10-bit);
//              bytes [32..64) are packed per-byte sign masks.
//   qh[8]    — 2 high bits of the codebook index per qs byte (4 packed/byte).
//   scales[8] — 8 sub-block 4-bit scales packed two per byte.
//
// IQ2S_GRID is uint64[1024] (8 i8 per entry, accessed as 2 int32s).
// Algorithm per (iqs, l_grp) lane:
//   qs_pair      = qs[4*(iqs/2) ..]                 (4 bytes via get_int_b2)
//   qh_byte      = qh[iqs/2]
//   signs_byte   = qs[QK_K/4 + 4*(iqs/2) + l_grp]  (one of 4 signs bytes)
//   grid_idx     = qs_pair_byte[l_grp] | ((qh_byte << (8 - 2*l_grp)) & 0x300)
//   grid_lo/hi   = ((int*)&IQ2S_GRID[grid_idx])[0,1]
//   sign-apply (per byte, using flambeau_iq3_s_apply_signs sibling).
//   sumi_lane   = dp4a(grid_lo, u0) + dp4a(grid_hi, u1)
//   Two scales: ls0 = scales[iqs/2] & 0xF (l_grp 0,1); ls1 = scales[iqs/2] >> 4
//               (l_grp 2,3). Final formula in float:
//   acc += d_sb * d_y * sumi_lane * (ls + 0.5) / 4.
//   Across all lanes, sum reduces to llama.cpp's
//   (sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2) / 4 modulo int rounding.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "iq_grid.cuh"
#include "mmvq_store.cuh"

#define MMVQ_IQ2_S_THREADS 256
#define MMVQ_IQ2_S_WARPS (MMVQ_IQ2_S_THREADS / WARP_SIZE)
#define MMVQ_IQ2_S_THREADS_PER_BLOCK 32
#define MMVQ_IQ2_S_BLOCKS_PER_ITER (MMVQ_IQ2_S_THREADS / MMVQ_IQ2_S_THREADS_PER_BLOCK)

static __device__ __forceinline__ int flambeau_iq2_s_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ int flambeau_iq2_s_apply_signs(
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

template<typename OutT>
__device__ void mmvq_iq2_s_dp4a_body(
    const flambeau_block_iq2_s* __restrict__ x,
    const flambeau_block_q8_1*  __restrict__ y,
    OutT* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid           = threadIdx.x;
    const int warp          = tid / WARP_SIZE;
    const int lane          = tid & (WARP_SIZE - 1);
    const int sb_idx_in_iter = tid / MMVQ_IQ2_S_THREADS_PER_BLOCK;
    const int lane_in_sb    = tid & (MMVQ_IQ2_S_THREADS_PER_BLOCK - 1);
    const int iqs_idx       = lane_in_sb >> 2;          // 0..7
    const int l_grp         = lane_in_sb & 3;           // 0..3

    const int iqs           = 2 * iqs_idx;
    const int l0            = 2 * l_grp;

    const flambeau_block_iq2_s* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int sb = sb_idx_in_iter; sb < n_superblocks_per_row;
         sb += MMVQ_IQ2_S_BLOCKS_PER_ITER) {
        const flambeau_block_iq2_s* bk = xrow + sb;

        const float d_sb = (float) bk->d;

        // qs_pair = get_int_b2(bq2->qs, iqs/2) reads 4 bytes at byte offset
        // 4 * (iqs/2) = 2 * iqs.
        const uint8_t* qs_pair_base = bk->qs + 2 * iqs;
        const uint8_t qs_byte = qs_pair_base[l_grp];

        const int qh = (int) bk->qh[iqs >> 1];
        const int grid_idx = (int) qs_byte | ((qh << (8 - l0)) & 0x300);

        // signs_packed_32 lives in qs[QK_K/32 + iqs/2] (get_int_b2 → byte off
        // 4 * (QK_K/32 + iqs/2) = 4 * QK_K/32 + 2*iqs = QK_K/8 + 2*iqs = 32 + 2*iqs.
        // Equivalent to qs[32 + 2*iqs + l_grp].
        const uint8_t signs_byte = bk->qs[32 + 2 * iqs + l_grp];

        // Flambeau convention (matches `mmvq_iq2_s.cu` + `dequant_iq2_s`):
        // one 4-bit scale per *sub-block*. scales[iqs/4] holds two nibbles
        // (low = even sub-block, high = odd). All four l_grp values share
        // this scale (NOT llama.cpp's ls0/ls1 split). Note this differs
        // from llama.cpp's MMVQ which uses scales[iqs/2] (8 bytes).
        const int sub = iqs >> 1;                   // 0..7
        const int sc_byte = (int) bk->scales[sub >> 1];
        const int ls = (sub & 1) ? (sc_byte >> 4) : (sc_byte & 0x0F);

        // IQ2S_GRID entry is uint64 → access as 2 int32s (low, high halves).
        const uint64_t grid_u64 = IQ2S_GRID[grid_idx];
        const uint32_t grid0 = (uint32_t) grid_u64;
        const uint32_t grid1 = (uint32_t)(grid_u64 >> 32);

        // Sign expansion: signs0 byte k from bits {0,1,2,3} of signs_byte;
        // signs1 byte k from bits {4,5,6,7}.
        uint32_t signs0_pack = 0;
        signs0_pack |= (signs_byte & 0x01) ? 0x000000FFu : 0;
        signs0_pack |= (signs_byte & 0x02) ? 0x0000FF00u : 0;
        signs0_pack |= (signs_byte & 0x04) ? 0x00FF0000u : 0;
        signs0_pack |= (signs_byte & 0x08) ? 0xFF000000u : 0;

        uint32_t signs1_pack = 0;
        signs1_pack |= (signs_byte & 0x10) ? 0x000000FFu : 0;
        signs1_pack |= (signs_byte & 0x20) ? 0x0000FF00u : 0;
        signs1_pack |= (signs_byte & 0x40) ? 0x00FF0000u : 0;
        signs1_pack |= (signs_byte & 0x80) ? 0xFF000000u : 0;

        const int grid_signed0 = flambeau_iq2_s_apply_signs(grid0, signs0_pack);
        const int grid_signed1 = flambeau_iq2_s_apply_signs(grid1, signs1_pack);

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + (iqs >> 1);
        const int u0 = ((const int*) ya->qs)[l0 + 0];
        const int u1 = ((const int*) ya->qs)[l0 + 1];
        const float d_y = (float) ya->d;

        int sumi = flambeau_iq2_s_dp4a(grid_signed0, u0, 0);
        sumi     = flambeau_iq2_s_dp4a(grid_signed1, u1, sumi);

        // Scalar reference: db = d * (0.5 + sc_nibble) * 0.25. All four
        // l_grp lanes for one sub-block share this scale.
        const float scale_factor = ((float) ls + 0.5f) * 0.25f;
        acc += d_sb * d_y * (float) sumi * scale_factor;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_IQ2_S_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_IQ2_S_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_IQ2_S_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_iq2_s_dp4a_q8_1(
    const flambeau_block_iq2_s* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq2_s_dp4a_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_iq2_s_dp4a_q8_1_f16(
    const flambeau_block_iq2_s* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq2_s_dp4a_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
