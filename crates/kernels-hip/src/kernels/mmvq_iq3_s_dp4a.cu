// mmvq_iq3_s_dp4a — IQ3_S weight × Q8_1 activation → {F32, F16}, DP4A.
//
// Ported from llama.cpp's `vec_dot_iq3_s_q8_1` (vecdotq.cuh:1148). First
// codebook-helper port of Phase 3 hard tier; lays the pattern that
// IQ3_XXS, IQ2_*, IQ1_* will inherit.
//
// IQ3_S block layout (256 elements per super-block):
//   d (F16)        — super-block scale
//   qs[64]         — low 8 bits of the 9-bit codebook index, per group of 4
//   qh[8]          — 9th bit of the codebook index, one bit per qs byte
//   signs[32]      — packed sign bits, one per element
//   scales[4]      — 4-bit nibble scales (1 + 2*x), 8 sub-block scales total
//
// Per (iqs, l_grp) lane:
//   l0 = 2 * l_grp (0,2,4,6)
//   Two 9-bit codebook indices → IQ3S_GRID[idx] = uint32_t with 4 packed
//   unsigned i8 magnitudes.
//   Per-byte sign from signs_byte's 8 bits, applied via `apply_signs_i8x4`
//   (manual XOR + per-byte negate — gfx906 has no __vcmpne4/__vsub4).
//   2 dp4as (low + high grid) → sumi. Scale: 1 + 2 * nibble.
//   contrib = d · d_y · scale · sumi.
//
// Launch: 256 threads/block, single output row, 32 threads per super-block
// (8 effective iqs values × 4 l_grp), BLOCKS_PER_ITER = 8.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "iq_grid.cuh"
#include "mmvq_store.cuh"

#define MMVQ_IQ3_S_THREADS 256
#define MMVQ_IQ3_S_WARPS (MMVQ_IQ3_S_THREADS / WARP_SIZE)
#define MMVQ_IQ3_S_THREADS_PER_BLOCK 32
#define MMVQ_IQ3_S_BLOCKS_PER_ITER (MMVQ_IQ3_S_THREADS / MMVQ_IQ3_S_THREADS_PER_BLOCK)

static __device__ __forceinline__ int flambeau_iq3_s_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Apply per-byte sign to a uint32_t of 4 packed unsigned i8 magnitudes.
// `signs` has per-byte values in {0x00, 0xFF}: 0x00 keeps the magnitude,
// 0xFF negates it. Returns int32_t with 4 packed signed i8 values suitable
// for dp4a. Per-byte negation avoids cross-byte borrow that would corrupt
// a direct int32 `(grid ^ signs) - signs` on AMD (no __vsub4 on gfx906).
static __device__ __forceinline__ int flambeau_iq3_s_apply_signs(
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
__device__ void mmvq_iq3_s_dp4a_body(
    const flambeau_block_iq3_s* __restrict__ x,
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
    const int sb_idx_in_iter = tid / MMVQ_IQ3_S_THREADS_PER_BLOCK;
    const int lane_in_sb    = tid & (MMVQ_IQ3_S_THREADS_PER_BLOCK - 1);
    const int iqs_idx       = lane_in_sb >> 2;          // 0..7 — which effective iqs
    const int l_grp         = lane_in_sb & 3;           // 0..3 — which l0 within {0,2,4,6}

    const int iqs           = 2 * iqs_idx;              // 0,2,4,6,8,10,12,14
    const int l0            = 2 * l_grp;

    const flambeau_block_iq3_s* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int sb = sb_idx_in_iter; sb < n_superblocks_per_row;
         sb += MMVQ_IQ3_S_BLOCKS_PER_ITER) {
        const flambeau_block_iq3_s* bk = xrow + sb;

        const float d_sb = (float) bk->d;

        // qs spans 8 bytes per iqs (= 2 int32s read at byte offset 4*iqs..4*iqs+8).
        // For QI3_S the MMVQ kernel steps iqs by 2, so iqs ∈ {0,2,...,14} covers
        // qs[0..63] fully.
        const uint8_t* qs_base = bk->qs + 4 * iqs;
        const uint8_t qs0 = qs_base[l0 + 0];
        const uint8_t qs1 = qs_base[l0 + 1];

        const uint32_t qh = bk->qh[iqs >> 1];
        const int g_idx0 = (int) qs0 | (((int) qh << (8 - l0)) & 0x100);
        const int g_idx1 = (int) qs1 | (((int) qh << (7 - l0)) & 0x100);
        const uint32_t grid0 = IQ3S_GRID[g_idx0];
        const uint32_t grid1 = IQ3S_GRID[g_idx1];

        // signs: signs_packed_32 = 4 bytes from signs[4 * (iqs/2) ..].
        // The MMVQ kernel reads it as an int32 and indexes byte l0/2 = l_grp.
        const uint8_t signs_byte = bk->signs[4 * (iqs >> 1) + l_grp];

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

        const int grid_signed0 = flambeau_iq3_s_apply_signs(grid0, signs0_pack);
        const int grid_signed1 = flambeau_iq3_s_apply_signs(grid1, signs1_pack);

        // Q8_1 sub-block index = iqs/2; int32 indices l0+0 and l0+1.
        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + (iqs >> 1);
        const int u0 = ((const int*) ya->qs)[l0 + 0];
        const int u1 = ((const int*) ya->qs)[l0 + 1];
        const float d_y = (float) ya->d;

        int sumi = flambeau_iq3_s_dp4a(grid_signed0, u0, 0);
        sumi     = flambeau_iq3_s_dp4a(grid_signed1, u1, sumi);

        // Scale: 1 + 2 * 4-bit nibble. The nibble index follows llama.cpp's
        // formula (bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F.
        const int sc_byte = bk->scales[iqs >> 2];
        const int sc_nibble = (sc_byte >> ((iqs << 1) & 0x04)) & 0x0F;
        const int scale = 1 + 2 * sc_nibble;

        acc += d_sb * d_y * (float) (sumi * scale);
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_IQ3_S_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_IQ3_S_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_IQ3_S_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_iq3_s_dp4a_q8_1(
    const flambeau_block_iq3_s* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq3_s_dp4a_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_iq3_s_dp4a_q8_1_f16(
    const flambeau_block_iq3_s* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq3_s_dp4a_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
