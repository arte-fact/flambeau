// mmvq_iq2_xs_dp4a — IQ2_XS weight × Q8_1 → {F32, F16}, DP4A.
//
// Ported from llama.cpp's `vec_dot_iq2_xs_q8_1` (vecdotq.cuh:1020).
// IQ2_XS packs both the codebook index and the sign mask into a single
// uint16: low 9 bits = IQ2XS_GRID index (512 entries), high 7 bits =
// signs (popcount-parity for the 8th bit) — no separate signs[] array
// or qh.
//
// Block layout (256 elems, 74 bytes):
//   d (F16)
//   qs[64]      — 32 uint16 entries (one per 8-element group)
//   scales[8]   — 8 bytes, 4-bit nibble pairs (flambeau uses only [0..4),
//                 see IQ2_S note).
//
// Algorithm per (iqs, l_grp) lane:
//   q2_entry = qs_u16[2*iqs + l_grp]                  (uint16)
//   grid_pos = IQ2XS_GRID[q2_entry & 0x1FF]            (uint64 → 2 int32)
//   signs    = unpack_ksigns(q2_entry >> 9)            (popcount-parity)
//   sign-apply byte-wise to grid_lo/grid_hi.
//   dp4a vs u0/u1, add to sumi_lane.
//   contrib  = d_super · d_y · sumi_lane · (ls + 0.5) · 0.25
//   where ls = flambeau scales[iqs/4] nibble (low/high per parity of iqs/2).

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "iq_grid.cuh"
#include "mmvq_store.cuh"

#define MMVQ_IQ2_XS_THREADS 256
#define MMVQ_IQ2_XS_WARPS (MMVQ_IQ2_XS_THREADS / WARP_SIZE)
#define MMVQ_IQ2_XS_THREADS_PER_BLOCK 32
#define MMVQ_IQ2_XS_BLOCKS_PER_ITER (MMVQ_IQ2_XS_THREADS / MMVQ_IQ2_XS_THREADS_PER_BLOCK)

static __device__ __forceinline__ int flambeau_iq2_xs_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ int flambeau_iq2_xs_apply_signs(
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

static __device__ __forceinline__ uint32_t flambeau_iq2_xs_unpack_ksigns(
    uint32_t v_7bit
) {
    const uint32_t v = v_7bit & 0x7F;
    const uint32_t p = (uint32_t) __builtin_popcount(v) & 1u;
    const uint32_t s = v ^ (p << 7);
    return s * 0x01010101u;
}

template<typename OutT>
__device__ void mmvq_iq2_xs_dp4a_body(
    const flambeau_block_iq2_xs* __restrict__ x,
    const flambeau_block_q8_1*   __restrict__ y,
    OutT* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid           = threadIdx.x;
    const int warp          = tid / WARP_SIZE;
    const int lane          = tid & (WARP_SIZE - 1);
    const int sb_idx_in_iter = tid / MMVQ_IQ2_XS_THREADS_PER_BLOCK;
    const int lane_in_sb    = tid & (MMVQ_IQ2_XS_THREADS_PER_BLOCK - 1);
    const int iqs_idx       = lane_in_sb >> 2;          // 0..7
    const int l_grp         = lane_in_sb & 3;           // 0..3

    const int iqs           = 2 * iqs_idx;
    const int l0            = 2 * l_grp;

    const flambeau_block_iq2_xs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int sb = sb_idx_in_iter; sb < n_superblocks_per_row;
         sb += MMVQ_IQ2_XS_BLOCKS_PER_ITER) {
        const flambeau_block_iq2_xs* bk = xrow + sb;

        const float d_sb = (float) bk->d;

        // q2 is the uint16 array view of qs. llama.cpp reads via get_int_b2
        // then casts to uint16*, picking entry l0/2 (= l_grp). We do the same
        // directly: q2[2*iqs + l_grp] for iqs ∈ {0,2,...,14}.
        const uint16_t* q2 = (const uint16_t*) bk->qs;
        const uint32_t q2_entry = (uint32_t) q2[2 * iqs + l_grp];

        const int grid_idx = (int) (q2_entry & 0x1FFu);
        const uint64_t grid_u64 = IQ2XS_GRID[grid_idx];
        const uint32_t grid0 = (uint32_t) grid_u64;
        const uint32_t grid1 = (uint32_t)(grid_u64 >> 32);

        const uint32_t signs_full = flambeau_iq2_xs_unpack_ksigns(q2_entry >> 9);

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

        const int grid_signed0 = flambeau_iq2_xs_apply_signs(grid0, signs0_pack);
        const int grid_signed1 = flambeau_iq2_xs_apply_signs(grid1, signs1_pack);

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + (iqs >> 1);
        const int u0 = ((const int*) ya->qs)[l0 + 0];
        const int u1 = ((const int*) ya->qs)[l0 + 1];
        const float d_y = (float) ya->d;

        int sumi = flambeau_iq2_xs_dp4a(grid_signed0, u0, 0);
        sumi     = flambeau_iq2_xs_dp4a(grid_signed1, u1, sumi);

        // Flambeau scale convention (see mmvq_iq2_s_dp4a note): one
        // 4-bit scale per sub-block, packed two-per-byte in scales[0..4).
        const int sub = iqs >> 1;
        const int sc_byte = (int) bk->scales[sub >> 1];
        const int ls = (sub & 1) ? (sc_byte >> 4) : (sc_byte & 0x0F);

        const float scale_factor = ((float) ls + 0.5f) * 0.25f;
        acc += d_sb * d_y * (float) sumi * scale_factor;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_IQ2_XS_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_IQ2_XS_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_IQ2_XS_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_iq2_xs_dp4a_q8_1(
    const flambeau_block_iq2_xs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq2_xs_dp4a_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_iq2_xs_dp4a_q8_1_f16(
    const flambeau_block_iq2_xs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq2_xs_dp4a_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
