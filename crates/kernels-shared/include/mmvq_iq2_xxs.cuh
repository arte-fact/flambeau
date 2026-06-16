#pragma once
// Templated single-row IQ2_XXS MMVQ (decode GEMV). IQ2_XXS (2.0625 bpw) packs,
// per 8-byte sub-block: 4 codebook-index bytes into `IQ2XXS_GRID` + an aux32 of
// four 7-bit sign-LUT indices and a 4-bit scale nibble. Ports llama.cpp's
// `vec_dot_iq2_xxs_q8_1`: 256 threads, 32 per super-block (4 lanes × 8
// sub-blocks). Each lane gathers one grid entry (uint64 → 2 int32), applies the
// sign LUT, and dp4a's against Q8_1. Backend-neutral: `dp4a` / the reduce /
// `IQ2XXS_GRID` from the arch + shared headers; `__builtin_popcount` is a
// compiler builtin on both; misaligned int loads go through `load_int_b2`.
// `OutT` is the destination dtype (F32 or F16-dst).
#include "quant_traits.cuh"
#include "iq_grid.cuh"
#include "mmvq_reduce.cuh"

// Apply per-byte sign flips (signs.byte[i] != 0 → negate grid.byte[i]).
static __device__ __forceinline__ int iq2xxs_apply_signs(unsigned int grid, unsigned int signs) {
    const int8_t g0 = (int8_t) (uint8_t) (grid & 0xFF);
    const int8_t g1 = (int8_t) (uint8_t) ((grid >> 8) & 0xFF);
    const int8_t g2 = (int8_t) (uint8_t) ((grid >> 16) & 0xFF);
    const int8_t g3 = (int8_t) (uint8_t) ((grid >> 24) & 0xFF);
    const int8_t r0 = (signs & 0xFF) ? (int8_t) (-g0) : g0;
    const int8_t r1 = ((signs >> 8) & 0xFF) ? (int8_t) (-g1) : g1;
    const int8_t r2 = ((signs >> 16) & 0xFF) ? (int8_t) (-g2) : g2;
    const int8_t r3 = ((signs >> 24) & 0xFF) ? (int8_t) (-g3) : g3;
    return (int) ((uint8_t) r0) | ((int) ((uint8_t) r1) << 8)
         | ((int) ((uint8_t) r2) << 16) | ((int) ((uint8_t) r3) << 24);
}

// 7-bit sign-LUT index → per-element sign mask broadcast to 4 bytes.
static __device__ __forceinline__ unsigned int iq2xxs_unpack_ksigns(unsigned int v_7bit) {
    const unsigned int v = v_7bit & 0x7F;
    const unsigned int p = (unsigned int) __builtin_popcount(v) & 1u;
    const unsigned int s = v ^ (p << 7);
    return s * 0x01010101u;
}

template<class OutT>
static __device__ __forceinline__ void mmvq_iq2_xxs_dp4a_row(
    const flambeau_block_iq2_xxs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_superblocks_per_row)
{
    constexpr int THREADS        = 256;
    constexpr int THREADS_PER_SB = 32;
    constexpr int SB_PER_ITER    = THREADS / THREADS_PER_SB;

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid        = threadIdx.x;
    const int sb_in_iter = tid >> 5;             // 0..7 — super-block in this iter
    const int lane_in_sb = tid & 31;
    const int iqs_idx    = lane_in_sb >> 2;      // 0..7
    const int l_grp      = lane_in_sb & 3;       // 0..3
    const int iqs        = 2 * iqs_idx;
    const int l0         = 2 * l_grp;

    const flambeau_block_iq2_xxs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;
    for (int sb = sb_in_iter; sb < n_superblocks_per_row; sb += SB_PER_ITER) {
        const flambeau_block_iq2_xxs* bk = xrow + sb;

        const float d_sb = (float) bk->d;
        const int8_t* qs_base = (const int8_t*) (bk->qs + 4 * iqs);
        const unsigned int idx_byte = (unsigned int) (uint8_t) qs_base[l_grp];
        const unsigned int aux32 = (unsigned int) load_int_b2(qs_base + 4, 0);

        const int scale_nibble = (int) (aux32 >> 28);
        const unsigned int signs_full = iq2xxs_unpack_ksigns(aux32 >> (7 * l_grp));

        const unsigned int sel_lo = signs_full & 0x08040201u;
        const unsigned int sel_hi = signs_full & 0x80402010u;
        unsigned int signs0_pack = 0, signs1_pack = 0;
        signs0_pack |= ((sel_lo >> 0) & 0xFFu) ? 0x000000FFu : 0;
        signs0_pack |= ((sel_lo >> 8) & 0xFFu) ? 0x0000FF00u : 0;
        signs0_pack |= ((sel_lo >> 16) & 0xFFu) ? 0x00FF0000u : 0;
        signs0_pack |= ((sel_lo >> 24) & 0xFFu) ? 0xFF000000u : 0;
        signs1_pack |= ((sel_hi >> 0) & 0xFFu) ? 0x000000FFu : 0;
        signs1_pack |= ((sel_hi >> 8) & 0xFFu) ? 0x0000FF00u : 0;
        signs1_pack |= ((sel_hi >> 16) & 0xFFu) ? 0x00FF0000u : 0;
        signs1_pack |= ((sel_hi >> 24) & 0xFFu) ? 0xFF000000u : 0;

        const uint64_t grid_u64 = IQ2XXS_GRID[idx_byte];
        const int grid_signed0 = iq2xxs_apply_signs((unsigned int) grid_u64, signs0_pack);
        const int grid_signed1 = iq2xxs_apply_signs((unsigned int) (grid_u64 >> 32), signs1_pack);

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + (iqs >> 1);
        const int u0 = load_int_b2(ya->qs, l0 + 0);
        const int u1 = load_int_b2(ya->qs, l0 + 1);
        const float d_y = (float) ya->d;

        int sumi = dp4a(grid_signed0, u0, 0);
        sumi     = dp4a(grid_signed1, u1, sumi);

        const float scale_factor = ((float) scale_nibble + 0.5f) * 0.25f;
        acc += d_sb * d_y * (float) sumi * scale_factor;
    }

    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}
