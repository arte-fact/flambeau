#pragma once
// Templated single-row Q3_K MMVQ (decode GEMV). Q3_K is a 256-element
// super-block: 1 high bit per element in `hmask[32]`, low 2 bits in `qs[64]`,
// sixteen signed 6-bit scales packed in `scales[12]`, fp16 super-block `d`.
// Ports llama.cpp's `vec_dot_q3_K_q8_1_impl_mmvq`: 256 threads, 64 per
// super-block (one (iqs,i) pair per lane), 4 super-blocks per iter. The 3-bit
// value is low-2-bits minus the (inverted) high bit; gfx906 has no per-byte
// SIMD subtract, so the two planes are separate dp4a calls subtracted as
// scalars (no cross-byte borrow). Backend-neutral: `dp4a` / the reduce from the
// arch header + shared headers; misaligned `qs`/`hmask` int loads go through
// `load_int_b2`. `OutT` is the destination dtype (F32 or F16-dst).
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<class OutT>
static __device__ __forceinline__ void mmvq_q3_k_dp4a_row(
    const flambeau_block_q3_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_superblocks_per_row)
{
    constexpr int THREADS        = 256;
    constexpr int THREADS_PER_SB = 64;
    constexpr int SB_PER_ITER    = THREADS / THREADS_PER_SB;
    constexpr int QI3_K = 16;
    constexpr int QR3_K = 4;
    constexpr int QI8_1 = 8;

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid        = threadIdx.x;
    const int sb_in_iter = tid / THREADS_PER_SB;
    const int lane_in_sb = tid & (THREADS_PER_SB - 1);
    const int iqs        = lane_in_sb >> 2;          // 0..15
    const int i          = lane_in_sb & 3;           // 0..3

    const int bq8_offset    = QR3_K * (iqs / (QI3_K / 2));   // 0 or 4
    const int scale_offset  = iqs - (iqs & (QI8_1 - 1)) + ((iqs & (QI8_1 - 1)) / (QI8_1 / 2));
    const int isc           = scale_offset + 2 * i;
    const int isc_low       = isc & 7;
    const int sc_shift_low  = 4 * (isc >> 3);
    const int isc_high      = isc & 3;
    const int sc_shift_high = 2 * (isc >> 2);

    const flambeau_block_q3_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;
    for (int sb = sb_in_iter; sb < n_superblocks_per_row; sb += SB_PER_ITER) {
        const flambeau_block_q3_K* bk = xrow + sb;

        const float d = (float) bk->d;

        const int sc_low  = (bk->scales[isc_low] >> sc_shift_low) & 0xF;
        const int sc_high = ((bk->scales[(QI3_K / 2) + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc      = (sc_low | sc_high) - 32;

        const int vl      = load_int_b2((const int8_t*) bk->qs, iqs);
        const int vh_full = ~load_int_b2((const int8_t*) bk->hmask, iqs & ((QI3_K / 2) - 1));
        const int vh      = vh_full >> bq8_offset;

        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + bq8_offset + i;
        const int u_i = load_int_b2(ya->qs, iqs & (QI8_1 - 1));
        const float d_y = (float) ya->d;

        const int sumi_l = dp4a(vil, u_i, 0);
        const int sumi_h = dp4a(vih, u_i, 0);
        const int sumi   = sumi_l - sumi_h;

        acc += d * (float) sc * d_y * (float) sumi;
    }

    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}
