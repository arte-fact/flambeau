#pragma once
// Templated Q4_K MMVQ (decode GEMV), r2 layout: 64 threads = 2 output rows per
// block, each row owned by a 32-lane group. Q4_K is a 256-element super-block of
// 8 × 32 sub-blocks: fp16 `d`/`dmin`, eight 6-bit (scale, min) pairs packed in
// `scales[12]`, 4-bit quants in `qs[128]` (low nibble = even sub-block of a
// pair, high nibble = odd). Ports llama.cpp's `vec_dot_q4_K_q8_1_impl_vmmq`:
// each lane covers one int32 of a sub-block pair → two dp4a (value + broadcast
// min) per sub-block, y = d·sc·q − dmin·m. Backend-neutral: `dp4a` / the 2-row
// reduce / `flambeau_q4k_scale_min` come from the arch header + shared headers;
// int loads go through `load_int_b2`. `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<class OutT>
static __device__ __forceinline__ void mmvq_q4_k_r2_dp4a_row(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_superblocks_per_row)
{
    const int row_pair = blockIdx.x;
    const int lane     = threadIdx.x;            // 0..63
    const int row_hi   = lane >> 5;              // 0/1 — which of the 2 rows
    const int lane_lo  = lane & 31;              // 0..31 — intra-row lane

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const int pair_idx = lane_lo >> 3;           // 0..3 — sub-block pair
    const int iqs      = lane_lo & 7;            // 0..7 — int32 within the pair's slice
    const int sub_lo   = pair_idx * 2;           // even sub-block (low nibble)
    const int sub_hi   = sub_lo + 1;             // odd sub-block (high nibble)

    const flambeau_block_q4_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;
    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q4_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        const int qs_int = load_int_b2((const int8_t*) bk->qs, pair_idx * 8 + iqs);
        const int q_lo = qs_int & 0x0F0F0F0F;
        const int q_hi = (qs_int >> 4) & 0x0F0F0F0F;

        uint8_t sc_lo = 0, m_lo = 0, sc_hi = 0, m_hi = 0;
        flambeau_q4k_scale_min(sub_lo, bk->scales, &sc_lo, &m_lo);
        flambeau_q4k_scale_min(sub_hi, bk->scales, &sc_hi, &m_hi);

        const flambeau_block_q8_1* ya_lo = y + (b * 8 + sub_lo);
        const flambeau_block_q8_1* ya_hi = y + (b * 8 + sub_hi);
        const int u_lo = load_int_b2(ya_lo->qs, iqs);
        const int u_hi = load_int_b2(ya_hi->qs, iqs);

        const int sumi_lo = dp4a(q_lo, u_lo, 0);
        const int sumi_hi = dp4a(q_hi, u_hi, 0);
        const int summ_lo = dp4a(0x01010101, u_lo, 0);
        const int summ_hi = dp4a(0x01010101, u_hi, 0);

        const float d_y_lo = (float) ya_lo->d;
        const float d_y_hi = (float) ya_hi->d;

        acc += d_y_lo * (d * (float) sc_lo * (float) sumi_lo - dmin * (float) m_lo * (float) summ_lo);
        acc += d_y_hi * (d * (float) sc_hi * (float) sumi_hi - dmin * (float) m_hi * (float) summ_hi);
    }

    mmvq_reduce_store_2row<OutT>(acc, dst, row, lane_lo);
}

// Single-row Q4_K MMVQ, 256 threads: 32 lanes per super-block (4 lanes × 8
// sub-blocks, two ints each), 8 super-blocks in flight per block. Same lane
// layout as `mmvq_q5_k_dp4a_row` without the fifth bit. The r2 layout above
// walks a row's super-blocks one at a time with one int per lane, which
// leaves too few loads in flight at decode. y = d·sc·q − dmin·m; the
// −dmin·m·Σu term folds through the Q8_1 header `s`, quartered across the 4
// lanes of a sub-block.
template<class OutT>
static __device__ __forceinline__ void mmvq_q4_k_dp4a_row(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_superblocks_per_row)
{
    constexpr int THREADS     = 256;
    constexpr int SB_PER_ITER = THREADS / 32;

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid        = threadIdx.x;
    const int sb_in_iter = tid >> 5;
    const int lane_in_sb = tid & 31;
    const int sub        = lane_in_sb >> 2;
    const int lane4      = lane_in_sb & 3;

    const flambeau_block_q4_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;
    for (int sb = sb_in_iter; sb < n_superblocks_per_row; sb += SB_PER_ITER) {
        const flambeau_block_q4_K* bk = xrow + sb;

        uint8_t sc_u8 = 0, m_u8 = 0;
        flambeau_q4k_scale_min(sub, bk->scales, &sc_u8, &m_u8);

        const int8_t* qs_pair = (const int8_t*) &bk->qs[(sub >> 1) * 32];
        const int vl_lo = load_int_b2(qs_pair, lane4);
        const int vl_hi = load_int_b2(qs_pair, lane4 + 4);
        const int v_lo = (sub & 1) ? ((vl_lo >> 4) & 0x0F0F0F0F) : (vl_lo & 0x0F0F0F0F);
        const int v_hi = (sub & 1) ? ((vl_hi >> 4) & 0x0F0F0F0F) : (vl_hi & 0x0F0F0F0F);

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + sub;
        const int u_lo = load_int_b2(ya->qs, lane4);
        const int u_hi = load_int_b2(ya->qs, lane4 + 4);

        const int sumi = dp4a(v_hi, u_hi, dp4a(v_lo, u_lo, 0));
        acc += (float) bk->d * (float) sc_u8 * (float) ya->d * (float) sumi
             - (float) bk->dmin * (float) m_u8 * (float) ya->s * 0.25f;
    }

    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}
