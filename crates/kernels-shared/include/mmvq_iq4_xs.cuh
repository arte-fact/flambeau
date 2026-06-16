#pragma once
// Templated single-row IQ4_XS MMVQ (decode GEMV). IQ4_XS is a 256-element
// super-block of 8 × 32 sub-blocks: fp16 `d`, eight 6-bit per-sub-block scales
// (high 2 bits in `scales_h`, low 4 in `scales_l[4]`), and `qs[128]` of 4-bit
// indices into the same `kvalues_iq4nl` codebook as IQ4_NL. Ports llama.cpp's
// `vec_dot_iq4_xs_q8_1`: 256 threads, 32 per super-block (4 lanes × 8
// sub-blocks). Reuses `iq4_nl_table_4` (byte_perm LUT) + the shared neutral
// `flambeau_iq4_xs_scale`; int loads go through `load_int_b2`. `OutT` is the
// destination dtype (F32 or F16-dst).
#include "mmvq_iq4_nl.cuh"

template<class OutT>
static __device__ __forceinline__ void mmvq_iq4_xs_dp4a_row(
    const flambeau_block_iq4_xs* __restrict__ x,
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
    const int sub        = lane_in_sb >> 2;      // 0..7 — sub-block
    const int lane4      = lane_in_sb & 3;       // 0..3 — int32 within the sub-block

    const flambeau_block_iq4_xs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;
    for (int sb = sb_in_iter; sb < n_superblocks_per_row; sb += SB_PER_ITER) {
        const flambeau_block_iq4_xs* bk = xrow + sb;

        const float d  = (float) bk->d;
        const int   ls = flambeau_iq4_xs_scale(sub, bk->scales_h, bk->scales_l);

        const int aux_q4 = load_int_b2((const int8_t*) bk->qs, sub * 4 + lane4);
        const unsigned int q_low  = (unsigned int) aux_q4 & 0x0F0F0F0Fu;
        const unsigned int q_high = ((unsigned int) aux_q4 >> 4) & 0x0F0F0F0Fu;

        const unsigned int v_lo = iq4_nl_table_4(q_low);
        const unsigned int v_hi = iq4_nl_table_4(q_high);

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + sub;
        const int u_lo = load_int_b2(ya->qs, lane4);
        const int u_hi = load_int_b2(ya->qs, lane4 + 4);
        const float d_y = (float) ya->d;

        int sumi = dp4a((int) v_lo, u_lo, 0);
        sumi     = dp4a((int) v_hi, u_hi, sumi);

        acc += d * (float) ls * d_y * (float) sumi;
    }

    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}
