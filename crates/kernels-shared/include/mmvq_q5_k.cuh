#pragma once
// Templated single-row Q5_K MMVQ (decode GEMV). Q5_K is a 256-element
// super-block of 8 × 32 sub-blocks: fp16 `d`/`dmin`, eight 6-bit (scale, min)
// pairs packed in `scales[12]`, one high bit per element in `qh[32]`, low
// nibbles in `qs[128]`. Ports llama.cpp's `vec_dot_q5_K_q8_1_impl_vmmq`: 256
// threads, 32 per super-block (4 lanes × 8 sub-blocks), 8 super-blocks per iter.
// Per element y = d·sc·v − dmin·m (v = nibble | high-bit<<4 ∈ [0,31]); the
// −dmin·m·Σu correction folds via the Q8_1 header `s`, quartered across the 4
// lanes. Backend-neutral: `dp4a` / the reduce / `flambeau_q4k_scale_min` come
// from the arch header + shared headers; misaligned int loads go through
// `load_int_b2`. `OutT` is the destination dtype (F32 or F16-dst).
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<class OutT>
static __device__ __forceinline__ void mmvq_q5_k_dp4a_row(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_superblocks_per_row)
{
    constexpr int THREADS         = 256;
    constexpr int THREADS_PER_SB  = 32;
    constexpr int SB_PER_ITER     = THREADS / THREADS_PER_SB;

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid         = threadIdx.x;
    const int sb_in_iter  = tid >> 5;             // 0..7 — super-block in this iter
    const int lane_in_sb  = tid & 31;            // 0..31 within the super-block group
    const int sub         = lane_in_sb >> 2;     // 0..7 — sub-block
    const int lane4       = lane_in_sb & 3;      // 0..3 — int32 within the sub-block

    const flambeau_block_q5_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;
    for (int sb = sb_in_iter; sb < n_superblocks_per_row; sb += SB_PER_ITER) {
        const flambeau_block_q5_K* bk = xrow + sb;

        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        uint8_t sc_u8 = 0, m_u8 = 0;
        flambeau_q4k_scale_min(sub, bk->scales, &sc_u8, &m_u8);
        const float sc_f = (float) sc_u8;
        const float m_f  = (float) m_u8;

        const int8_t* qs_pair = (const int8_t*) &bk->qs[(sub >> 1) * 32];
        const int vl_lo = load_int_b2(qs_pair, lane4);
        const int vl_hi = load_int_b2(qs_pair, lane4 + 4);
        const int vh_lo = load_int_b2((const int8_t*) bk->qh, lane4);
        const int vh_hi = load_int_b2((const int8_t*) bk->qh, lane4 + 4);

        const int vlq_lo = (sub & 1) ? ((vl_lo >> 4) & 0x0F0F0F0F) : (vl_lo & 0x0F0F0F0F);
        const int vlq_hi = (sub & 1) ? ((vl_hi >> 4) & 0x0F0F0F0F) : (vl_hi & 0x0F0F0F0F);
        const int vhq_lo = ((vh_lo >> sub) << 4) & 0x10101010;
        const int vhq_hi = ((vh_hi >> sub) << 4) & 0x10101010;
        const int v_lo = vlq_lo | vhq_lo;
        const int v_hi = vlq_hi | vhq_hi;

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + sub;
        const int u_lo = load_int_b2(ya->qs, lane4);
        const int u_hi = load_int_b2(ya->qs, lane4 + 4);
        const float d_y = (float) ya->d;
        const float s_y = (float) ya->s;

        int sumi_d = dp4a(v_lo, u_lo, 0);
        sumi_d     = dp4a(v_hi, u_hi, sumi_d);
        acc += d * sc_f * d_y * (float) sumi_d - dmin * m_f * s_y * 0.25f;
    }

    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}
