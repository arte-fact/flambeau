#pragma once
// Templated fused gate+up Q5_K dense MMVQ (decode GEMV). Q5_K is a 256-element
// super-block: fp16 d/dmin, eight 6-bit (scale, min) pairs in `scales[12]`,
// 4-bit low nibbles in `qs[128]`, the 5th-bit plane in `qh[32]`. 256 threads =
// 8 super-blocks/iter × 32 threads each; within a super-block `sub` (0..7)
// selects the 32-element sub-block and `lane4` (0..3) the int32 slice. The gate
// and up rows share one Q8_1 activation read per (super-block, sub); each output
// reconstructs `q5 = nibble | (qh_bit << 4)`, two dp4a (value + broadcast min),
// `y = d·sc·q − dmin·m`. Backend-neutral: `dp4a` / `flambeau_q4k_scale_min` /
// the two-output reduce come from the shared headers; every misaligned int load
// goes through `load_int_b2`. `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

static __device__ __forceinline__ float mmvq_q5_k_gate_up_partial(
    const flambeau_block_q5_K& bk, int sub, int lane4,
    int u_lo, int u_hi, float d_y, float s_y)
{
    const float d    = (float) bk.d;
    const float dmin = (float) bk.dmin;

    uint8_t sc_u8 = 0, m_u8 = 0;
    flambeau_q4k_scale_min(sub, bk.scales, &sc_u8, &m_u8);

    const int8_t* qs_pair = (const int8_t*) &bk.qs[(sub >> 1) * 32];
    const int vl_lo = load_int_b2(qs_pair, lane4);
    const int vl_hi = load_int_b2(qs_pair, lane4 + 4);
    const int vh_lo = load_int_b2((const int8_t*) bk.qh, lane4);
    const int vh_hi = load_int_b2((const int8_t*) bk.qh, lane4 + 4);

    const int vlq_lo = (sub & 1) ? ((vl_lo >> 4) & 0x0F0F0F0F) : (vl_lo & 0x0F0F0F0F);
    const int vlq_hi = (sub & 1) ? ((vl_hi >> 4) & 0x0F0F0F0F) : (vl_hi & 0x0F0F0F0F);
    const int vhq_lo = ((vh_lo >> sub) << 4) & 0x10101010;
    const int vhq_hi = ((vh_hi >> sub) << 4) & 0x10101010;

    int sumi_d = dp4a(vlq_lo | vhq_lo, u_lo, 0);
    sumi_d     = dp4a(vlq_hi | vhq_hi, u_hi, sumi_d);
    return d * (float) sc_u8 * d_y * (float) sumi_d - dmin * (float) m_u8 * s_y * 0.25f;
}

template<int THREADS, class OutT = float>
static __device__ __forceinline__ void mmvq_q5_k_gate_up_dp4a_row(
    const flambeau_block_q5_K* __restrict__ gate_w,
    const flambeau_block_q5_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ gate_out,
    OutT* __restrict__ up_out,
    int n_rows_gate, int n_rows_up, int n_superblocks_per_row)
{
    constexpr int THREADS_PER_SB  = 32;
    constexpr int BLOCKS_PER_ITER = THREADS / THREADS_PER_SB;

    const int row = blockIdx.x;
    const bool do_gate = row < n_rows_gate;
    const bool do_up   = row < n_rows_up;
    if (!do_gate && !do_up) return;

    const int tid        = threadIdx.x;
    const int sb_in_iter = tid >> 5;
    const int lane_in_sb = tid & 31;
    const int sub        = lane_in_sb >> 2;
    const int lane4      = lane_in_sb & 3;

    const flambeau_block_q5_K* g_row = gate_w + (size_t) row * n_superblocks_per_row;
    const flambeau_block_q5_K* u_row = up_w   + (size_t) row * n_superblocks_per_row;

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int sb = sb_in_iter; sb < n_superblocks_per_row; sb += BLOCKS_PER_ITER) {
        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + sub;
        const int u_lo = load_int_b2(ya->qs, lane4);
        const int u_hi = load_int_b2(ya->qs, lane4 + 4);
        const float d_y = (float) ya->d;
        const float s_y = (float) ya->s;

        if (do_gate)
            acc_g += mmvq_q5_k_gate_up_partial(g_row[sb], sub, lane4, u_lo, u_hi, d_y, s_y);
        if (do_up)
            acc_u += mmvq_q5_k_gate_up_partial(u_row[sb], sub, lane4, u_lo, u_hi, d_y, s_y);
    }

    mmvq_reduce_store_gate_up<THREADS, OutT>(acc_g, acc_u, gate_out, up_out, row, do_gate, do_up);
}
