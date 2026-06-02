// indexed_moe_mmvq_q2_k_r2_dp4a — Q2_K MoE MMVQ with DP4A inner loop.
// Drop-in replacement for `indexed_moe_mmvq_q2_k.cu` (scalar FP32). Same
// per-element math as dense Lever 1 Q2_K (`mmvq_q2_k_r2_dp4a.cu`) plus
// the standard MoE wiring: expert_ids indirection on the weight pointer,
// per-token activation slab, per-(token, slot, row) dst.
//
// Launch: blockDim = { 64 } (wave64), gridDim = { n_row_pairs,
// n_tokens * top_k, 1 }. 2 output rows per block, 32 lanes per row;
// each lane processes 2 (iqs, i) pairs per super-block via the outer
// j ∈ {0, 1} loop (j=0 → iqs ∈ 0..7, j=1 → iqs ∈ 8..15).
//
// Per (iqs, i): two DP4As — one for the 2-bit qs nibbles (vi), one for
// the broadcast 4-bit min (m_packed) — then acc += d_y * (d_sb * sc_lo
// * sumi_d − dmin_sb * sumi_m), applied inside the loop because d/dmin
// vary per super-block.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"

#define MOE_Q2K_QI8_1   8
#define MOE_Q2K_QR2_K   4

static __device__ __forceinline__ int flambeau_indexed_moe_q2_k_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q2_k_r2_dp4a_q8_1(
    const flambeau_block_q2_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    float* __restrict__ dst,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row
) {
    const int row_pair = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;

    if (token >= n_tokens) return;

    const int lane    = threadIdx.x;
    const int row_hi  = lane >> 5;
    const int lane_lo = lane & 31;

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const flambeau_block_q2_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    const int iqs_lo = lane_lo >> 2;               // 0..7
    const int i      = lane_lo & 3;                // 0..3

    float acc = 0.0f;

    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        const flambeau_block_q2_K* bk = xrow + sb;
        const float d_sb    = (float) bk->d;
        const float dmin_sb = (float) bk->dmin;

        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int iqs           = iqs_lo + 8 * j;     // 0..15
            const int bq8_offset    = MOE_Q2K_QR2_K * (iqs / MOE_Q2K_QI8_1);
            const int scale_offset  = iqs - (iqs & (MOE_Q2K_QI8_1 - 1))
                                    + ((iqs & (MOE_Q2K_QI8_1 - 1)) / (MOE_Q2K_QI8_1 / 2));

            const int sc_byte  = bk->scales[scale_offset + 2 * i];
            const int sc_lo    = sc_byte & 0x0F;
            const int m_4bit   = (sc_byte >> 4) & 0x0F;
            const int m_packed = m_4bit * 0x01010101;

            const int v  = ((const int*) bk->qs)[iqs];
            const int vi = (v >> (2 * i)) & 0x03030303;

            const flambeau_block_q8_1* ya = y_row + (size_t) sb * 8 + bq8_offset + i;
            const int u_i = ((const int*) ya->qs)[iqs & (MOE_Q2K_QI8_1 - 1)];
            const float d_y = (float) ya->d;

            const int sumi_d = flambeau_indexed_moe_q2_k_dp4a(vi, u_i, 0);
            const int sumi_m = flambeau_indexed_moe_q2_k_dp4a(m_packed, u_i, 0);

            acc += d_y * (d_sb * (float) sc_lo * (float) sumi_d
                          - dmin_sb * (float) sumi_m);
        }
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
