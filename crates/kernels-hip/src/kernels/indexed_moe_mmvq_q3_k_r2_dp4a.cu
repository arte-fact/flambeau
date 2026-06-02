// indexed_moe_mmvq_q3_k_r2_dp4a — Q3_K MoE MMVQ with DP4A inner loop.
// Drop-in replacement for `indexed_moe_mmvq_q3_k.cu` (which does scalar
// FP32 multiplies). Same per-element math as `mmvq_q3_k_r2_dp4a.cu`
// (dense Lever 1 Q3_K) plus the standard MoE wiring: expert_ids
// indirection on the weight pointer, per-token activation slab,
// per-(token, slot, row) dst.
//
// Launch: blockDim = { 64 } (wave64), gridDim = { n_row_pairs,
// n_tokens * top_k, 1 }. 2 output rows per block (lanes 0..31 own
// row R, lanes 32..63 own row R+1). Each lane processes 2 (iqs, i)
// pairs per super-block via an outer j ∈ {0, 1} loop (j=0 → iqs ∈
// 0..7, j=1 → iqs ∈ 8..15).
//
// Per (iqs, i): two DP4As — one for the 2-bit qs nibbles (vil), one
// for the hmask high-bit subtraction (vih) — sumi_l - sumi_h gives
// the signed dot product before scaling.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"

#define MOE_Q3K_QI3_K  16
#define MOE_Q3K_QR3_K   4
#define MOE_Q3K_QI8_1   8

static __device__ __forceinline__ int flambeau_indexed_moe_q3_k_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q3_k_r2_dp4a_q8_1(
    const flambeau_block_q3_K* __restrict__ x,
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

    const int lane     = threadIdx.x;              // 0..63
    const int row_hi   = lane >> 5;                // 0 → row R, 1 → row R+1
    const int lane_lo  = lane & 31;                // 0..31

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const flambeau_block_q3_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    const int iqs_lo = lane_lo >> 2;               // 0..7
    const int i      = lane_lo & 3;                // 0..3

    float acc = 0.0f;

    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        const flambeau_block_q3_K* bk = xrow + sb;
        const float d = (float) bk->d;

        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int iqs           = iqs_lo + 8 * j;            // 0..15

            const int bq8_offset    = MOE_Q3K_QR3_K * (iqs / (MOE_Q3K_QI3_K / 2));   // 0 or 4
            const int scale_offset  = iqs - (iqs & (MOE_Q3K_QI8_1 - 1))
                                    + ((iqs & (MOE_Q3K_QI8_1 - 1)) / (MOE_Q3K_QI8_1 / 2));
            const int isc           = scale_offset + 2 * i;
            const int isc_low       = isc & 7;
            const int sc_shift_low  = 4 * (isc >> 3);
            const int isc_high      = isc & 3;
            const int sc_shift_high = 2 * (isc >> 2);

            const int sc_low  = (bk->scales[isc_low] >> sc_shift_low) & 0xF;
            const int sc_high = ((bk->scales[(MOE_Q3K_QI3_K / 2) + isc_high]
                                  >> sc_shift_high) & 3) << 4;
            const int sc      = (sc_low | sc_high) - 32;

            const int vl      = ((const int*) bk->qs)[iqs];
            const int vh_full = ~((const int*) bk->hmask)[iqs & ((MOE_Q3K_QI3_K / 2) - 1)];
            const int vh      = vh_full >> bq8_offset;

            const int vil = (vl >> (2 * i)) & 0x03030303;
            const int vih = ((vh >> i) << 2) & 0x04040404;

            const flambeau_block_q8_1* ya = y_row + sb * 8 + bq8_offset + i;
            const int u_i = ((const int*) ya->qs)[iqs & (MOE_Q3K_QI8_1 - 1)];
            const float d_y = (float) ya->d;

            const int sumi_l = flambeau_indexed_moe_q3_k_dp4a(vil, u_i, 0);
            const int sumi_h = flambeau_indexed_moe_q3_k_dp4a(vih, u_i, 0);
            const int sumi   = sumi_l - sumi_h;

            acc += d * (float) sc * d_y * (float) sumi;
        }
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
