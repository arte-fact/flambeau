// mmvq_q2_k_r2_dp4a — Q2_K MMVQ DP4A, multi-row r2 (wave64).
//
// r2 refactor of `mmvq_q2_k_dp4a.cu`. Same per-element math (two
// DP4A calls per (iqs, i) — one for the 2-bit qs nibbles, one for
// the broadcast 4-bit min — then `d * sc * sumi_d - dmin * sumi_m`,
// applied inside the loop because d/dmin vary per super-block).
//
// Launch shape mirrors Q3_K r2 dp4a: 64 threads, 2 rows per block,
// 32 lanes per row, each lane processes 2 (iqs, i) pairs per
// super-block via an outer j ∈ {0, 1} loop (j=0 → iqs ∈ 0..7,
// j=1 → iqs ∈ 8..15).
//
// Wins (same shape as Q3_K r2 — half the launches, activation reuse
// across the two rows in a wave, half-warp DPP reduce in lieu of
// cross-warp LDS).

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "mmvq_store.cuh"

#define Q2K_R2_QI2_K  16
#define Q2K_R2_QR2_K   4
#define Q2K_R2_QI8_1   8

static __device__ __forceinline__ int flambeau_q2_k_r2_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template<typename OutT>
__device__ void mmvq_q2_k_r2_dp4a_body(
    const flambeau_block_q2_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row_pair = blockIdx.x;
    const int lane     = threadIdx.x;              // 0..63
    const int row_hi   = lane >> 5;                // 0 → row R, 1 → row R+1
    const int lane_lo  = lane & 31;                // 0..31

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const flambeau_block_q2_K* xrow = x + (size_t) row * n_superblocks_per_row;

    const int iqs_lo = lane_lo >> 2;               // 0..7
    const int i      = lane_lo & 3;                // 0..3

    float acc = 0.0f;

    for (int sb = 0; sb < n_superblocks_per_row; ++sb) {
        const flambeau_block_q2_K* bk = xrow + sb;
        const float d_sb    = (float) bk->d;
        const float dmin_sb = (float) bk->dmin;

        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int iqs           = iqs_lo + 8 * j;     // 0..15
            const int bq8_offset    = Q2K_R2_QR2_K * (iqs / Q2K_R2_QI8_1);
            const int scale_offset  = iqs - (iqs & (Q2K_R2_QI8_1 - 1))
                                    + ((iqs & (Q2K_R2_QI8_1 - 1)) / (Q2K_R2_QI8_1 / 2));

            const int sc_byte = bk->scales[scale_offset + 2 * i];
            const int sc_lo   = sc_byte & 0x0F;
            const int m_4bit  = (sc_byte >> 4) & 0x0F;
            const int m_packed = m_4bit * 0x01010101;

            const int v  = ((const int*) bk->qs)[iqs];
            const int vi = (v >> (2 * i)) & 0x03030303;

            const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + bq8_offset + i;
            const int u_i = ((const int*) ya->qs)[iqs & (Q2K_R2_QI8_1 - 1)];
            const float d_y = (float) ya->d;

            const int sumi_d = flambeau_q2_k_r2_dp4a(vi, u_i, 0);
            const int sumi_m = flambeau_q2_k_r2_dp4a(m_packed, u_i, 0);

            acc += d_y * (d_sb * (float) sc_lo * (float) sumi_d
                          - dmin_sb * (float) sumi_m);
        }
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        mmvq_store<OutT>(dst, row, acc);
    }
}

extern "C" __global__ void flambeau_mmvq_q2_k_r2_dp4a_q8_1(
    const flambeau_block_q2_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_q2_k_r2_dp4a_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q2_k_r2_dp4a_q8_1_f16(
    const flambeau_block_q2_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_q2_k_r2_dp4a_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
