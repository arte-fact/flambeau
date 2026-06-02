// mmvq_q3_k_r2_dp4a — Q3_K dense MMVQ DP4A, multi-row r2 (wave64).
//
// r2 refactor of `mmvq_q3_k_dp4a.cu`: same per-element math (two DP4A
// calls per (iqs, i) — one for the 2-bit qs nibbles, one for the
// hmask high-bit subtraction — then sumi_l - sumi_h). The launch
// shape collapses from 256 threads single-row to 64 threads / 2 rows
// per block (lanes 0..31 own row R, lanes 32..63 own row R+1).
//
// Wins (matches Q4_K dp4a r2 pattern):
//   - Half the launches per matmul.
//   - Activation reuse: both rows in a wave fetch the same Q8_1 int32
//     so the second row hits L1 instead of HBM.
//   - No cross-warp reduce: replaces __shared__-based warp gather
//     with a register-resident half-warp DPP reduce.
//
// Each lane processes 2 (iqs, i) pairs per super-block via an outer
// j ∈ {0, 1} loop (j=0 → iqs ∈ 0..7, j=1 → iqs ∈ 8..15) so the same
// 16 × 4 = 64 pair coverage of the single-row kernel still happens,
// just split across half the lanes.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "mmvq_store.cuh"

#define Q3K_R2_QI3_K  16
#define Q3K_R2_QR3_K   4
#define Q3K_R2_QI8_1   8

static __device__ __forceinline__ int flambeau_q3_k_r2_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template<typename OutT>
__device__ void mmvq_q3_k_r2_dp4a_body(
    const flambeau_block_q3_K* __restrict__ x,
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

    const flambeau_block_q3_K* xrow = x + (size_t) row * n_superblocks_per_row;

    const int iqs_lo = lane_lo >> 2;               // 0..7
    const int i      = lane_lo & 3;                // 0..3

    float acc = 0.0f;

    for (int sb = 0; sb < n_superblocks_per_row; ++sb) {
        const flambeau_block_q3_K* bk = xrow + sb;
        const float d = (float) bk->d;

        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int iqs = iqs_lo + 8 * j;        // 0..15

            const int bq8_offset   = Q3K_R2_QR3_K * (iqs / (Q3K_R2_QI3_K / 2));   // 0 or 4
            const int scale_offset = iqs - (iqs & (Q3K_R2_QI8_1 - 1))
                                   + ((iqs & (Q3K_R2_QI8_1 - 1)) / (Q3K_R2_QI8_1 / 2));
            const int isc           = scale_offset + 2 * i;
            const int isc_low       = isc & 7;
            const int sc_shift_low  = 4 * (isc >> 3);
            const int isc_high      = isc & 3;
            const int sc_shift_high = 2 * (isc >> 2);

            const int sc_low  = (bk->scales[isc_low] >> sc_shift_low) & 0xF;
            const int sc_high = ((bk->scales[(Q3K_R2_QI3_K / 2) + isc_high]
                                  >> sc_shift_high) & 3) << 4;
            const int sc      = (sc_low | sc_high) - 32;

            const int vl      = ((const int*) bk->qs)[iqs];
            const int vh_full = ~((const int*) bk->hmask)[iqs & ((Q3K_R2_QI3_K / 2) - 1)];
            const int vh      = vh_full >> bq8_offset;

            const int vil = (vl >> (2 * i)) & 0x03030303;
            const int vih = ((vh >> i) << 2) & 0x04040404;

            const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + bq8_offset + i;
            const int u_i = ((const int*) ya->qs)[iqs & (Q3K_R2_QI8_1 - 1)];
            const float d_y = (float) ya->d;

            const int sumi_l = flambeau_q3_k_r2_dp4a(vil, u_i, 0);
            const int sumi_h = flambeau_q3_k_r2_dp4a(vih, u_i, 0);
            const int sumi   = sumi_l - sumi_h;

            acc += d * (float) sc * d_y * (float) sumi;
        }
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        mmvq_store<OutT>(dst, row, acc);
    }
}

extern "C" __global__ void flambeau_mmvq_q3_k_r2_dp4a_q8_1(
    const flambeau_block_q3_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_q3_k_r2_dp4a_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q3_k_r2_dp4a_q8_1_f16(
    const flambeau_block_q3_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_q3_k_r2_dp4a_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
