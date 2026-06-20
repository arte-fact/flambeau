#pragma once
// Batched dp4a Q6_K MMVQ (decode GEMV, N slots). The single-row
// mmvq_q6_k_dp4a_row decode (vl/vh/scales/d -> raw0/raw1 dp4a operands) is
// hoisted out of an N-slot inner loop, so one Q6_K weight read feeds N
// activation columns. The (raw-32)·y bias uses the dp4a identity
// raw·y - 32·Σy. Misaligned super-block int loads go through load_int_b2.
// `OutT` is the destination dtype (F32 or F16-dst).
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<int N, class OutT = float>
static __device__ __forceinline__ void mmvq_q6_k_dp4a_batched_row(
    const flambeau_block_q6_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_superblocks_per_row)
{
    constexpr int QR = 2;
    constexpr int QI = 32;
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int lane     = threadIdx.x;
    const int iqs      = lane & 31;
    const int super_hi = lane >> 5;
    const int bq8_offset   = QR * (iqs / (QI / 2)) * 2 + (iqs % (QI / 2)) / (QI / 4);
    const int scale_offset = (QI / 4) * (iqs / (QI / 2)) + (iqs % (QI / 2)) / (QI / 8);
    const int vh_shift     = 2 * ((iqs % (QI / 2)) / (QI / 4));
    const int qh_int_idx   = (QI / 4) * (iqs / (QI / 2)) + iqs % (QI / 4);
    const flambeau_block_q6_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc[N];
    #pragma unroll
    for (int s = 0; s < N; ++s) acc[s] = 0.0f;

    for (int b_base = 0; b_base < n_superblocks_per_row; b_base += 2) {
        const int b = b_base + super_hi;
        if (b >= n_superblocks_per_row) continue;
        const flambeau_block_q6_K* bk = xrow + b;

        const int vl = load_int_b2((const int8_t*) bk->ql, iqs);
        const int vh = load_int_b2((const int8_t*) bk->qh, qh_int_idx) >> vh_shift;
        const int8_t* sc_ptr = bk->scales + scale_offset;
        const int sc0  = (int) sc_ptr[0];
        const int sc1  = (int) sc_ptr[4];
        const int raw0 = ((vl >> 0) & 0x0F0F0F0F) | (((vh >> 0) << 4) & 0x30303030);
        const int raw1 = ((vl >> 4) & 0x0F0F0F0F) | (((vh >> 4) << 4) & 0x30303030);
        const float bkd = (float) bk->d;

        #pragma unroll
        for (int s = 0; s < N; ++s) {
            const flambeau_block_q8_1* ya0 =
                y + (size_t) s * n_superblocks_per_row * 8 + (b * 8 + bq8_offset + 0);
            const flambeau_block_q8_1* ya1 =
                y + (size_t) s * n_superblocks_per_row * 8 + (b * 8 + bq8_offset + 2);
            const int u0 = load_int_b2(ya0->qs, iqs % 8);
            const int u1 = load_int_b2(ya1->qs, iqs % 8);
            const float sumf =
                (float) ya0->d * ((float) (dp4a(raw0, u0, 0) - 32 * dp4a(0x01010101, u0, 0)) * (float) sc0)
              + (float) ya1->d * ((float) (dp4a(raw1, u1, 0) - 32 * dp4a(0x01010101, u1, 0)) * (float) sc1);
            acc[s] += bkd * sumf;
        }
    }
    mmvq_reduce_store_batched<64, N, OutT>(acc, dst, row, n_rows);
}
