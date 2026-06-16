#pragma once
// Templated single-row Q6_K MMVQ (decode GEMV). Q6_K is a 256-element
// super-block: 4-bit `ql` + 2-bit `qh` reassemble a 6-bit weight, biased by
// -32, with 16 signed int8 sub-block scales and one fp16 super-block scale.
// Ports llama.cpp's `vec_dot_q6_K_q8_1_impl_mmvq`: 64 threads per row, each
// lane owns one of 32 `iqs` positions across a stride-2 super-block pair. The
// (raw-32)·y bias uses the DP4A identity raw·y - 32·Σy (one extra
// `dp4a(0x01010101, u, 0)` per term) — gfx906 has no per-byte saturating
// subtract. Backend-neutral: `dp4a` / the reduce come from the arch header +
// the shared helpers; misaligned super-block int loads go through
// `load_int_b2` (a 4-byte read of `ql`/`qh` faults on NVIDIA for odd
// super-blocks). `OutT` is the destination dtype (F32 or F16-dst).
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<class OutT>
static __device__ __forceinline__ void mmvq_q6_k_dp4a_row(
    const flambeau_block_q6_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_superblocks_per_row)
{
    constexpr int QR = 2;
    constexpr int QI = 32;

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane     = threadIdx.x;             // 0..63
    const int iqs      = lane & 31;               // iqs position within super-block
    const int super_hi = lane >> 5;               // 0/1 — stride-2 super-block offset

    const int bq8_offset   = QR * (iqs / (QI / 2)) * 2 + (iqs % (QI / 2)) / (QI / 4);
    const int scale_offset = (QI / 4) * (iqs / (QI / 2)) + (iqs % (QI / 2)) / (QI / 8);
    const int vh_shift     = 2 * ((iqs % (QI / 2)) / (QI / 4));
    const int qh_int_idx   = (QI / 4) * (iqs / (QI / 2)) + iqs % (QI / 4);

    const flambeau_block_q6_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;
    for (int b_base = 0; b_base < n_superblocks_per_row; b_base += 2) {
        const int b = b_base + super_hi;
        if (b >= n_superblocks_per_row) continue;

        const flambeau_block_q6_K* bk = xrow + b;

        const int vl = load_int_b2((const int8_t*) bk->ql, iqs);
        const int vh = load_int_b2((const int8_t*) bk->qh, qh_int_idx) >> vh_shift;
        const int8_t* sc_ptr = bk->scales + scale_offset;

        const flambeau_block_q8_1* ya0 = y + (b * 8 + bq8_offset + 0);
        const flambeau_block_q8_1* ya1 = y + (b * 8 + bq8_offset + 2);
        const int u0 = load_int_b2(ya0->qs, iqs % 8);
        const int u1 = load_int_b2(ya1->qs, iqs % 8);
        const float d8_0 = (float) ya0->d;
        const float d8_1 = (float) ya1->d;

        float sumf = 0.0f;
        {
            const int sc   = (int) sc_ptr[0];
            const int vil  = (vl >> 0) & 0x0F0F0F0F;
            const int vih  = ((vh >> 0) << 4) & 0x30303030;
            const int raw  = vil | vih;                      // [0, 63] per byte
            const int dot  = dp4a(raw, u0, 0);
            const int sumy = dp4a(0x01010101, u0, 0);
            sumf += d8_0 * ((float) (dot - 32 * sumy) * (float) sc);
        }
        {
            const int sc   = (int) sc_ptr[4];
            const int vil  = (vl >> 4) & 0x0F0F0F0F;
            const int vih  = ((vh >> 4) << 4) & 0x30303030;
            const int raw  = vil | vih;
            const int dot  = dp4a(raw, u1, 0);
            const int sumy = dp4a(0x01010101, u1, 0);
            sumf += d8_1 * ((float) (dot - 32 * sumy) * (float) sc);
        }

        acc += (float) bk->d * sumf;
    }

    mmvq_reduce_store<64, OutT>(acc, dst, row);
}
