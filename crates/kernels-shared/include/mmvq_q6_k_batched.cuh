#pragma once
// Templated Q6_K batched MMVQ (decode GEMV, N>1 slots). Scalar-FP32 super-block
// path (single-row `mmvq_q6_k` structure): 64 threads / one output row per
// block, `h` (0/1) the super-block half, `pos` (0..31) the element. Q6_K packs a
// 4-bit low nibble in `ql` + 2 high bits in `qh` + per-16-element i8 scales;
// each thread decodes 4 quants per super-block (`x = d·sc·(raw_q − 32)`) and
// multiplies them against the matching activation byte of each of the N slots
// (slot-major `y` / `dst`). Backend-neutral: the N-way reduce comes from the
// shared header; every load is a single byte or fp16 (no `load_int_b2`).
// `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<int THREADS, int N, class OutT = float>
static __device__ __forceinline__ void mmvq_q6_k_batched_row(
    const flambeau_block_q6_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_superblocks_per_row)
{
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane = threadIdx.x;
    const int h    = lane >> 5;
    const int pos  = lane & 31;
    const int lsub = pos >> 4;

    const flambeau_block_q6_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc[N];
    #pragma unroll
    for (int s = 0; s < N; ++s) acc[s] = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q6_K* bk = xrow + b;
        const float d = (float) bk->d;
        const uint8_t qh_byte = bk->qh[32 * h + pos];

        #pragma unroll
        for (int q_idx = 0; q_idx < 4; ++q_idx) {
            const int ql_off  = 64 * h + ((q_idx & 1) ? pos + 32 : pos);
            const int ql_byte = (int) bk->ql[ql_off];
            const int nibble  = (q_idx < 2) ? (ql_byte & 0x0F) : (ql_byte >> 4);
            const int qh_bits = (qh_byte >> (2 * q_idx)) & 0x3;
            const int raw_q   = (nibble | (qh_bits << 4)) - 32;

            const int sc = (int) bk->scales[8 * h + 2 * q_idx + lsub];
            const float x_val = d * (float) sc * (float) raw_q;

            const int y_block = h * 4 + q_idx;
            #pragma unroll
            for (int s = 0; s < N; ++s) {
                const flambeau_block_q8_1* ya =
                    y + (size_t) s * n_superblocks_per_row * 8 + b * 8 + y_block;
                acc[s] += x_val * (float) ya->d * (float) (int) ya->qs[pos];
            }
        }
    }

    mmvq_reduce_store_batched<THREADS, N, OutT>(acc, dst, row, n_rows);
}
