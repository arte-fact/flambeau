// Q6_K batched per-N MMVQ. Single-row Q6_K MMVQ structure (64-thread
// wave64, per-element decode `x = d·sc·(raw_q − 32)`) with an inner
// N-slot loop.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q6K_BATCHED_THREADS 64

template <int N>
__device__ __forceinline__ void flambeau_mmvq_q6_k_batched_body(
    const flambeau_block_q6_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
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
            const int ql_off = 64 * h + ((q_idx & 1) ? pos + 32 : pos);
            const int ql_byte = (int) bk->ql[ql_off];
            const int nibble = (q_idx < 2) ? (ql_byte & 0x0F) : (ql_byte >> 4);
            const int qh_bits = (qh_byte >> (2 * q_idx)) & 0x3;
            const int raw_q = (nibble | (qh_bits << 4)) - 32;

            const int scale_idx = 8 * h + 2 * q_idx + lsub;
            const int sc = (int) bk->scales[scale_idx];

            const float x_val = d * (float) sc * (float) raw_q;

            const int y_block = h * 4 + q_idx;
            #pragma unroll
            for (int s = 0; s < N; ++s) {
                const flambeau_block_q8_1* ya =
                    y + (size_t) s * n_superblocks_per_row * 8 + b * 8 + y_block;
                const float d_y = (float) ya->d;
                const int   qi  = (int) ya->qs[pos];
                acc[s] += x_val * d_y * (float) qi;
            }
        }
    }

    #pragma unroll
    for (int s = 0; s < N; ++s) {
        acc[s] = gfx906_warp_reduce_sum(acc[s]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int s = 0; s < N; ++s) {
            dst[(size_t) s * n_rows + row] = acc[s];
        }
    }
}

extern "C" __global__ void flambeau_mmvq_q6_k_q8_1_batched_n2(
    const flambeau_block_q6_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    flambeau_mmvq_q6_k_batched_body<2>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q6_k_q8_1_batched_n3(
    const flambeau_block_q6_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    flambeau_mmvq_q6_k_batched_body<3>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q6_k_q8_1_batched_n4(
    const flambeau_block_q6_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    flambeau_mmvq_q6_k_batched_body<4>(x, y, dst, n_rows, n_superblocks_per_row);
}
