// mmvq_q5_k_r2_batched — Q5_K MMVQ × N activation cols, r2 multi-row.
//
// K3 — Q5_K batched MMVQ with compile-time N specialization. Extends
// `mmvq_q5_k_r2` (P29 r2 multi-row pattern) with an inner activation-
// column loop bounded by compile-time N ∈ {2, 3, 4}, so each block
// reads one Q5_K super-block ONCE and applies it across N activation
// columns. Same amortization lever as `mmvq_q4_0_batched`.
//
// Wave structure (unchanged from r2):
//   - 64 threads / block = 1 wave64
//   - lane >> 5 = row_hi ∈ {0, 1} — which of the 2 output rows
//   - lane & 31 = lane_lo — element index within the half-warp
//   - block.x = row_pair index → handles rows R+0 and R+1
//
// Per sub-block s (8 of them per Q5_K super-block):
//   1. Decode Q5_K weight element once into `x_val` (depends on s + lane_lo,
//      shared across the N activation cols).
//   2. For each col c in 0..N: load activation byte from y_c[s][lane_lo],
//      multiply by x_val, accumulate into acc[c].
//
// Output layout: `dst[N, n_rows]` slot-major F32 — matches `qmatmul` ABI
// and the Q4_0 batched output shape.

#include "block_quant.cuh"
#include "gfx906.cuh"

template <int N>
__device__ __forceinline__ void flambeau_mmvq_q5_k_r2_batched_body(
    const flambeau_block_q5_K* __restrict__ x,    // [n_rows, n_superblocks_per_row]
    const flambeau_block_q8_1* __restrict__ y,    // [N, n_superblocks_per_row * 8]
    float* __restrict__ dst,                       // [N, n_rows]
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row_pair = blockIdx.x;
    const int lane     = threadIdx.x;
    const int row_hi   = lane >> 5;
    const int lane_lo  = lane & 31;

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const flambeau_block_q5_K* xrow = x + (size_t) row * n_superblocks_per_row;

    // Per-column accumulators. Compile-time N → register-resident.
    float acc[N];
    #pragma unroll
    for (int c = 0; c < N; ++c) {
        acc[c] = 0.0f;
    }

    // Per-column activation base pointer. Each column's Q8_1 row stride
    // is `n_superblocks_per_row * 8` blocks (since Q5_K super-block
    // covers 8 Q8_1 blocks worth of elements).
    const int act_blocks_per_row = n_superblocks_per_row * 8;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q5_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;
        const uint8_t qh_byte = bk->qh[lane_lo];

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint8_t sc = 0, m = 0;
            flambeau_q4k_scale_min(s, bk->scales, &sc, &m);

            const int byte_idx = (s >> 1) * 32 + lane_lo;
            const int byte_v   = (int) bk->qs[byte_idx];
            const int raw_q4   = (s & 1) ? (byte_v >> 4) : (byte_v & 0x0F);

            const int mask = 1 << s;
            const int raw_q = raw_q4 + ((qh_byte & mask) ? 16 : 0);

            // Decode Q5_K weight element ONCE per sub-block — amortized
            // across the N activation columns below.
            const float x_val = d * (float) sc * (float) raw_q - dmin * (float) m;

            // Per-column inner loop, fully unrolled at compile time.
            #pragma unroll
            for (int c = 0; c < N; ++c) {
                const flambeau_block_q8_1* ya =
                    y + (size_t) c * act_blocks_per_row + b * 8 + s;
                const float d_y = (float) ya->d;
                const int   qi  = (int) ya->qs[lane_lo];
                const float y_val = d_y * (float) qi;
                acc[c] += x_val * y_val;
            }
        }
    }

    // Per-column half-warp reduce + write. Each half-warp owns one row;
    // lane_lo == 0 writes the row's column-c output for each c.
    #pragma unroll
    for (int c = 0; c < N; ++c) {
        acc[c] = gfx906_half_warp_reduce_sum(acc[c]);
        if (lane_lo == 0) {
            dst[(size_t) c * n_rows + row] = acc[c];
        }
    }
}

extern "C" __global__ void flambeau_mmvq_q5_k_r2_q8_1_batched_n2(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    flambeau_mmvq_q5_k_r2_batched_body<2>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q5_k_r2_q8_1_batched_n3(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    flambeau_mmvq_q5_k_r2_batched_body<3>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q5_k_r2_q8_1_batched_n4(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    flambeau_mmvq_q5_k_r2_batched_body<4>(x, y, dst, n_rows, n_superblocks_per_row);
}
