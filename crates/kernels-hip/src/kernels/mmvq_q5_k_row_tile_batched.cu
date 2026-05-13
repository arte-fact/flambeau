// mmvq_q5_k_row_tile_batched — Q5_K MMVQ with row-tiled activation reuse.
// R=8 output rows per block share one LDS-resident Q8_1 activation strip
// across all N decode slots.
//
// Why: `mmvq_q5_k_r2_batched` processes 2 rows per block via half-warp
// split, but the same Q8_1 activation is HBM-fetched again every (sub,
// col) iteration — at n_rows in the thousands, each activation byte is
// re-fetched 1000s of times across blocks. gfx906 L2 (4 MB) can't hold
// the full activation strip when n_rows ≫ N.
//
// Row-tile design:
//   - Block: 256 threads = 4 wave64. Each wave splits into two half-warps;
//     half-warp h owns row = row_base + (warp * 2 + row_hi). R=8 rows per
//     block.
//   - Outer iter strides 1 Q5_K super-block (= 256 elements of K, 8 Q8_1
//     sub-blocks).
//   - LDS holds 8 × N Q8_1 sub-blocks (qs[32] + d + s) per outer iter,
//     loaded by the first N×8 ≤ 32 threads — 1 sub-block per thread.
//   - Each half-warp's 32 lanes then walk all 8 sub-blocks within the
//     super-block against activations from LDS instead of HBM, per the
//     `mmvq_q5_k_r2_batched` arithmetic.
//
// Compile-time N ∈ {2, 3, 4} kept for ABI parity with K3.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define RTQ5_R 8
#define RTQ5_THREADS 256
#define RTQ5_WARPS 4

template <int N>
__device__ __forceinline__ void flambeau_mmvq_q5_k_row_tile_batched_body(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_sb_per_row
) {
    const int tid     = threadIdx.x;
    const int warp    = tid / WARP_SIZE;
    const int lane    = tid & (WARP_SIZE - 1);
    const int row_hi  = lane >> 5;
    const int lane_lo = lane & 31;

    const int row_in_tile = warp * 2 + row_hi;
    const int row         = blockIdx.x * RTQ5_R + row_in_tile;
    const bool row_ok     = row < n_rows;

    const flambeau_block_q5_K* xrow =
        row_ok ? (x + (size_t) row * n_sb_per_row) : x;

    __shared__ int   s_y_qs[N][8][8];
    __shared__ float s_y_d [N][8];
    __shared__ float s_y_s [N][8];

    float acc[N];
    #pragma unroll
    for (int c = 0; c < N; ++c) acc[c] = 0.0f;

    const int act_blocks_per_row = n_sb_per_row * 8;

    for (int b = 0; b < n_sb_per_row; ++b) {
        if (tid < N * 8) {
            const int c   = tid >> 3;
            const int sub = tid & 7;
            const flambeau_block_q8_1* py =
                y + (size_t) c * act_blocks_per_row + b * 8 + sub;
            const int* py_i = (const int*) py->qs;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                s_y_qs[c][sub][j] = py_i[j];
            }
            s_y_d[c][sub] = (float) py->d;
            s_y_s[c][sub] = (float) py->s;
        }
        __syncthreads();

        if (row_ok) {
            const flambeau_block_q5_K* bk = xrow + b;
            const float d_w       = (float) bk->d;
            const float dmin_w    = (float) bk->dmin;
            const uint8_t qh_byte = bk->qh[lane_lo];

            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                uint8_t sc = 0, m_v = 0;
                flambeau_q4k_scale_min(s, bk->scales, &sc, &m_v);

                const int byte_idx = (s >> 1) * 32 + lane_lo;
                const int byte_v   = (int) bk->qs[byte_idx];
                const int raw_q4   = (s & 1) ? (byte_v >> 4) : (byte_v & 0x0F);
                const int mask     = 1 << s;
                const int raw_q    = raw_q4 + ((qh_byte & mask) ? 16 : 0);

                const float x_val =
                    d_w * (float) sc * (float) raw_q - dmin_w * (float) m_v;

                #pragma unroll
                for (int c = 0; c < N; ++c) {
                    const int qi    = (int)((const int8_t*) s_y_qs[c][s])[lane_lo];
                    const float d_y = s_y_d[c][s];
                    const float y_val = d_y * (float) qi;
                    acc[c] += x_val * y_val;
                }
            }
        }
        __syncthreads();
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < N; ++c) {
        acc[c] = gfx906_half_warp_reduce_sum(acc[c]);
        if (lane_lo == 0) {
            dst[(size_t) c * n_rows + row] = acc[c];
        }
    }
}

extern "C" __global__ __launch_bounds__(RTQ5_THREADS, 1)
void flambeau_mmvq_q5_k_row_tile_q8_1_batched_n2(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_sb_per_row
) {
    flambeau_mmvq_q5_k_row_tile_batched_body<2>(x, y, dst, n_rows, n_sb_per_row);
}

extern "C" __global__ __launch_bounds__(RTQ5_THREADS, 1)
void flambeau_mmvq_q5_k_row_tile_q8_1_batched_n3(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_sb_per_row
) {
    flambeau_mmvq_q5_k_row_tile_batched_body<3>(x, y, dst, n_rows, n_sb_per_row);
}

extern "C" __global__ __launch_bounds__(RTQ5_THREADS, 1)
void flambeau_mmvq_q5_k_row_tile_q8_1_batched_n4(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_sb_per_row
) {
    flambeau_mmvq_q5_k_row_tile_batched_body<4>(x, y, dst, n_rows, n_sb_per_row);
}
