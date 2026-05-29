// mmvq_q8_0_row_tile_batched — Q8_0 MMVQ with row-tiled activation reuse.
// R=4 output rows per block share one LDS-resident Q8_1 activation strip
// across all N decode slots. Q8_0 sibling of `mmvq_q4_0_row_tile_batched`.
//
// Why: `mmvq_q8_0_batched` processes 1 row per block, so the same Q8_1
// activation HBM-fetches n_rows times across blocks. At Qwen3.6 GDN
// alpha/beta widths (n_rows = local_d_inner = 2048-4096 per rank), the
// activation traffic dominates — same structural fix as the Q4_0
// row-tile, just with Q8_0's i8 dequant path (no nibble unpack, no
// -8 * s correction).
//
// Row-tile design:
//   - Block: 256 threads = 4 wave64. Warp w in {0..3} owns
//     row = row_base + w.
//   - Outer iter strides 8 Q8_0 super-blocks (= 8 * 32 = 256 elements of K).
//   - LDS holds 8 * N Q8_1 blocks (qs[32], d, s) per outer iter, loaded by
//     the first N*8 <= 32 threads — 1 block per thread (8 i32 qs + d + s).
//   - Each warp's 64 lanes run dp4a across the 8 staged blocks against
//     its row's weight bytes, 1 i32 of weight per lane, 8 lanes per
//     block — activation comes from LDS, not HBM, and is reused
//     across 4 rows.
//
// Compile-time N in {2, 3, 4} kept for ABI parity with sibling kernels.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define RTQ80_WARPS 4
#define RTQ80_THREADS (RTQ80_WARPS * WARP_SIZE)
#define RTQ80_OUTER_BLOCKS 8

static __device__ __forceinline__ int flambeau_dp4a_rtq80(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template <int N>
__device__ __forceinline__ void flambeau_mmvq_q8_0_row_tile_batched_body(
    const flambeau_block_q8_0* __restrict__ w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int tid   = threadIdx.x;
    const int warp  = tid / WARP_SIZE;
    const int lane  = tid & (WARP_SIZE - 1);
    const int lane8 = lane & 7;
    const int bi_in = lane >> 3;

    const int row_base = blockIdx.x * RTQ80_WARPS;
    const int row      = row_base + warp;
    const bool row_ok  = row < n_rows;

    const flambeau_block_q8_0* w_row =
        row_ok ? (w + (size_t) row * n_blocks_per_row) : w;

    __shared__ int   s_y_qs[N][RTQ80_OUTER_BLOCKS][8];
    __shared__ float s_y_d [N][RTQ80_OUTER_BLOCKS];

    float acc[N];
    #pragma unroll
    for (int c = 0; c < N; ++c) acc[c] = 0.0f;

    for (int b_outer = 0; b_outer < n_blocks_per_row; b_outer += RTQ80_OUTER_BLOCKS) {
        if (tid < N * RTQ80_OUTER_BLOCKS) {
            const int c  = tid / RTQ80_OUTER_BLOCKS;
            const int bi = tid % RTQ80_OUTER_BLOCKS;
            const int block_idx = b_outer + bi;
            if (block_idx < n_blocks_per_row) {
                const flambeau_block_q8_1* py =
                    y + (size_t) c * n_blocks_per_row + block_idx;
                const int* py_i = (const int*) py->qs;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    s_y_qs[c][bi][j] = py_i[j];
                }
                s_y_d[c][bi] = (float) py->d;
            } else {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    s_y_qs[c][bi][j] = 0;
                }
                s_y_d[c][bi] = 0.0f;
            }
        }
        __syncthreads();

        if (row_ok && bi_in < RTQ80_OUTER_BLOCKS) {
            const int block_idx = b_outer + bi_in;
            if (block_idx < n_blocks_per_row) {
                const flambeau_block_q8_0* wbk = w_row + block_idx;
                const int w_v    = ((const int*) wbk->qs)[lane8];
                const float w_dx = (float) wbk->d;

                #pragma unroll
                for (int c = 0; c < N; ++c) {
                    const int yu = s_y_qs[c][bi_in][lane8];
                    const float d_y = s_y_d[c][bi_in];
                    const int sumi = flambeau_dp4a_rtq80(w_v, yu, 0);
                    acc[c] += (float) sumi * w_dx * d_y;
                }
            }
        }
        __syncthreads();
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < N; ++c) {
        const float v = gfx906_warp_reduce_sum(acc[c]);
        if (lane == 0) {
            dst[(size_t) c * n_rows + row] = v;
        }
    }
}

extern "C" __global__ __launch_bounds__(RTQ80_THREADS, 1)
void flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n2(
    const flambeau_block_q8_0* __restrict__ w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q8_0_row_tile_batched_body<2>(w, y, dst, n_rows, n_blocks_per_row);
}

extern "C" __global__ __launch_bounds__(RTQ80_THREADS, 1)
void flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n3(
    const flambeau_block_q8_0* __restrict__ w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q8_0_row_tile_batched_body<3>(w, y, dst, n_rows, n_blocks_per_row);
}

extern "C" __global__ __launch_bounds__(RTQ80_THREADS, 1)
void flambeau_mmvq_q8_0_row_tile_dp4a_q8_1_batched_n4(
    const flambeau_block_q8_0* __restrict__ w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q8_0_row_tile_batched_body<4>(w, y, dst, n_rows, n_blocks_per_row);
}
