// mmvq_q4_0_row_tile_batched — Q4_0 MMVQ with row-tiled activation reuse.
// R=4 output rows per block share one LDS-resident Q8_1 activation strip
// across all N decode slots. Single weight matrix variant of
// `mmvq_q4_0_gate_up_row_tile_batched` (drops the fused-up output).
//
// Why: `mmvq_q4_0_batched` processes 1 row per block, so the same Q8_1
// activation HBM-fetches n_rows times across blocks. gfx906 L2 (4 MB)
// can't hold a many-block activation strip when n_rows is large (~14k
// for GDN ssm_out at intermediate=14336, ~5k for attn_output_proj at
// hidden=5120), and the activation traffic dominated kernel time at
// n_rows >> N. Same structural fix as the gate+up variant — R rows
// share one HBM activation fetch.
//
// Row-tile design:
//   - Block: 256 threads = 4 wave64. Warp w in {0..3} owns row = row_base + w.
//   - Outer iter strides 16 Q4_0 super-blocks (= 16 * 32 = 512 elements of K).
//   - LDS holds 16 * N Q8_1 blocks (qs[32], d, s) per outer iter, loaded by
//     the first N*16 <= 64 threads — 1 block per thread (8 i32 qs + d + s).
//   - Each warp's 64 lanes then run dp4a across the 16 staged blocks against
//     its row's weight bytes, 1 i32 of weight per lane, 4 lanes per
//     block — activation comes from LDS, not HBM, and is reused across 4
//     rows.
//
// Compile-time N in {2, 3, 4} kept for ABI parity with sibling kernels.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define RTQ40_WARPS 4
#define RTQ40_THREADS (RTQ40_WARPS * WARP_SIZE)
#define RTQ40_OUTER_BLOCKS 16

static __device__ __forceinline__ int flambeau_dp4a_rtq40(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template <int N>
__device__ __forceinline__ void flambeau_mmvq_q4_0_row_tile_batched_body(
    const flambeau_block_q4_0* __restrict__ w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int tid   = threadIdx.x;
    const int warp  = tid / WARP_SIZE;
    const int lane  = tid & (WARP_SIZE - 1);
    const int lane4 = lane & 3;
    const int bi_in = lane >> 2;

    const int row_base = blockIdx.x * RTQ40_WARPS;
    const int row      = row_base + warp;
    const bool row_ok  = row < n_rows;

    const flambeau_block_q4_0* w_row =
        row_ok ? (w + (size_t) row * n_blocks_per_row) : w;

    __shared__ int   s_y_qs[N][RTQ40_OUTER_BLOCKS][8];
    __shared__ float s_y_d [N][RTQ40_OUTER_BLOCKS];
    __shared__ float s_y_s [N][RTQ40_OUTER_BLOCKS];

    float acc[N];
    #pragma unroll
    for (int c = 0; c < N; ++c) acc[c] = 0.0f;

    for (int b_outer = 0; b_outer < n_blocks_per_row; b_outer += RTQ40_OUTER_BLOCKS) {
        if (tid < N * RTQ40_OUTER_BLOCKS) {
            const int c  = tid / RTQ40_OUTER_BLOCKS;
            const int bi = tid % RTQ40_OUTER_BLOCKS;
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
                s_y_s[c][bi] = (float) py->s;
            } else {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    s_y_qs[c][bi][j] = 0;
                }
                s_y_d[c][bi] = 0.0f;
                s_y_s[c][bi] = 0.0f;
            }
        }
        __syncthreads();

        if (row_ok && bi_in < RTQ40_OUTER_BLOCKS) {
            const int block_idx = b_outer + bi_in;
            if (block_idx < n_blocks_per_row) {
                const flambeau_block_q4_0* wbk = w_row + block_idx;
                const int w_v = ((const int*) wbk->qs)[lane4];
                const int w_vi_lo = (w_v >> 0) & 0x0F0F0F0F;
                const int w_vi_hi = (w_v >> 4) & 0x0F0F0F0F;
                const float w_dx  = (float) wbk->d;

                #pragma unroll
                for (int c = 0; c < N; ++c) {
                    const int yu_lo = s_y_qs[c][bi_in][lane4];
                    const int yu_hi = s_y_qs[c][bi_in][lane4 + 4];
                    const float d_y = s_y_d[c][bi_in];
                    const float s_y = s_y_s[c][bi_in];

                    int sumi = 0;
                    sumi = flambeau_dp4a_rtq40(w_vi_lo, yu_lo, sumi);
                    sumi = flambeau_dp4a_rtq40(w_vi_hi, yu_hi, sumi);
                    acc[c] += sumi * (w_dx * d_y) - 8.0f * w_dx * s_y * 0.25f;
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

extern "C" __global__ __launch_bounds__(RTQ40_THREADS, 1)
void flambeau_mmvq_q4_0_row_tile_dp4a_q8_1_batched_n2(
    const flambeau_block_q4_0* __restrict__ w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q4_0_row_tile_batched_body<2>(w, y, dst, n_rows, n_blocks_per_row);
}

extern "C" __global__ __launch_bounds__(RTQ40_THREADS, 1)
void flambeau_mmvq_q4_0_row_tile_dp4a_q8_1_batched_n3(
    const flambeau_block_q4_0* __restrict__ w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q4_0_row_tile_batched_body<3>(w, y, dst, n_rows, n_blocks_per_row);
}

extern "C" __global__ __launch_bounds__(RTQ40_THREADS, 1)
void flambeau_mmvq_q4_0_row_tile_dp4a_q8_1_batched_n4(
    const flambeau_block_q4_0* __restrict__ w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q4_0_row_tile_batched_body<4>(w, y, dst, n_rows, n_blocks_per_row);
}
