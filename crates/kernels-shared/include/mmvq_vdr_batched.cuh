#pragma once
// Batched (N = 1..8 activation columns) single-row MMVQ in llama.cpp's CUDA
// layout (`mul_mat_vec_q`, ncols_dst = N): one output row per block, the
// weight-block split per thread of `vec_dot_<type>_q8_1` (VDR), each thread
// decoding its weight ints once and dotting them against the N columns, N
// accumulators reduced together at the end. Columns are slot-major: column c
// of `y` starts at `y + c * blocks_per_col` Q8_1 blocks, `dst[c * n_rows + row]`.
// A trait gives the weight block type, the threads per weight block, the Q8_1
// blocks per weight block, and `step<N>` = one weight block's contribution.
// Backend-neutral: dp4a / byte_perm / the LUT and reduce come from the arch
// header and shared headers.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"
#include "mmvq_iq4_nl.cuh"

namespace mmvq_vdr {

// IQ4_XS, VDR 4: 8 threads per super-block, one 32-weight sub-block each.
struct Iq4Xs {
    typedef flambeau_block_iq4_xs block_t;
    static constexpr int TPB = 8;
    static constexpr int YPB = 8;
    template<int N, int R>
    static __device__ __forceinline__ void step(
        const block_t* const (&bk)[R], const flambeau_block_q8_1* y, size_t ys, int lane,
        float (&acc)[R][N])
    {
        int lo[R][4], hi[R][4];
        float d[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                iq4_nl_table_8((unsigned int) load_int_b4(bk[r]->qs, 4 * lane + j), lo[r][j], hi[r][j]);
            }
            d[r] = (float) bk[r]->d * (float) flambeau_iq4_xs_scale(lane, bk[r]->scales_h, bk[r]->scales_l);
        }
        #pragma unroll
        for (int c = 0; c < N; ++c) {
            const flambeau_block_q8_1* ya = y + c * ys + lane;
            int u[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) u[j] = load_int_b4(ya->qs, j);
            const float dy = (float) ya->d;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sumi = dp4a(lo[r][j], u[j], sumi);
                    sumi = dp4a(hi[r][j], u[j + 4], sumi);
                }
                acc[r][c] += d[r] * dy * (float) sumi;
            }
        }
    }
};

}  // namespace mmvq_vdr

// Reduce the R x N accumulators of a block (every thread holds partial sums)
// and write dst[c * n_rows + row0 + r] for the rows below n_rows. A block of
// at most one warp reduces with a THREADS-wide butterfly, no LDS.
template<int THREADS, int R, int N, class OutT>
static __device__ __forceinline__ void mmvq_reduce_store_rows(
    float (&acc)[R][N], OutT* __restrict__ dst, int row0, int n_rows)
{
    if constexpr (THREADS <= WARP_SIZE) {
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            #pragma unroll
            for (int c = 0; c < N; ++c) {
                #pragma unroll
                for (int off = THREADS / 2; off > 0; off >>= 1) {
                    acc[r][c] += __shfl_xor(acc[r][c], off, THREADS);
                }
            }
        }
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r < n_rows) {
                    #pragma unroll
                    for (int c = 0; c < N; ++c) mmvq_store<OutT>(dst, c * n_rows + row0 + r, acc[r][c]);
                }
            }
        }
    } else {
        constexpr int WARPS = THREADS / WARP_SIZE;
        const int lane = threadIdx.x % WARP_SIZE;
        const int warp = threadIdx.x / WARP_SIZE;
        __shared__ float s_part[WARPS][R * N];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            #pragma unroll
            for (int c = 0; c < N; ++c) {
                const float v = warp_reduce_sum(acc[r][c]);
                if (lane == 0) s_part[warp][r * N + c] = v;
            }
        }
        __syncthreads();
        const int i = threadIdx.x;
        if (i < R * N) {
            const int r = i / N;
            const int c = i % N;
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) v += s_part[w][i];
            if (row0 + r < n_rows) mmvq_store<OutT>(dst, c * n_rows + row0 + r, v);
        }
    }
}

// R output rows per block, THREADS threads, N columns: each weight block of
// the R rows is decoded once and each column read once for the R rows (the
// llama.cpp `rows_per_cuda_block` amortisation). `n_blocks_per_row` is in
// weight blocks; a column holds n_blocks_per_row * Tr::YPB Q8_1 blocks.
template<class Tr, int N, int THREADS, int R, class OutT>
static __device__ __forceinline__ void mmvq_vdr_batched_row(
    const typename Tr::block_t* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_blocks_per_row)
{
    constexpr int BLK_PER_ITER = THREADS / Tr::TPB;

    const int row0 = blockIdx.x * R;
    const int tid  = threadIdx.x;
    const int lane = tid % Tr::TPB;
    const size_t ys = (size_t) n_blocks_per_row * Tr::YPB;

    const typename Tr::block_t* xrow[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r < n_rows ? row0 + r : n_rows - 1;
        xrow[r] = x + (size_t) row * n_blocks_per_row;
    }

    float acc[R][N];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        #pragma unroll
        for (int c = 0; c < N; ++c) acc[r][c] = 0.0f;
    }

    for (int b = tid / Tr::TPB; b < n_blocks_per_row; b += BLK_PER_ITER) {
        const typename Tr::block_t* bk[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) bk[r] = xrow[r] + b;
        Tr::template step<N, R>(bk, y + (size_t) b * Tr::YPB, ys, lane, acc);
    }

    mmvq_reduce_store_rows<THREADS, R, N, OutT>(acc, dst, row0, n_rows);
}
