#pragma once
// Templated F16-weight × Q8_1-activation MMVQ (used for F16 layers in
// UD-Q8_K_XL GGUFs). 256-thread block, one weight row per block; lane_in_grp =
// tid&3 picks an int32-pair slot, block_idx = tid>>2 strides the Q8_1 blocks.
// Weights are F16 so dp4a doesn't apply — plain F32 FMA over 8 elements/thread,
// then the warp-size-agnostic `mmvq_reduce_store`. Backend-neutral: Q8_1 int
// loads through `load_int_b2`. `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<int THREADS>
static __device__ __forceinline__ float flambeau_f16_q8_1_acc(
    const fb_fp16_t* __restrict__ xrow,
    const flambeau_block_q8_1* __restrict__ y_row,
    int n_blocks_per_row)
{
    constexpr int THREADS_PER_QBLK = 4;
    constexpr int BLOCKS_PER_ITER  = THREADS / THREADS_PER_QBLK;

    const int tid         = threadIdx.x;
    const int lane_in_grp = tid & 3;
    const int block_idx   = tid >> 2;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += BLOCKS_PER_ITER) {
        const flambeau_block_q8_1* by = y_row + b;
        const int elem_off = lane_in_grp * 8;

        const int yi0 = load_int_b2(by->qs, lane_in_grp * 2 + 0);
        const int yi1 = load_int_b2(by->qs, lane_in_grp * 2 + 1);
        const float d_y = (float) by->d;

        signed char q[8];
        q[0] = (signed char) ((yi0      ) & 0xff);
        q[1] = (signed char) ((yi0 >>  8) & 0xff);
        q[2] = (signed char) ((yi0 >> 16) & 0xff);
        q[3] = (signed char) ((yi0 >> 24) & 0xff);
        q[4] = (signed char) ((yi1      ) & 0xff);
        q[5] = (signed char) ((yi1 >>  8) & 0xff);
        q[6] = (signed char) ((yi1 >> 16) & 0xff);
        q[7] = (signed char) ((yi1 >> 24) & 0xff);

        const size_t x_off = (size_t) b * 32 + elem_off;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float xv = (float) xrow[x_off + j];
            acc += xv * (float) q[j] * d_y;
        }
    }
    return acc;
}

template<int THREADS, class OutT = float>
static __device__ __forceinline__ void mmvq_f16_q8_1_row(
    const fb_fp16_t* __restrict__ x_f16,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_blocks_per_row)
{
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const size_t n_cols = (size_t) n_blocks_per_row * 32;
    const float acc = flambeau_f16_q8_1_acc<THREADS>(
        x_f16 + (size_t) row * n_cols, y, n_blocks_per_row);
    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}
