#pragma once
// Templated single-row IQ1_S MMVQ (decode GEMV). IQ1_S (1.5625 bpw): each
// element is an 11-bit index into `IQ1S_GRID` (uint64 of 8 signed-i8
// magnitudes), with a per-sub-block 3-bit scale and a ±IQ1_DELTA bias both
// carried in `qh`. Ports llama.cpp's `vec_dot_iq1_s_q8_1`: 256 threads, 32 per
// super-block; the bias folds in as `delta·Σu` (one extra dp4a vs 0x01010101).
// Backend-neutral; only the Q8_1 int loads are misaligned (`load_int_b2`).
#include "quant_traits.cuh"
#include "iq_grid.cuh"
#include "mmvq_reduce.cuh"

template<class OutT>
static __device__ __forceinline__ void mmvq_iq1_s_dp4a_row(
    const flambeau_block_iq1_s* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_superblocks_per_row)
{
    constexpr int THREADS        = 256;
    constexpr int THREADS_PER_SB = 32;
    constexpr int SB_PER_ITER    = THREADS / THREADS_PER_SB;

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid        = threadIdx.x;
    const int sb_in_iter = tid >> 5;
    const int lane_in_sb = tid & 31;
    const int iqs        = lane_in_sb >> 2;      // 0..7 — sub-block
    const int l_grp      = lane_in_sb & 3;       // 0..3
    const int l0         = 2 * l_grp;

    const flambeau_block_iq1_s* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;
    for (int sb = sb_in_iter; sb < n_superblocks_per_row; sb += SB_PER_ITER) {
        const flambeau_block_iq1_s* bk = xrow + sb;

        const float d_sb = (float) bk->d;
        const int qh_u16 = (int) bk->qh[2 * iqs] | ((int) bk->qh[2 * iqs + 1] << 8);

        const float dl    = d_sb * (2.0f * (float) ((qh_u16 >> 12) & 7) + 1.0f);
        const float delta = (qh_u16 & 0x8000) ? -IQ1_DELTA : IQ1_DELTA;

        const int idx_lo = (int) bk->qs[4 * iqs + l_grp];
        const int high3  = (qh_u16 >> (3 * l_grp)) & 7;
        const int idx    = idx_lo | (high3 << 8);

        const uint64_t grid_u64 = IQ1S_GRID[idx];
        const int grid_lo = (int) (uint32_t) grid_u64;
        const int grid_hi = (int) (uint32_t) (grid_u64 >> 32);

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + iqs;
        const int u0 = load_int_b2(ya->qs, l0 + 0);
        const int u1 = load_int_b2(ya->qs, l0 + 1);
        const float d_y = (float) ya->d;

        int sumi = dp4a(grid_lo, u0, 0);
        sumi     = dp4a(grid_hi, u1, sumi);
        int sum_q8 = dp4a(0x01010101, u0, 0);
        sum_q8     = dp4a(0x01010101, u1, sum_q8);

        acc += dl * d_y * ((float) sumi + delta * (float) sum_q8);
    }

    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}
