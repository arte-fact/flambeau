#pragma once
// Templated single-row IQ1_M MMVQ (decode GEMV). IQ1_M (1.75 bpw) extends IQ1_S:
// same `IQ1S_GRID` codebook, but the super-block scale is reassembled from four
// 16-bit `scales` words, with per-half-sub-block 3-bit scales and the ±IQ1_DELTA
// bias carried in `qh` nibbles. Ports llama.cpp's `vec_dot_iq1_m_q8_1`: 256
// threads, 32 per super-block; bias folds in as `delta·Σu`. Backend-neutral;
// only the Q8_1 int loads are misaligned (`load_int_b2`).
#include "quant_traits.cuh"
#include "iq_grid.cuh"
#include "mmvq_reduce.cuh"

// Reassemble the fp16 super-block scale from the top nibble of each of the four
// 16-bit `scales` words.
static __device__ __forceinline__ float iq1_m_reassemble_d(const uint8_t* scales) {
    const int sc0 = (int) scales[0] | ((int) scales[1] << 8);
    const int sc1 = (int) scales[2] | ((int) scales[3] << 8);
    const int sc2 = (int) scales[4] | ((int) scales[5] << 8);
    const int sc3 = (int) scales[6] | ((int) scales[7] << 8);
    const int d_bits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000);
    const fb_fp16_t d_fp16 = *reinterpret_cast<const fb_fp16_t*>(&d_bits);
    return (float) d_fp16;
}

template<class OutT>
static __device__ __forceinline__ void mmvq_iq1_m_dp4a_row(
    const flambeau_block_iq1_m* __restrict__ x,
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
    const int iqs        = lane_in_sb >> 2;      // 0..7
    const int l_grp      = lane_in_sb & 3;       // 0..3
    const int l0         = 2 * l_grp;

    const flambeau_block_iq1_m* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;
    for (int sb = sb_in_iter; sb < n_superblocks_per_row; sb += SB_PER_ITER) {
        const flambeau_block_iq1_m* bk = xrow + sb;

        const float d_super = iq1_m_reassemble_d(bk->scales);

        const int sc_word = (int) bk->scales[2 * (iqs >> 1)] | ((int) bk->scales[2 * (iqs >> 1) + 1] << 8);
        const int shift1   = 6 * (iqs & 1);
        const int dl_scale = (l_grp < 2) ? ((sc_word >> shift1) & 7) : ((sc_word >> (shift1 + 3)) & 7);
        const float dl = d_super * (2.0f * (float) dl_scale + 1.0f);

        const int qh_pick = (l_grp < 2) ? (2 * iqs) : (2 * iqs + 1);
        const unsigned int qh_byte = bk->qh[qh_pick];
        const int shift_idx = 8 - 4 * (l_grp & 1);
        const int idx = (int) bk->qs[4 * iqs + l_grp] | (((int) qh_byte << shift_idx) & 0x700);
        const int delta_bit = (l_grp & 1) == 0 ? 0x08 : 0x80;
        const float delta = (qh_byte & delta_bit) ? -IQ1_DELTA : IQ1_DELTA;

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
