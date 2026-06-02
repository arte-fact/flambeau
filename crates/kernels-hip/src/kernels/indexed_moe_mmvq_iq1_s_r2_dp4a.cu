// indexed_moe_mmvq_iq1_s_r2_dp4a — IQ1_S MoE MMVQ with DP4A inner loop.
// Drop-in replacement for `indexed_moe_mmvq_iq1_s.cu` (scalar FP32).
// Same per-element math as dense `mmvq_iq1_s_dp4a.cu` plus the standard
// MoE wiring. IQ1_S has no sign mask (codebook entries are signed i8);
// the delta offset (±IQ1_DELTA per sub-block, picked by qh bit 15)
// contributes via a separate dp4a vs broadcast 0x01010101.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "iq_grid.cuh"

static __device__ __forceinline__ int flambeau_indexed_moe_iq1_s_dp4a(
    int a, int b, int c
) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_iq1_s_r2_dp4a_q8_1(
    const flambeau_block_iq1_s* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    float* __restrict__ dst,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row
) {
    const int row_pair = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;

    if (token >= n_tokens) return;

    const int lane    = threadIdx.x;
    const int row_hi  = lane >> 5;
    const int lane_lo = lane & 31;

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const flambeau_block_iq1_s* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    const int iqs   = lane_lo >> 2;    // 0..7 sub-block index (no x2)
    const int l_grp = lane_lo & 3;     // 0..3
    const int l0    = 2 * l_grp;

    float acc = 0.0f;

    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        const flambeau_block_iq1_s* bk = xrow + sb;
        const float d_sb = (float) bk->d;

        const int qh_u16 = (int) bk->qh[2 * iqs]
                         | ((int) bk->qh[2 * iqs + 1] << 8);

        const float dl    = d_sb * (2.0f * (float)((qh_u16 >> 12) & 7) + 1.0f);
        const float delta = (qh_u16 & 0x8000) ? -IQ1_DELTA : IQ1_DELTA;

        const int idx_lo = (int) bk->qs[4 * iqs + l_grp];
        const int high3  = (qh_u16 >> (3 * l_grp)) & 7;
        const int idx    = idx_lo | (high3 << 8);

        const uint64_t grid_u64 = IQ1S_GRID[idx];
        const int grid_lo = (int)(uint32_t) grid_u64;
        const int grid_hi = (int)(uint32_t)(grid_u64 >> 32);

        const flambeau_block_q8_1* ya = y_row + sb * 8 + iqs;
        const int u0 = ((const int*) ya->qs)[l0 + 0];
        const int u1 = ((const int*) ya->qs)[l0 + 1];
        const float d_y = (float) ya->d;

        int sumi = flambeau_indexed_moe_iq1_s_dp4a(grid_lo, u0, 0);
        sumi     = flambeau_indexed_moe_iq1_s_dp4a(grid_hi, u1, sumi);

        int sum_q8 = flambeau_indexed_moe_iq1_s_dp4a(0x01010101, u0, 0);
        sum_q8     = flambeau_indexed_moe_iq1_s_dp4a(0x01010101, u1, sum_q8);

        acc += dl * d_y * ((float) sumi + delta * (float) sum_q8);
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
