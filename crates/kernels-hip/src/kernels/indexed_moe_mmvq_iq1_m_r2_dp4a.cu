// indexed_moe_mmvq_iq1_m_r2_dp4a — IQ1_M MoE MMVQ with DP4A inner loop.
// Drop-in replacement for `indexed_moe_mmvq_iq1_m.cu` (scalar FP32).
// Same per-element math as dense `mmvq_iq1_m_dp4a.cu` plus the standard
// MoE wiring. IQ1_M extends IQ1_S with d reassembled from 4 u16 scale
// words, two 3-bit scales per sub-block, per-l_grp delta sign.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "iq_grid.cuh"

static __device__ __forceinline__ int flambeau_indexed_moe_iq1_m_dp4a(
    int a, int b, int c
) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ float flambeau_indexed_moe_iq1_m_reassemble_d(
    const uint8_t* __restrict__ scales
) {
    const int sc0 = (int) scales[0] | ((int) scales[1] << 8);
    const int sc1 = (int) scales[2] | ((int) scales[3] << 8);
    const int sc2 = (int) scales[4] | ((int) scales[5] << 8);
    const int sc3 = (int) scales[6] | ((int) scales[7] << 8);
    const int d_bits = (sc0 >> 12)
                     | ((sc1 >> 8) & 0x00F0)
                     | ((sc2 >> 4) & 0x0F00)
                     | (sc3 & 0xF000);
    fb_fp16_t d_fp16 = *reinterpret_cast<const fb_fp16_t*>(&d_bits);
    return (float) d_fp16;
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_iq1_m_r2_dp4a_q8_1(
    const flambeau_block_iq1_m* __restrict__ x,
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

    const flambeau_block_iq1_m* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    const int iqs   = lane_lo >> 2;     // 0..7
    const int l_grp = lane_lo & 3;      // 0..3
    const int l0    = 2 * l_grp;

    float acc = 0.0f;

    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        const flambeau_block_iq1_m* bk = xrow + sb;

        const float d_super = flambeau_indexed_moe_iq1_m_reassemble_d(bk->scales);

        const int sc_word = (int) bk->scales[2 * (iqs >> 1)]
                          | ((int) bk->scales[2 * (iqs >> 1) + 1] << 8);
        const int shift1 = 6 * (iqs & 1);
        const int dl_scale = (l_grp < 2)
            ? ((sc_word >> shift1)       & 7)
            : ((sc_word >> (shift1 + 3)) & 7);
        const float dl = d_super * (2.0f * (float) dl_scale + 1.0f);

        const int qh_pick = (l_grp < 2) ? (2 * iqs) : (2 * iqs + 1);
        const uint8_t qh_byte = bk->qh[qh_pick];
        const int shift_idx = 8 - 4 * (l_grp & 1);
        const int idx = (int) bk->qs[4 * iqs + l_grp]
                      | (((int) qh_byte << shift_idx) & 0x700);
        const int delta_bit = (l_grp & 1) == 0 ? 0x08 : 0x80;
        const float delta = (qh_byte & delta_bit) ? -IQ1_DELTA : IQ1_DELTA;

        const uint64_t grid_u64 = IQ1S_GRID[idx];
        const int grid_lo = (int)(uint32_t) grid_u64;
        const int grid_hi = (int)(uint32_t)(grid_u64 >> 32);

        const flambeau_block_q8_1* ya = y_row + sb * 8 + iqs;
        const int u0 = ((const int*) ya->qs)[l0 + 0];
        const int u1 = ((const int*) ya->qs)[l0 + 1];
        const float d_y = (float) ya->d;

        int sumi = flambeau_indexed_moe_iq1_m_dp4a(grid_lo, u0, 0);
        sumi     = flambeau_indexed_moe_iq1_m_dp4a(grid_hi, u1, sumi);

        int sum_q8 = flambeau_indexed_moe_iq1_m_dp4a(0x01010101, u0, 0);
        sum_q8     = flambeau_indexed_moe_iq1_m_dp4a(0x01010101, u1, sum_q8);

        acc += dl * d_y * ((float) sumi + delta * (float) sum_q8);
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
