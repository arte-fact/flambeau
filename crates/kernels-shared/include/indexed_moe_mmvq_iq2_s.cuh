#pragma once
// Templated IQ2_S indexed-MoE MMVQ, r2 (2 rows/block, 32-lane groups). IQ2_S:
// a 10-bit codebook index (qs byte + qh 2-bit extension into `IQ2S_GRID`) with a
// separate signs byte per element, and `scales[ib32>>1]` 4-bit sub-block scales
// (flambeau convention). Reuses the dense `iq2xxs_apply_signs`. Each 32-lane
// group owns one output row and covers a super-block as (iqs_idx ∈ 0..7) ×
// (l_grp ∈ 0..3). MoE addressing: weight row by `expert_ids[token,slot]`,
// per-token activation `y[token,:]`, output `dst[token,slot,row]`. Backend-
// neutral: helpers from the dense header + `load_int_b2`; group reduce from the
// arch header.
#include "mmvq_iq2_s.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_iq2_s_r2_dp4a_row(
    const flambeau_block_iq2_s* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int top_k, int n_sb_per_row)
{
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
    const flambeau_block_iq2_s* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_sb_per_row * 8;

    const int iqs_idx = lane_lo >> 2;      // 0..7
    const int l_grp   = lane_lo & 3;       // 0..3
    const int iqs     = 2 * iqs_idx;
    const int l0      = 2 * l_grp;

    float acc = 0.0f;
    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        const flambeau_block_iq2_s* bk = xrow + sb;
        const float d_sb = (float) bk->d;

        const uint8_t* qs_pair_base = bk->qs + 2 * iqs;
        const uint8_t qs_byte = qs_pair_base[l_grp];

        const int qh = (int) bk->qh[iqs >> 1];
        const int grid_idx = (int) qs_byte | ((qh << (8 - l0)) & 0x300);

        const uint8_t signs_byte = bk->qs[32 + 2 * iqs + l_grp];

        const int sub = iqs >> 1;
        const int sc_byte = (int) bk->scales[sub >> 1];
        const int ls = (sub & 1) ? (sc_byte >> 4) : (sc_byte & 0x0F);

        const uint64_t grid_u64 = IQ2S_GRID[grid_idx];

        unsigned int signs0_pack = 0, signs1_pack = 0;
        signs0_pack |= (signs_byte & 0x01) ? 0x000000FFu : 0;
        signs0_pack |= (signs_byte & 0x02) ? 0x0000FF00u : 0;
        signs0_pack |= (signs_byte & 0x04) ? 0x00FF0000u : 0;
        signs0_pack |= (signs_byte & 0x08) ? 0xFF000000u : 0;
        signs1_pack |= (signs_byte & 0x10) ? 0x000000FFu : 0;
        signs1_pack |= (signs_byte & 0x20) ? 0x0000FF00u : 0;
        signs1_pack |= (signs_byte & 0x40) ? 0x00FF0000u : 0;
        signs1_pack |= (signs_byte & 0x80) ? 0xFF000000u : 0;

        const int grid_signed0 = iq2xxs_apply_signs((unsigned int) grid_u64, signs0_pack);
        const int grid_signed1 = iq2xxs_apply_signs((unsigned int) (grid_u64 >> 32), signs1_pack);

        const flambeau_block_q8_1* ya = y_row + sb * 8 + (iqs >> 1);
        const int u0 = load_int_b2(ya->qs, l0 + 0);
        const int u1 = load_int_b2(ya->qs, l0 + 1);
        const float d_y = (float) ya->d;

        int sumi = dp4a(grid_signed0, u0, 0);
        sumi     = dp4a(grid_signed1, u1, sumi);

        const float scale_factor = ((float) ls + 0.5f) * 0.25f;
        acc += d_sb * d_y * (float) sumi * scale_factor;
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_group<32, OutT>(acc, dst, dst_idx, lane_lo);
}
