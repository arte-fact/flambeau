#pragma once
// Templated IQ3_XXS indexed-MoE MMVQ, r2 (2 rows/block, 32-lane groups). IQ3_XXS
// packs per 8-byte sub-block: 8 codebook-index bytes into the u32 `IQ3XXS_GRID`
// + an aux32 of four 7-bit sign-LUT indices and a 4-bit scale. Reuses the dense
// `iq2xxs_apply_signs` / `iq2xxs_unpack_ksigns`. Each 32-lane group owns one
// output row and covers a super-block as (iqs_idx ∈ 0..7) × (l_grp ∈ 0..3). MoE
// addressing: weight row by `expert_ids[token,slot]`, per-token activation
// `y[token,:]`, output `dst[token,slot,row]`. Backend-neutral: helpers from the
// dense header + `load_int_b2`; group reduce from the arch header.
#include "mmvq_iq3_xxs.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_iq3_xxs_r2_dp4a_row(
    const flambeau_block_iq3_xxs* __restrict__ x,
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
    const flambeau_block_iq3_xxs* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_sb_per_row * 8;

    const int iqs_idx = lane_lo >> 2;      // 0..7
    const int l_grp   = lane_lo & 3;       // 0..3
    const int iqs     = 2 * iqs_idx;
    const int l0      = 2 * l_grp;

    float acc = 0.0f;
    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        const flambeau_block_iq3_xxs* bk = xrow + sb;
        const float d_sb = (float) bk->d;

        const uint8_t* q3_base = bk->qs + 4 * iqs;
        const uint8_t q3_0 = q3_base[l0 + 0];
        const uint8_t q3_1 = q3_base[l0 + 1];

        const unsigned int aux32 =
            (unsigned int) load_int_b2((const int8_t*) (bk->qs + 64 + 2 * iqs), 0);
        const int ls = (int) (aux32 >> 28);

        const unsigned int signs_full = iq2xxs_unpack_ksigns(aux32 >> (7 * l_grp));
        const unsigned int sel_lo = signs_full & 0x08040201u;
        const unsigned int sel_hi = signs_full & 0x80402010u;
        unsigned int signs0_pack = 0, signs1_pack = 0;
        signs0_pack |= ((sel_lo >> 0) & 0xFFu) ? 0x000000FFu : 0;
        signs0_pack |= ((sel_lo >> 8) & 0xFFu) ? 0x0000FF00u : 0;
        signs0_pack |= ((sel_lo >> 16) & 0xFFu) ? 0x00FF0000u : 0;
        signs0_pack |= ((sel_lo >> 24) & 0xFFu) ? 0xFF000000u : 0;
        signs1_pack |= ((sel_hi >> 0) & 0xFFu) ? 0x000000FFu : 0;
        signs1_pack |= ((sel_hi >> 8) & 0xFFu) ? 0x0000FF00u : 0;
        signs1_pack |= ((sel_hi >> 16) & 0xFFu) ? 0x00FF0000u : 0;
        signs1_pack |= ((sel_hi >> 24) & 0xFFu) ? 0xFF000000u : 0;

        const unsigned int grid0 = IQ3XXS_GRID[q3_0];
        const unsigned int grid1 = IQ3XXS_GRID[q3_1];
        const int grid_signed0 = iq2xxs_apply_signs(grid0, signs0_pack);
        const int grid_signed1 = iq2xxs_apply_signs(grid1, signs1_pack);

        const flambeau_block_q8_1* ya = y_row + sb * 8 + (iqs >> 1);
        const int u0 = load_int_b2(ya->qs, l0 + 0);
        const int u1 = load_int_b2(ya->qs, l0 + 1);
        const float d_y = (float) ya->d;

        int sumi = dp4a(grid_signed0, u0, 0);
        sumi     = dp4a(grid_signed1, u1, sumi);

        sumi = (ls * sumi + sumi / 2) / 2;
        acc += d_sb * d_y * (float) sumi;
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_group<32, OutT>(acc, dst, dst_idx, lane_lo);
}
