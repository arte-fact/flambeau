#pragma once
// Templated IQ3_S single-pass tiled MMQ (prefill GEMM). Per Q8_1 sub-block: scale
// nibble from `scales[sub>>1]` → db = 1 + 2·nibble; 8 codebook indices in
// `qs[sub*8..]` each extended by one high bit from `qh[sub]` (IQ3S_GRID, 4 i8
// magnitudes/entry), signed via the explicit `signs[sub*4+l]` byte. `sums[c] +=
// super_d · Σ_sub d8 · sumi · db`. Backend-neutral: `dp4a` / `IQ3S_GRID` /
// `flambeau_iq2xxs_apply_signs_packed` from the shared headers, y loads through
// `load_int_b2`. Warp-size agnostic via MMQ_Y (no LDS). `OutT` is the dest dtype.
#include "quant_traits.cuh"
#include "iq_grid.cuh"
#include "mmq_iq2_xxs_wave64.cuh"
#include "mmvq_store.cuh"

template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_iq3_s_wave64_row(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    OutT* __restrict__ dst,
    int ncols_x, int nrows_x, int ncols_y, int nrows_y, int nrows_dst)
{
    const int tile_m = blockIdx.x * MMQ_Y;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid    = threadIdx.x;

    const int row     = tile_m + tid;
    const bool row_ok = row < nrows_x;

    const flambeau_block_iq3_s* x = (const flambeau_block_iq3_s*) vx;
    const flambeau_block_q8_1*  y = (const flambeau_block_q8_1*)  vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;   // 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f;
        const flambeau_block_iq3_s* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d = (float) bx->d;
        }

        float sumf[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) sumf[c] = 0.0f;

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int v[8] = {0};
            float db_no_super = 0.0f;
            if (row_ok) {
                const uint8_t sc_byte   = bx->scales[sub >> 1];
                const int     sc_nibble = (sub & 1) ? (sc_byte >> 4) : (sc_byte & 0x0F);
                db_no_super = 1.0f + 2.0f * (float) sc_nibble;

                const uint8_t qh_byte = bx->qh[sub];
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int g1_idx = (int) bx->qs[sub * 8 + 2 * l + 0]
                                     | (((int) qh_byte << (8 - 2 * l)) & 0x100);
                    const int g2_idx = (int) bx->qs[sub * 8 + 2 * l + 1]
                                     | (((int) qh_byte << (7 - 2 * l)) & 0x100);
                    const int g1_u32 = (int) IQ3S_GRID[g1_idx];
                    const int g2_u32 = (int) IQ3S_GRID[g2_idx];
                    const int signs  = (int) bx->signs[sub * 4 + l];
                    v[2 * l + 0] = flambeau_iq2xxs_apply_signs_packed(g1_u32,  signs       & 0x0F);
                    v[2 * l + 1] = flambeau_iq2xxs_apply_signs_packed(g2_u32, (signs >> 4) & 0x0F);
                }
            }

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int col = tile_n + c;
                if (col >= ncols_y) break;
                const flambeau_block_q8_1* by =
                    &y[(size_t) col * blocks_per_col_y + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) sumi = dp4a(v[j], load_int_b2(by->qs, j), sumi);
                sumf[c] += d8 * (float) sumi * db_no_super;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) sums[c] += super_d * sumf[c];
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int col = tile_n + c;
        if (col < ncols_y && row < nrows_dst) mmvq_store<OutT>(dst, col * nrows_dst + row, sums[c]);
    }
}
