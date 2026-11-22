#pragma once
// Templated IQ4_XS single-pass tiled MMQ (prefill GEMM). Q4_K-shaped super-block
// but symmetric (no dmin/min term): weight nibbles decode through the 16-entry
// signed-i8 IQ4_NL LUT (packed 4/i32 so dp4a sees i8×i8), per-sub-block signed
// 6-bit scale `ls = flambeau_iq4_xs_scale(sub, scales_h, scales_l)` (biased -32).
// `sums[c] += super_d · Σ_sub d8 · sumi_d · ls`. Backend-neutral: `dp4a` /
// `flambeau_iq4nl_lut` / `flambeau_pack_iq4_lut` / `flambeau_iq4_xs_scale` from
// the shared headers, int loads through `load_int_b2`. Warp-size agnostic via
// MMQ_Y (no LDS). `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "iq_grid.cuh"
#include "mmq_iq4_nl_wave64.cuh"
#include "mmvq_store.cuh"

template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_iq4_xs_wave64_row(
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

    const flambeau_block_iq4_xs* x = (const flambeau_block_iq4_xs*) vx;
    const flambeau_block_q8_1*   y = (const flambeau_block_q8_1*)   vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;   // 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float    super_d  = 0.0f;
        uint16_t scales_h = 0;
        const flambeau_block_iq4_xs* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d  = (float) bx->d;
            scales_h = bx->scales_h;
        }

        float sumf_d[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) sumf_d[c] = 0.0f;

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int v[8] = {0};
            int ls   = 0;
            if (row_ok) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int word = load_int_b2((const int8_t*) (bx->qs + 16 * sub), j);
                    v[j]     = flambeau_pack_iq4_lut(word        & 0x0F0F0F0F);
                    v[j + 4] = flambeau_pack_iq4_lut((word >> 4) & 0x0F0F0F0F);
                }
                ls = flambeau_iq4_xs_scale(sub, scales_h, bx->scales_l);
            }

            const float ls_f = (float) ls;

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int col = tile_n + c;
                if (col >= ncols_y) break;
                const flambeau_block_q8_1* by =
                    &y[(size_t) col * blocks_per_col_y + ib * q8_per_super + sub];
                const float d8 = (float) by->d;

                int sumi_d = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) sumi_d = dp4a(v[j], load_int_b2(by->qs, j), sumi_d);
                sumf_d[c] += d8 * (float) sumi_d * ls_f;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) sums[c] += super_d * sumf_d[c];
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int col = tile_n + c;
        if (col < ncols_y && row < nrows_dst) mmvq_store<OutT>(dst, col * nrows_dst + row, sums[c]);
    }
}
