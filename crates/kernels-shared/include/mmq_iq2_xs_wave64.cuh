#pragma once
// Templated IQ2_XS single-pass tiled MMQ (prefill GEMM). Per Q8_1 sub-block four
// u16 entries (9-bit grid index into IQ2XS_GRID | 7-bit sign-LUT index into
// KSIGNS_IQ2XS); scale nibble from `scales[sub>>1]`, db = (0.5 + nibble)·0.25.
// `sums[c] += super_d · Σ_sub d8 · sumi · db`. Backend-neutral: `dp4a` /
// `IQ2XS_GRID` / `KSIGNS_IQ2XS` / `flambeau_iq2xxs_apply_signs_packed` from the
// shared headers, y loads through `load_int_b2`. Warp-size agnostic via MMQ_Y
// (no LDS). `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "iq_grid.cuh"
#include "mmq_iq2_xxs_wave64.cuh"
#include "mmvq_store.cuh"

template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_iq2_xs_wave64_row(
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

    const flambeau_block_iq2_xs* x = (const flambeau_block_iq2_xs*) vx;
    const flambeau_block_q8_1*   y = (const flambeau_block_q8_1*)   vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;   // 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f;
        const flambeau_block_iq2_xs* bx = nullptr;
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
            float scale_factor = 0.0f;
            if (row_ok) {
                const uint8_t sc_byte = bx->scales[sub >> 1];
                const int sc_nibble = (sub & 1) ? (sc_byte >> 4) : (sc_byte & 0x0F);
                scale_factor = (0.5f + (float) sc_nibble) * 0.25f;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* qp = bx->qs + 8 * sub + 2 * l;
                    const int q_u16 = (int) qp[0] | ((int) qp[1] << 8);
                    const uint64_t g_u64 = IQ2XS_GRID[q_u16 & 0x1FF];
                    const int g_lo = (int) (g_u64 & 0xFFFFFFFF);
                    const int g_hi = (int) ((g_u64 >> 32) & 0xFFFFFFFF);
                    const uint8_t sign_byte = KSIGNS_IQ2XS[(q_u16 >> 9) & 0x7F];
                    v[2 * l + 0] = flambeau_iq2xxs_apply_signs_packed(g_lo, sign_byte & 0x0F);
                    v[2 * l + 1] = flambeau_iq2xxs_apply_signs_packed(g_hi, (sign_byte >> 4) & 0x0F);
                }
            }

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int col = tile_n + c;
                if (col >= ncols_y) break;
                const flambeau_block_q8_1* by =
                    &y[(size_t) col * blocks_per_col_y + ib * q8_per_super + sub];
                const float d_y = (float) by->d;
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) sumi = dp4a(v[j], load_int_b2(by->qs, j), sumi);
                sumf[c] += d_y * (float) sumi * scale_factor;
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
