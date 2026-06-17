#pragma once
// Templated IQ1_S single-pass tiled MMQ (prefill GEMM). IQ1S_GRID stores signed
// i8 directly, plus a per-sub-block ±delta offset: y = dl·(g_i8 + delta), so
// partial = dl·(d_y·Σ(g·qi) + delta·s) where s = d_y·Σqi is the Q8_1 `s` field.
// Per sub-block: 4 × 11-bit grid indices (IQ1S_GRID, 2048 × u64), dl/delta from
// the qh u16. `sums[c] += Σ_sub dl·(d_y·sumi + delta·s_y)`. Backend-neutral:
// `dp4a` / `IQ1S_GRID` / `IQ1_DELTA` from the shared headers, y loads through
// `load_int_b2`. Warp-size agnostic via MMQ_Y (no LDS). `OutT` is the dest dtype.
#include "quant_traits.cuh"
#include "iq_grid.cuh"
#include "mmvq_store.cuh"

template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_iq1_s_wave64_row(
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

    const flambeau_block_iq1_s* x = (const flambeau_block_iq1_s*) vx;
    const flambeau_block_q8_1*  y = (const flambeau_block_q8_1*)  vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;   // 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f;
        const flambeau_block_iq1_s* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d = (float) bx->d;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int v[8] = {0};
            float dl = 0.0f, delta = 0.0f;
            if (row_ok) {
                const int qh_u16 = (int) bx->qh[2 * sub]
                                 | ((int) bx->qh[2 * sub + 1] << 8);
                dl = super_d * (2.0f * (float) ((qh_u16 >> 12) & 7) + 1.0f);
                delta = (qh_u16 & 0x8000) ? -IQ1_DELTA : IQ1_DELTA;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int idx = (int) bx->qs[4 * sub + l]
                                  | (((qh_u16 >> (3 * l)) & 7) << 8);
                    const uint64_t g_u64 = IQ1S_GRID[idx];
                    v[2 * l + 0] = (int) (g_u64 & 0xFFFFFFFF);
                    v[2 * l + 1] = (int) ((g_u64 >> 32) & 0xFFFFFFFF);
                }
            }

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int col = tile_n + c;
                if (col >= ncols_y) break;
                const flambeau_block_q8_1* by =
                    &y[(size_t) col * blocks_per_col_y + ib * q8_per_super + sub];
                const float d_y = (float) by->d;
                const float s_y = (float) by->s;
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) sumi = dp4a(v[j], load_int_b2(by->qs, j), sumi);
                sums[c] += dl * (d_y * (float) sumi + delta * s_y);
            }
        }
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int col = tile_n + c;
        if (col < ncols_y && row < nrows_dst) mmvq_store<OutT>(dst, col * nrows_dst + row, sums[c]);
    }
}
