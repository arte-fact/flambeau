#pragma once
// Templated IQ2_S single-pass tiled MMQ (prefill GEMM). Per Q8_1 sub-block four
// codebook indices in `qs[4*sub+l]` promoted to 10 bits via 2 high bits from
// `qh[sub]` (IQ2S_GRID, 1024 entries); sign bytes stored directly in
// `qs[32+4*sub+l]` (no LUT). Scale nibble from `scales[sub>>1]`, db = (0.5 +
// nibble)·0.25. `sums[c] += super_d · Σ_sub d8 · sumi · db`. Backend-neutral:
// `dp4a` / `IQ2S_GRID` from the shared headers, y loads through `load_int_b2`.
// Warp-size agnostic via MMQ_Y (no LDS). `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "iq_grid.cuh"
#include "mmvq_store.cuh"

static __device__ __forceinline__ int flambeau_iq2s_apply_signs_byte(int g_u32, uint8_t signs, int byte_off) {
    int out = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int mag = (g_u32 >> (8 * j)) & 0xFF;
        const int neg = (signs >> (byte_off * 4 + j)) & 1;
        const int v   = neg ? -mag : mag;
        out |= (v & 0xFF) << (8 * j);
    }
    return out;
}

template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_iq2_s_wave64_row(
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

    const flambeau_block_iq2_s* x = (const flambeau_block_iq2_s*) vx;
    const flambeau_block_q8_1*  y = (const flambeau_block_q8_1*)  vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;   // 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f;
        const flambeau_block_iq2_s* bx = nullptr;
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
                const uint8_t qh_byte = bx->qh[sub];
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int idx = (int) bx->qs[4 * sub + l]
                                  | (((int) qh_byte << (8 - 2 * l)) & 0x300);
                    const uint64_t g_u64 = IQ2S_GRID[idx];
                    const int g_lo = (int) (g_u64 & 0xFFFFFFFF);
                    const int g_hi = (int) ((g_u64 >> 32) & 0xFFFFFFFF);
                    const uint8_t signs = bx->qs[32 + 4 * sub + l];
                    v[2 * l + 0] = flambeau_iq2s_apply_signs_byte(g_lo, signs, 0);
                    v[2 * l + 1] = flambeau_iq2s_apply_signs_byte(g_hi, signs, 1);
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
