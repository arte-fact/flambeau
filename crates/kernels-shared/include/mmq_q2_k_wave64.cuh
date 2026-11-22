#pragma once
// Templated Q2_K single-pass tiled MMQ (prefill GEMM). Q2_K super-block: 16
// packed 4-bit (scale, min) pairs in `scales[16]` (lo nibble = scale, hi = min),
// 2-bit quants in `qs[64]`. Each Q8_1 sub-block of 32 covers two Q2_K
// sub-blocks (is_a = 2·sub, is_b = 2·sub+1): accumulate
// `Σ_sub d8·(sc_a·dot_a + sc_b·dot_b)` and `Σ_sub d8·(m_a·Σy_a + m_b·Σy_b)`
// (Σy via dp4a(0x01010101,y)), then combine `d·Σd − dmin·Σm`. Backend-neutral:
// `dp4a` from the shared headers, int loads through `load_int_b2`. Warp-size
// agnostic via MMQ_Y (no LDS). `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_store.cuh"

template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_q2_k_wave64_row(
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

    const flambeau_block_q2_K* x = (const flambeau_block_q2_K*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;   // 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f, super_dmin = 0.0f;
        uint8_t sc_buf[16] = {0};
        const flambeau_block_q2_K* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d    = (float) bx->d;
            super_dmin = (float) bx->dmin;
            #pragma unroll
            for (int j = 0; j < 16; ++j) sc_buf[j] = bx->scales[j];
        }

        float sumf_d[TILE_N], sumf_m[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) { sumf_d[c] = 0.0f; sumf_m[c] = 0.0f; }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int chunk_idx  = sub >> 2;
            const int shift      = 2 * (sub & 3);

            int v[8] = {0};
            if (row_ok) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = load_int_b2((const int8_t*) (bx->qs + chunk_idx * 32), j);
                    v[j] = (ql_word >> shift) & 0x03030303;
                }
            }

            const int sc_a = (int) (sc_buf[2 * sub + 0] & 0xF);
            const int sc_b = (int) (sc_buf[2 * sub + 1] & 0xF);
            const int m_a  = (int) (sc_buf[2 * sub + 0] >> 4);
            const int m_b  = (int) (sc_buf[2 * sub + 1] >> 4);

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int col = tile_n + c;
                if (col >= ncols_y) break;
                const flambeau_block_q8_1* by =
                    &y[(size_t) col * blocks_per_col_y + ib * q8_per_super + sub];
                const float d8 = (float) by->d;

                int sumi_a = 0, sumi_b = 0, sumi_y_a = 0, sumi_y_b = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int y_j = load_int_b2(by->qs, j);
                    sumi_a   = dp4a(v[j], y_j, sumi_a);
                    sumi_y_a = dp4a(0x01010101, y_j, sumi_y_a);
                }
                #pragma unroll
                for (int j = 4; j < 8; ++j) {
                    const int y_j = load_int_b2(by->qs, j);
                    sumi_b   = dp4a(v[j], y_j, sumi_b);
                    sumi_y_b = dp4a(0x01010101, y_j, sumi_y_b);
                }

                sumf_d[c] += d8 * (float) (sc_a * sumi_a + sc_b * sumi_b);
                sumf_m[c] += d8 * (float) (m_a * sumi_y_a + m_b * sumi_y_b);
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) sums[c] += super_d * sumf_d[c] - super_dmin * sumf_m[c];
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int col = tile_n + c;
        if (col < ncols_y && row < nrows_dst) mmvq_store<OutT>(dst, col * nrows_dst + row, sums[c]);
    }
}
