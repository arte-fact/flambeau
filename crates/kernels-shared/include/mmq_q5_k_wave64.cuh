#pragma once
// Templated Q5_K single-pass tiled MMQ (prefill GEMM). Q4_K's super-block
// decode plus the 5th-bit plane: the high bit of each quant comes from `qh[32]`
// (`(qh_word >> sub) << 4 & 0x10101010`), OR-ed onto the 4-bit `qs` nibble.
// Per super-block accumulate `Σ_sub d8·sumi_d·sc` and `Σ_sub d8·sumi_y·m`, then
// combine `d·Σd − dmin·Σm`. Backend-neutral: `dp4a` / `flambeau_q4k_scale_min`
// (Q5_K packs (scale, min) identically) from the shared headers, int loads
// through `load_int_b2`. Warp-size agnostic via MMQ_Y (no LDS). `OutT` is the
// destination dtype.
#include "quant_traits.cuh"
#include "mmvq_store.cuh"

template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_q5_k_wave64_row(
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

    const flambeau_block_q5_K* x = (const flambeau_block_q5_K*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;   // 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f, super_dmin = 0.0f;
        uint8_t sub_sc[8] = {0}, sub_m[8] = {0};
        int qh_word[8] = {0};
        const flambeau_block_q5_K* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d    = (float) bx->d;
            super_dmin = (float) bx->dmin;
            #pragma unroll
            for (int j = 0; j < 8; ++j) flambeau_q4k_scale_min(j, bx->scales, &sub_sc[j], &sub_m[j]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) qh_word[j] = load_int_b2((const int8_t*) bx->qh, j);
        }

        float sumf_d[TILE_N], sumf_m[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) { sumf_d[c] = 0.0f; sumf_m[c] = 0.0f; }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int il   = sub >> 1;
            const int half = sub & 1;
            int v[8] = {0};
            if (row_ok) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = load_int_b2((const int8_t*) (bx->qs + 32 * il), j);
                    const int vl = (half == 0) ? (ql_word & 0x0F0F0F0F) : ((ql_word >> 4) & 0x0F0F0F0F);
                    const int vh = ((qh_word[j] >> sub) << 4) & 0x10101010;
                    v[j] = vl | vh;
                }
            }
            const float sc_f = (float) sub_sc[sub];
            const float m_f  = (float) sub_m[sub];

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int col = tile_n + c;
                if (col >= ncols_y) break;
                const flambeau_block_q8_1* by =
                    &y[(size_t) col * blocks_per_col_y + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                int sumi_d = 0, sumi_y = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = load_int_b2(by->qs, j);
                    sumi_d = dp4a(v[j], y_j, sumi_d);
                    sumi_y = dp4a(0x01010101, y_j, sumi_y);
                }
                sumf_d[c] += d8 * (float) sumi_d * sc_f;
                sumf_m[c] += d8 * (float) sumi_y * m_f;
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
