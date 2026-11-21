#pragma once
// Templated Q8_K single-pass tiled MMQ (prefill GEMM). Simplest K-quant: f32
// super-block scale `d`, plain i8 `qs[256]`, no nibble unpack, no per-sub-block
// scale or min. Each super-block contributes
// `sums[c] += d_x · Σ_sub d8 · dot32(qs_sub, y_sub)`. Backend-neutral: `dp4a`
// from the shared headers, int loads through `load_int_b2`. Warp-size agnostic
// via MMQ_Y (no LDS). `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_store.cuh"

template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_q8_k_wave64_row(
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

    const flambeau_block_q8_K* x = (const flambeau_block_q8_K*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;   // 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float x_d = 0.0f;
        const flambeau_block_q8_K* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            x_d = bx->d;
        }

        float sumf[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) sumf[c] = 0.0f;

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int x_packed[8] = {0};
            if (row_ok) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) x_packed[j] = load_int_b2((const int8_t*) (bx->qs + sub * 32), j);
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
                for (int j = 0; j < 8; ++j) sumi = dp4a(x_packed[j], load_int_b2(by->qs, j), sumi);
                sumf[c] += d_y * (float) sumi;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) sums[c] += x_d * sumf[c];
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int col = tile_n + c;
        if (col < ncols_y && row < nrows_dst) mmvq_store<OutT>(dst, col * nrows_dst + row, sums[c]);
    }
}
