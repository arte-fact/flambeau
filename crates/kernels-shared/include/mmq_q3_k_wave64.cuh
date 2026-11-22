#pragma once
// Templated Q3_K single-pass tiled MMQ (prefill GEMM). Q3_K super-block: 1-bit
// high plane in `hmask[32]`, 2 low bits in `qs[64]`, sixteen packed 6-bit
// signed scales in `scales[12]` (unpacked via flambeau_q3k_unpack_scales, -32
// bias applied at use). Per Q8_1 sub-block the signed quant raw4 = raw2 |
// (hmask_bit<<2) ∈ [0,7]; the (raw4-4)·y bias trick gives corrected = sumi -
// 4·Σy. `sums += d·d8·(sc_a·corr_a + sc_b·corr_b)`. Backend-neutral: `dp4a` from
// the shared headers, all int loads through `load_int_b2` (Q3_K's 110 B stride
// is non-mul-4). Warp-size agnostic via MMQ_Y (no LDS). `OutT` is the dest dtype.
#include "quant_traits.cuh"
#include "mmvq_store.cuh"

template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_q3_k_wave64_row(
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

    const flambeau_block_q3_K* x = (const flambeau_block_q3_K*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;   // 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f;
        int8_t sc_buf[16] = {0};
        const flambeau_block_q3_K* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d = (float) bx->d;
            flambeau_q3k_unpack_scales(bx->scales, sc_buf);
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int blk128        = sub >> 2;
            const int shift         = 2 * (sub & 3);
            const int hmask_bit_pos = (sub & 3) + 4 * blk128;

            int v[8] = {0};
            if (row_ok) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = load_int_b2((const int8_t*) (bx->qs + blk128 * 32), j);
                    const int qh_word = load_int_b2((const int8_t*) bx->hmask, j);
                    const int raw2 = (ql_word >> shift) & 0x03030303;
                    const int hi   = ((qh_word >> hmask_bit_pos) << 2) & 0x04040404;
                    v[j] = raw2 | hi;
                }
            }

            const int sc_a = (int) sc_buf[2 * sub + 0] - 32;
            const int sc_b = (int) sc_buf[2 * sub + 1] - 32;

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

                const int corrected_a = sumi_a - 4 * sumi_y_a;
                const int corrected_b = sumi_b - 4 * sumi_y_b;
                sums[c] += super_d * d8 * (float) (sc_a * corrected_a + sc_b * corrected_b);
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
