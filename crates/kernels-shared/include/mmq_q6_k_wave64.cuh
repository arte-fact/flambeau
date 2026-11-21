#pragma once
// Templated Q6_K single-pass tiled MMQ (prefill GEMM). Same independent-thread
// `MMQ_Y×TILE_N` tile as the simple-dtype `mmq_wave64` skeleton (each thread one
// weight row, no LDS / no reduce → warp-size agnostic via MMQ_Y), but a Q6_K
// super-block decode: 8 sub-blocks per super-block, each decoding 32 weights to
// 8 packed int32 (`q = ql_nibble | (qh_bits<<4)` in [0,63]) and applying the
// `(raw−32)·y = raw·y − 32·Σy` bias (no byte-wise subtract — the borrow chain
// corrupts) plus the per-16-element signed i8 scales. Backend-neutral: `dp4a`
// from the arch header, the misaligned `ql`/`qh`/`qs` int loads through
// `load_int_b2` (block_q6_K is 210 B → odd super-blocks are non-mul-4). `OutT`
// is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_store.cuh"

template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_q6_k_wave64_row(
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

    const flambeau_block_q6_K* x = (const flambeau_block_q6_K*) vx;
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
        const flambeau_block_q6_K* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d = (float) bx->d;
            #pragma unroll
            for (int j = 0; j < 16; ++j) sc_buf[j] = bx->scales[j];
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int h        = sub >> 2;
            const int q_idx    = sub & 3;
            const int qh_shift = 2 * q_idx;

            int v[8] = {0};
            if (row_ok) {
                const int ql_base = 64 * h + ((q_idx & 1) ? 32 : 0);
                const int qh_base = 32 * h;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = load_int_b2((const int8_t*) (bx->ql + ql_base), j);
                    const int qh_word = load_int_b2((const int8_t*) (bx->qh + qh_base), j);
                    const int nib4 = (q_idx < 2)
                        ? (ql_word & 0x0F0F0F0F)
                        : ((ql_word >> 4) & 0x0F0F0F0F);
                    const int hi4 = ((qh_word >> qh_shift) << 4) & 0x30303030;
                    v[j] = nib4 | hi4;
                }
            }

            const int sc_a = (int) sc_buf[8 * h + 2 * q_idx + 0];
            const int sc_b = (int) sc_buf[8 * h + 2 * q_idx + 1];

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
                    const int yj = load_int_b2(by->qs, j);
                    sumi_a   = dp4a(v[j], yj, sumi_a);
                    sumi_y_a = dp4a(0x01010101, yj, sumi_y_a);
                }
                #pragma unroll
                for (int j = 4; j < 8; ++j) {
                    const int yj = load_int_b2(by->qs, j);
                    sumi_b   = dp4a(v[j], yj, sumi_b);
                    sumi_y_b = dp4a(0x01010101, yj, sumi_y_b);
                }

                const int corrected_a = sumi_a - 32 * sumi_y_a;
                const int corrected_b = sumi_b - 32 * sumi_y_b;
                sums[c] += super_d * d8 * ((float) (sc_a * corrected_a + sc_b * corrected_b));
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
