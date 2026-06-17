#pragma once
// Templated IQ1_M single-pass tiled MMQ (prefill GEMM). Same arithmetic as IQ1_S
// (signed IQ1S_GRID + ±delta, bias-corrected via Σqi) but: no per-block `d` (it
// is reassembled from the top nibble of each of the 4 u16 `scales` words), and
// each inner `l` within a sub-block carries its own dl/delta. Σqi is recomputed
// per-l via dp4a(0x01010101,y) (no Q8_1 `s` field). `sums[c] += Σ_sub Σ_l
// dl_l·d_y·(sumi + delta_l·sumi_y)`. Backend-neutral: `dp4a` / `IQ1S_GRID` /
// `IQ1_DELTA` / `fb_fp16_t` from the shared headers, y loads through
// `load_int_b2`. Warp-size agnostic via MMQ_Y (no LDS). `OutT` is the dest dtype.
#include "quant_traits.cuh"
#include "iq_grid.cuh"
#include "mmvq_store.cuh"

static __device__ __forceinline__ float flambeau_iq1m_reassemble_d(const uint8_t* __restrict__ scales) {
    const int sc0 = (int) scales[0] | ((int) scales[1] << 8);
    const int sc1 = (int) scales[2] | ((int) scales[3] << 8);
    const int sc2 = (int) scales[4] | ((int) scales[5] << 8);
    const int sc3 = (int) scales[6] | ((int) scales[7] << 8);
    const int d_bits = (sc0 >> 12)
                     | ((sc1 >> 8) & 0x00F0)
                     | ((sc2 >> 4) & 0x0F00)
                     | (sc3 & 0xF000);
    fb_fp16_t d_fp16 = *reinterpret_cast<const fb_fp16_t*>(&d_bits);
    return (float) d_fp16;
}

template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_iq1_m_wave64_row(
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

    const flambeau_block_iq1_m* x = (const flambeau_block_iq1_m*) vx;
    const flambeau_block_q8_1*  y = (const flambeau_block_q8_1*)  vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;   // 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float d = 0.0f;
        const flambeau_block_iq1_m* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            d = flambeau_iq1m_reassemble_d(bx->scales);
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int v[8] = {0};
            float dl_l[4] = {0}, delta_l[4] = {0};
            if (row_ok) {
                const int sc_word = (int) bx->scales[2 * (sub >> 1)]
                                  | ((int) bx->scales[2 * (sub >> 1) + 1] << 8);
                const int shift1 = 6 * (sub & 1);
                const int shift2 = shift1 + 3;
                const float dl1 = d * (2.0f * (float) ((sc_word >> shift1) & 7) + 1.0f);
                const float dl2 = d * (2.0f * (float) ((sc_word >> shift2) & 7) + 1.0f);
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int qh_pick = (l < 2) ? (2 * sub) : (2 * sub + 1);
                    const uint8_t qh_byte = bx->qh[qh_pick];
                    const int shift_idx = 8 - 4 * (l & 1);
                    const int idx = (int) bx->qs[4 * sub + l]
                                  | (((int) qh_byte << shift_idx) & 0x700);
                    const int delta_bit = (l & 1) == 0 ? 0x08 : 0x80;
                    dl_l[l]    = (l < 2) ? dl1 : dl2;
                    delta_l[l] = (qh_byte & delta_bit) ? -IQ1_DELTA : IQ1_DELTA;
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
                float partial = 0.0f;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int y0 = load_int_b2(by->qs, 2 * l + 0);
                    const int y1 = load_int_b2(by->qs, 2 * l + 1);
                    int sumi = 0, sumi_y = 0;
                    sumi   = dp4a(v[2 * l + 0], y0, sumi);
                    sumi   = dp4a(v[2 * l + 1], y1, sumi);
                    sumi_y = dp4a(0x01010101, y0, sumi_y);
                    sumi_y = dp4a(0x01010101, y1, sumi_y);
                    partial += dl_l[l] * d_y * ((float) sumi + delta_l[l] * (float) sumi_y);
                }
                sums[c] += partial;
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
