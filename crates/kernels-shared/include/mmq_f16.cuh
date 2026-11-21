#pragma once
// Templated F16-weight × Q8_1-activation MMQ — multi-activation-row variant of
// mmvq_f16 (grid = (n_rows, n_tokens)): replaces m independent MMVQ launches at
// prefill with one launch, amortising launch overhead across all token rows.
// Per-block math is byte-identical to mmvq_f16 (shared `flambeau_f16_q8_1_acc`);
// blockIdx.y picks the activation row, offsetting y + dst. Backend-neutral, no
// LDS beyond the reduce. `OutT` is the destination dtype.
#include "mmvq_f16.cuh"

template<int THREADS, class OutT = float>
static __device__ __forceinline__ void mmq_f16_q8_1_tile(
    const fb_fp16_t* __restrict__ x_f16,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int n_blocks_per_row)
{
    const int row = blockIdx.x;
    const int tok = blockIdx.y;
    if (row >= n_rows || tok >= n_tokens) return;

    const size_t n_cols = (size_t) n_blocks_per_row * 32;
    const float acc = flambeau_f16_q8_1_acc<THREADS>(
        x_f16 + (size_t) row * n_cols,
        y + (size_t) tok * n_blocks_per_row,
        n_blocks_per_row);
    mmvq_reduce_store<THREADS, OutT>(acc, dst, (int) ((size_t) tok * n_rows + row));
}

// Real tile-M MMQ: MMQ_Y output rows/block (each thread one row), TILE_N token
// cols/block. The weight row stays register-resident (32 F16/ib) and is reused
// across all TILE_N activation cols (HBM amortisation); the activation tile is
// L1-hot across the block. Independent threads — no LDS, no cross-lane reduce —
// so warp-size agnostic via MMQ_Y. Plain byte/F16 loads; `OutT` is the dest dtype.
template<int MMQ_Y, int TILE_N, class OutT = float>
static __device__ __forceinline__ void mmq_f16_tile_row(
    const fb_fp16_t* __restrict__ x_f16,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int n_blocks_per_row)
{
    const int tile_m = blockIdx.x * MMQ_Y;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid    = threadIdx.x;

    const int row     = tile_m + tid;
    const bool row_ok = row < n_rows;

    const size_t n_cols = (size_t) n_blocks_per_row * 32;
    const fb_fp16_t* xrow = row_ok ? x_f16 + (size_t) row * n_cols : nullptr;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    int slot_token[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int t = tile_n + c;
        slot_token[c] = (t < n_tokens) ? t : -1;
    }

    for (int ib = 0; ib < n_blocks_per_row; ++ib) {
        float w_vals[32];
        if (row_ok) {
            const size_t w_off = (size_t) ib * 32;
            #pragma unroll
            for (int j = 0; j < 32; ++j) w_vals[j] = (float) xrow[w_off + j];
        } else {
            #pragma unroll
            for (int j = 0; j < 32; ++j) w_vals[j] = 0.0f;
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            if (slot_token[c] < 0) continue;
            const flambeau_block_q8_1* by = &y[(size_t) slot_token[c] * n_blocks_per_row + ib];
            const float d_y = (float) by->d;
            const int8_t* qs = (const int8_t*) by->qs;

            float acc = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; ++j) acc += w_vals[j] * (float) qs[j];
            sums[c] += d_y * acc;
        }
    }

    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        if (slot_token[c] < 0) continue;
        mmvq_store<OutT>(dst, (int) ((size_t) slot_token[c] * n_rows + row), sums[c]);
    }
}
