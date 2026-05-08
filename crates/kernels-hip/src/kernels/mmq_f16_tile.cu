// mmq_f16_tile — 9.a tile-M MMQ for F16 weight × Q8_1 activation.
// shipped a trivial multi-row variant (same per-block math as the
// MMVQ, grid.y = n_tokens) that gave +4 % on 27B-UD-Q8_K_XL prefill —
// pure launch-overhead saving, no compute amortisation. 7 head-to-head
// showed the F16 path at 40 % of llama.cpp (56 vs 142 tok/s L=512).
// This kernel is the real tile-M: MMQ_Y = 64 output rows per block, MMQ_X
// = 8 activation rows per block, 64 threads / wave64. Each thread owns
// one output row's dot against all 8 activation rows. The weight is
// read once per K-sub-block per thread (into registers — 32 F16 fit
// easily); the activation tile (8 Q8_1 blocks = 8 × 34 B = 272 B) is
// read from HBM once per ib and L1-broadcast across the 64 threads.
// Key difference from 4.b's null Y-LDS port on Q4_K tile8: Y is read
// by all 64 threads but the broadcast pattern hits L1 hot. We don't LDS-
// stage either side — we just amortise weight HBM by having each of 64
// threads do 8× work instead of 1×.
// Launch: block = (64, 1, 1), grid = (⌈n_rows / 64⌉, ⌈n_tokens / 8⌉, 1).

#include "block_quant.cuh"
#include "gfx906.cuh"
#include <hip/hip_fp16.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef QK8_1
#define QK8_1 32
#endif

#define MMQ_Y 64
#define TILE_N 8

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_f16_tile_q8_1(
    const fb_fp16_t* __restrict__ x_f16,              // [n_rows, n_cols] row-major
    const flambeau_block_q8_1* __restrict__ y,        // [n_tokens, n_blocks_per_row]
    float* __restrict__ dst,                          // [n_tokens, n_rows] row-major
    const int n_rows,
    const int n_tokens,
    const int n_blocks_per_row                        // = n_cols / 32
) {
    const int tile_m = blockIdx.x * MMQ_Y;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid    = threadIdx.x;

    const int row     = tile_m + tid;
    const bool row_ok = (row < n_rows);

    const size_t n_cols = (size_t) n_blocks_per_row * 32;
    const fb_fp16_t* xrow = row_ok ? x_f16 + (size_t) row * n_cols : nullptr;

    // Persistent sums across ib loop — one per activation tile column.
    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    // Slot-token + early-exit for partial last tile.
    int slot_token[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int t = tile_n + c;
        slot_token[c] = (t < n_tokens) ? t : -1;
    }

    for (int ib = 0; ib < n_blocks_per_row; ++ib) {
        // Load 32 F16 weights for this thread's row into registers once.
        // 32 × 2 bytes = 64 bytes = 16 int32 — loaded as 4 × int128 worth of
        // vector reads on gfx906. Register cost: 32 F16 = 16 VGPRs (packed).
        float w_vals[32];
        if (row_ok) {
            const size_t w_off = (size_t) ib * 32;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                w_vals[j] = (float) xrow[w_off + j];
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 32; ++j) w_vals[j] = 0.0f;
        }

        // For each of TILE_N activation rows: read the Q8_1 block, dequant,
        // FMA against our weight row. Y is L1-hot across the 64 threads
        // (all threads read the same Q8_1 block per inner iter).
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            if (slot_token[c] < 0) continue;
            const flambeau_block_q8_1* by =
                &y[(size_t) slot_token[c] * n_blocks_per_row + ib];
            const float d_y = (float) by->d;
            const int8_t* qs = (const int8_t*) by->qs;

            float acc = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                acc += w_vals[j] * ((float) qs[j]);
            }
            sums[c] += d_y * acc;
        }
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        if (slot_token[c] < 0) continue;
        dst[(size_t) slot_token[c] * n_rows + row] = sums[c];
    }
}
