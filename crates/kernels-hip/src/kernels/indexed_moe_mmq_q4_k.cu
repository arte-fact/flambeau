// indexed_moe_mmq_q4_k — 4-warp LDS-tiled MoE MMQ for Q4_K × Q8_1.
// Pattern from `mmq_q8_0_4warp.cu`, adapted for:
// * Q4_K weights (128-element super-blocks, 4-bit packed nibbles + packed
// 6-bit scales/mins), dequantised into LDS as F32;
// * Per-block expert indirection — caller (CPU) sorts (token, slot) pairs
// into per-expert buckets so that all MMQ_X slots in a single block
// share ONE expert. This is the architectural choice that makes the
// weight-tile amortisation work: `n_rows × top_k` blocks would need
// a different weight tile per block without sorting.
// Tile shape:
// MMQ_Y = 16 output rows per block
// MMQ_X = 8 (token, slot) pairs per block (all sharing one expert)
// Thread layout (128 threads = 2 wave64):
// row_in_tile = tid / 8 (0..15)
// col_in_tile = tid & 7 (0..7)
// LDS budget:
// x_f32[MMQ_Y * 256] = 16 * 256 * 4 = 16 KB (dequantised weight tile)
// y_f32[MMQ_X * 256] = 8 * 256 * 4 = 8 KB (dequantised activation tile)
// ≈ 24 KB / 64 KB LDS.
// Sentinel: `bucket_slots[i] = -1` marks an unfilled tail slot. Those threads
// skip the Y load + the output write.

#include "block_quant.cuh"
#include <hip/hip_runtime.h>

#define MMQ_Y 16
#define MMQ_X 8
#define MMQ_K 256     // one Q4_K super-block
#define THREADS 128

extern "C" __global__ void flambeau_indexed_moe_mmq_q4_k_q8_1(
    const flambeau_block_q4_K* __restrict__ x,     // [n_experts, n_rows, n_sb_per_row]
    const flambeau_block_q8_1* __restrict__ y,     // [n_tokens, n_sb_per_row * 8]
    const int* __restrict__ bucket_expert,         // [n_buckets]
    const int* __restrict__ bucket_slots,          // [n_buckets, MMQ_X] — (token<<16 | slot) or -1
    float* __restrict__ dst,                       // [n_tokens, top_k, n_rows]
    const int n_rows,
    const int n_sb_per_row,
    const int top_k
) {
    const int row_tile = blockIdx.x;
    const int bucket   = blockIdx.y;

    const int tid         = threadIdx.x;
    const int row_in_tile = tid / MMQ_X;            // 0..15
    const int col_in_tile = tid & (MMQ_X - 1);      // 0..7

    const int row  = row_tile * MMQ_Y + row_in_tile;
    const bool row_valid = row < n_rows;

    const int expert = bucket_expert[bucket];
    const int slot_ref = bucket_slots[(size_t) bucket * MMQ_X + col_in_tile];
    const bool slot_valid = slot_ref >= 0;
    const int token = slot_valid ? (slot_ref >> 16) : 0;
    const int slot  = slot_valid ? (slot_ref & 0xFFFF) : 0;

    // Pointer to this block's expert weight rows. All MMQ_Y rows of this
    // block come from the same expert, so the base is constant across the
    // block's K-loop.
    const flambeau_block_q4_K* w_base =
        x + ((size_t) expert * n_rows + row_tile * MMQ_Y) * n_sb_per_row;

    __shared__ float x_tile[MMQ_Y * MMQ_K];
    __shared__ float y_tile[MMQ_X * MMQ_K];

    float acc = 0.0f;

    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        // -- Phase 1: dequant X super-block into LDS.
        // MMQ_Y × 256 elements = 4096 F32 slots. 128 threads → 32 elements
        // per thread, strided. Each thread's assigned rows vary per sb so
        // we walk rows and positions via a flat index.
        #pragma unroll 4
        for (int flat = tid; flat < MMQ_Y * MMQ_K; flat += THREADS) {
            const int r_in = flat / MMQ_K;                 // 0..15
            const int p    = flat - r_in * MMQ_K;           // 0..255 — position within super-block
            float x_val = 0.0f;
            if (row_tile * MMQ_Y + r_in < n_rows) {
                const flambeau_block_q4_K* bk =
                    w_base + (size_t) r_in * n_sb_per_row + sb;
                // Dequant layout: elements 0..31 = low nibbles of bytes 0..31,
                // 32..63 = high nibbles of bytes 0..31, 64..95 = low of 32..63,
                // 96..127 = high of 32..63, etc. Same pattern the MMVQ
                // kernel uses.
                const int sub       = p / 32;                 // 0..7
                const int grp       = sub / 2;                 // 0..3
                const int hi_half   = sub & 1;
                const int pos_in_sub = p - sub * 32;           // 0..31
                const int byte_v = (int) bk->qs[grp * 32 + pos_in_sub];
                const int raw_q = hi_half ? (byte_v >> 4) : (byte_v & 0x0F);
                uint8_t sc = 0, m = 0;
                flambeau_q4k_scale_min(sub, bk->scales, &sc, &m);
                const float d    = (float) bk->d;
                const float dmin = (float) bk->dmin;
                x_val = d * (float) sc * (float) raw_q - dmin * (float) m;
            }
            x_tile[flat] = x_val;
        }

        // -- Phase 2: dequant Y super-block (per slot) into LDS.
        // MMQ_X × 256 elements = 2048 F32 slots. 128 threads → 16 elements
        // per thread, strided.
        #pragma unroll 4
        for (int flat = tid; flat < MMQ_X * MMQ_K; flat += THREADS) {
            const int s_in = flat / MMQ_K;                 // 0..7
            const int p    = flat - s_in * MMQ_K;           // 0..255
            float y_val = 0.0f;
            const int slot_ref_s =
                bucket_slots[(size_t) bucket * MMQ_X + s_in];
            if (slot_ref_s >= 0) {
                const int tok = slot_ref_s >> 16;
                const flambeau_block_q8_1* y_sb =
                    y + (size_t) tok * n_sb_per_row * 8 + sb * 8;
                const int sub = p / 32;                    // 0..7
                const int pos_in_sub = p - sub * 32;        // 0..31
                const flambeau_block_q8_1* ya = y_sb + sub;
                y_val = (float) ya->d * (float) ya->qs[pos_in_sub];
            }
            y_tile[flat] = y_val;
        }

        __syncthreads();

        // -- Phase 3: per-thread dot over the super-block.
        if (row_valid && slot_valid) {
            float partial = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < MMQ_K; ++k) {
                const float xv = x_tile[(size_t) row_in_tile * MMQ_K + k];
                const float yv = y_tile[(size_t) col_in_tile * MMQ_K + k];
                partial += xv * yv;
            }
            acc += partial;
        }

        __syncthreads();
    }

    if (row_valid && slot_valid) {
        dst[((size_t) token * top_k + slot) * n_rows + row] = acc;
    }
}
