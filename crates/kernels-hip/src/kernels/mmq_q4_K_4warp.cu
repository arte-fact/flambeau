// mmq_q4_K_4warp — 4-warp LDS-tiled MMQ for Q4_K weights × Q8_1 activation.
//
// First-class Q4_K prefill kernel for V1.4. Inner arithmetic is byte-identical
// to `indexed_moe_mmq_q4_k.cu`; only the expert indirection is removed so
// this can serve the dense (non-MoE) matmul path (attention Q/K/V/O
// projections + dense FFN gate/up/down).
//
// Tile shape:
//   MMQ_Y = 16 output rows per block
//   MMQ_X =  8 output columns per block
//   MMQ_K = 256 K-positions per iter (= one Q4_K super-block)
//
// Thread layout (128 threads = 2 wave64):
//   row_in_tile = tid / 8     (0..15)
//   col_in_tile = tid & 7     (0..7)
//
// LDS budget:
//   x_f32[MMQ_Y * 256] = 16 KB
//   y_f32[MMQ_X * 256] =  8 KB
//                        ≈ 24 KB / 64 KB LDS.
//
// The F32-tile variant lands correctness first. The int8-tile / dp4a refactor
// (per-subblock sc*d and m*dmin scales + int32 dot + sum(y) correction) is
// the next V1.4 perf step and supersedes this kernel behind the same
// `impl_id`.

#include "block_quant.cuh"
#include <hip/hip_runtime.h>

#define MMQ_Y 16
#define MMQ_X 8
#define MMQ_K 256
#define THREADS 128

extern "C" __global__ void flambeau_mmq_q4_K_4warp_q8_1(
    const flambeau_block_q4_K* __restrict__ x,   // [n_rows, n_sb_per_row]
    const flambeau_block_q8_1* __restrict__ y,   // [n_batches, n_sb_per_row * 8]
    float* __restrict__ dst,                     // [n_batches, n_rows]
    const int n_rows,
    const int n_batches,
    const int n_sb_per_row
) {
    const int row_base   = blockIdx.x * MMQ_Y;
    const int batch_base = blockIdx.y * MMQ_X;

    const int tid         = threadIdx.x;
    const int row_in_tile = tid / MMQ_X;            // 0..15
    const int col_in_tile = tid & (MMQ_X - 1);      // 0..7

    const int row   = row_base + row_in_tile;
    const int batch = batch_base + col_in_tile;
    const bool row_valid   = row   < n_rows;
    const bool batch_valid = batch < n_batches;

    __shared__ float x_tile[MMQ_Y * MMQ_K];
    __shared__ float y_tile[MMQ_X * MMQ_K];

    float acc = 0.0f;

    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        // -- Phase 1: dequant X super-block into LDS.
        //
        // MMQ_Y × 256 = 4096 F32 slots; 128 threads → 32 elements per thread.
        #pragma unroll 4
        for (int flat = tid; flat < MMQ_Y * MMQ_K; flat += THREADS) {
            const int r_in = flat / MMQ_K;                 // 0..15
            const int p    = flat - r_in * MMQ_K;           // 0..255
            float x_val = 0.0f;
            const int r_abs = row_base + r_in;
            if (r_abs < n_rows) {
                const flambeau_block_q4_K* bk =
                    x + (size_t) r_abs * n_sb_per_row + sb;
                const int sub        = p / 32;              // 0..7
                const int grp        = sub / 2;              // 0..3
                const int hi_half    = sub & 1;
                const int pos_in_sub = p - sub * 32;         // 0..31
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

        // -- Phase 2: dequant Y super-block into LDS.
        //
        // MMQ_X × 256 = 2048 F32 slots; 128 threads → 16 elements per thread.
        #pragma unroll 4
        for (int flat = tid; flat < MMQ_X * MMQ_K; flat += THREADS) {
            const int c_in = flat / MMQ_K;                 // 0..7
            const int p    = flat - c_in * MMQ_K;           // 0..255
            float y_val = 0.0f;
            const int b_abs = batch_base + c_in;
            if (b_abs < n_batches) {
                const flambeau_block_q8_1* y_sb =
                    y + (size_t) b_abs * n_sb_per_row * 8 + sb * 8;
                const int sub        = p / 32;              // 0..7
                const int pos_in_sub = p - sub * 32;         // 0..31
                const flambeau_block_q8_1* ya = y_sb + sub;
                y_val = (float) ya->d * (float) ya->qs[pos_in_sub];
            }
            y_tile[flat] = y_val;
        }

        __syncthreads();

        // -- Phase 3: per-thread dot over the super-block.
        if (row_valid && batch_valid) {
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

    if (row_valid && batch_valid) {
        dst[(size_t) batch * n_rows + row] = acc;
    }
}
