// mmq_q6_K_4warp — 4-warp LDS-tiled MMQ for Q6_K weights × Q8_1 activation.
//
// First-class Q6_K prefill kernel for V1.4. Same tile shape and LDS layout as
// `mmq_q4_K_4warp.cu`; only the weight dequant step changes to match Q6_K's
// split ql/qh/scales layout.
//
// Tile shape:
//   MMQ_Y = 16 output rows per block
//   MMQ_X =  8 output columns per block
//   MMQ_K = 256 K-positions per iter (= one Q6_K super-block)
//
// Thread layout (128 threads = 2 wave64):
//   row_in_tile = tid / 8     (0..15)
//   col_in_tile = tid & 7     (0..7)
//
// Q6_K super-block dequant (same math as `mmvq_q6_k.cu`), indexed by
// position p ∈ 0..255:
//   sub         = p / 32        (0..7)   — Q8_1 sub-block index
//   pos_in_sub  = p & 31        (0..31)
//   h           = sub >> 2      (0..1)   — which 128-element half
//   q_idx       = sub & 3       (0..3)
//   lsub        = pos_in_sub >> 4 (0..1)
//   scale_idx   = 8*h + 2*q_idx + lsub
//   ql_byte     = ql[64*h + ((q_idx & 1) ? pos_in_sub + 32 : pos_in_sub)]
//   nibble      = (q_idx < 2) ? (ql_byte & 0x0F) : (ql_byte >> 4)
//   qh_bits     = (qh[32*h + pos_in_sub] >> (2*q_idx)) & 0x3
//   raw_q       = (nibble | (qh_bits << 4)) - 32
//   x_val       = d * scales[scale_idx] * raw_q

#include "block_quant.cuh"
#include <hip/hip_runtime.h>

#define MMQ_Y 16
#define MMQ_X 8
#define MMQ_K 256
#define THREADS 128

extern "C" __global__ void flambeau_mmq_q6_K_4warp_q8_1(
    const flambeau_block_q6_K* __restrict__ x,   // [n_rows, n_sb_per_row]
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
        #pragma unroll 4
        for (int flat = tid; flat < MMQ_Y * MMQ_K; flat += THREADS) {
            const int r_in = flat / MMQ_K;                 // 0..15
            const int p    = flat - r_in * MMQ_K;           // 0..255
            float x_val = 0.0f;
            const int r_abs = row_base + r_in;
            if (r_abs < n_rows) {
                const flambeau_block_q6_K* bk =
                    x + (size_t) r_abs * n_sb_per_row + sb;
                const int sub        = p >> 5;              // p/32
                const int pos_in_sub = p & 31;
                const int h          = sub >> 2;
                const int q_idx      = sub & 3;
                const int lsub       = pos_in_sub >> 4;

                const int ql_off = 64 * h + ((q_idx & 1) ? pos_in_sub + 32 : pos_in_sub);
                const int ql_byte = (int) bk->ql[ql_off];
                const int nibble  = (q_idx < 2) ? (ql_byte & 0x0F) : (ql_byte >> 4);
                const int qh_bits = (bk->qh[32 * h + pos_in_sub] >> (2 * q_idx)) & 0x3;
                const int raw_q = (nibble | (qh_bits << 4)) - 32;

                const int scale_idx = 8 * h + 2 * q_idx + lsub;
                const int sc = (int) bk->scales[scale_idx];
                const float d = (float) bk->d;
                x_val = d * (float) sc * (float) raw_q;
            }
            x_tile[flat] = x_val;
        }

        // -- Phase 2: dequant Y super-block into LDS. Same as Q4_K MMQ.
        #pragma unroll 4
        for (int flat = tid; flat < MMQ_X * MMQ_K; flat += THREADS) {
            const int c_in = flat / MMQ_K;                 // 0..7
            const int p    = flat - c_in * MMQ_K;           // 0..255
            float y_val = 0.0f;
            const int b_abs = batch_base + c_in;
            if (b_abs < n_batches) {
                const flambeau_block_q8_1* y_sb =
                    y + (size_t) b_abs * n_sb_per_row * 8 + sb * 8;
                const int sub        = p >> 5;
                const int pos_in_sub = p & 31;
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
