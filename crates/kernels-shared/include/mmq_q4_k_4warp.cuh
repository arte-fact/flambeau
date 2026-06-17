#pragma once
// Templated Q4_K 4-warp LDS-tiled MMQ (prefill GEMM), F32-tile variant. Each
// K-iter dequantises one Q4_K super-block of MMQ_Y weight rows and MMQ_X
// activation cols into LDS as F32, then every thread does a 256-wide F32 dot for
// its (row, col). Already warp-size agnostic: the thread→tile map (`tid/MMQ_X`,
// `tid & (MMQ_X-1)`) and the strided LDS fills (`flat += THREADS`) are all
// tid-based, no cross-lane ops. Backend-neutral: `flambeau_q4k_scale_min` from
// the shared headers; all weight/activation reads are byte loads. `OutT` is the
// destination dtype. THREADS must equal MMQ_Y*MMQ_X.
#include "quant_traits.cuh"
#include "mmvq_store.cuh"

template<int MMQ_Y, int MMQ_X, int THREADS, class OutT = float>
static __device__ __forceinline__ void mmq_q4_k_4warp_row(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    OutT* __restrict__ dst,
    int n_rows, int n_batches, int n_sb_per_row)
{
    constexpr int MMQ_K = QK_K;   // 256 K-positions per super-block

    const flambeau_block_q4_K* x = (const flambeau_block_q4_K*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int row_base   = blockIdx.x * MMQ_Y;
    const int batch_base = blockIdx.y * MMQ_X;

    const int tid         = threadIdx.x;
    const int row_in_tile = tid / MMQ_X;
    const int col_in_tile = tid & (MMQ_X - 1);

    const int row   = row_base + row_in_tile;
    const int batch = batch_base + col_in_tile;
    const bool row_valid   = row   < n_rows;
    const bool batch_valid = batch < n_batches;

    __shared__ float x_tile[MMQ_Y * MMQ_K];
    __shared__ float y_tile[MMQ_X * MMQ_K];

    float acc = 0.0f;
    for (int sb = 0; sb < n_sb_per_row; ++sb) {
        #pragma unroll 4
        for (int flat = tid; flat < MMQ_Y * MMQ_K; flat += THREADS) {
            const int r_in = flat / MMQ_K;
            const int p    = flat - r_in * MMQ_K;
            float x_val = 0.0f;
            const int r_abs = row_base + r_in;
            if (r_abs < n_rows) {
                const flambeau_block_q4_K* bk = x + (size_t) r_abs * n_sb_per_row + sb;
                const int sub        = p / 32;
                const int grp        = sub / 2;
                const int hi_half    = sub & 1;
                const int pos_in_sub = p - sub * 32;
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

        #pragma unroll 4
        for (int flat = tid; flat < MMQ_X * MMQ_K; flat += THREADS) {
            const int c_in = flat / MMQ_K;
            const int p    = flat - c_in * MMQ_K;
            float y_val = 0.0f;
            const int b_abs = batch_base + c_in;
            if (b_abs < n_batches) {
                const flambeau_block_q8_1* y_sb = y + (size_t) b_abs * n_sb_per_row * 8 + sb * 8;
                const int sub        = p / 32;
                const int pos_in_sub = p - sub * 32;
                const flambeau_block_q8_1* ya = y_sb + sub;
                y_val = (float) ya->d * (float) ya->qs[pos_in_sub];
            }
            y_tile[flat] = y_val;
        }

        __syncthreads();

        if (row_valid && batch_valid) {
            float partial = 0.0f;
            #pragma unroll 8
            for (int kk = 0; kk < MMQ_K; ++kk) {
                partial += x_tile[(size_t) row_in_tile * MMQ_K + kk]
                         * y_tile[(size_t) col_in_tile * MMQ_K + kk];
            }
            acc += partial;
        }

        __syncthreads();
    }

    if (row_valid && batch_valid)
        mmvq_store<OutT>(dst, (int) ((size_t) batch * n_rows + row), acc);
}
