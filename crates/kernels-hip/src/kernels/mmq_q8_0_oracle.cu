// mmq_q8_0_oracle — correctness-oracle MMQ for Q8_0 weights × Q8_1 activation.
// Grid layout:
// blockDim = { 256 } (single wave64 × 4, same as MMVQ)
// gridDim = { n_rows, n_batches }
// shared = 0
// Each block computes ONE output element `dst[batch, row] = sum_k x[row, k] * y[batch, k]`.
// Inner loop is identical to `mmvq_q8_0_q8_1`; only difference is the `y`
// pointer is offset by `batch * n_blocks_per_row` to reach the activation
// row for this batch.
// This is the correctness oracle — slow (no shared-mem tiling, no
// re-use of X across batches) but proven against the per-shape reference.
// The first-class 4-warp LDS-tiled port from llamacpp-turbo replaces this
// impl_id once certed.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMQ_ORACLE_THREADS 256

extern "C" __global__ void flambeau_mmq_q8_0_oracle_q8_1(
    const flambeau_block_q8_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_batches,
    const int n_blocks_per_row
) {
    const int row   = blockIdx.x;
    const int batch = blockIdx.y;
    if (row >= n_rows || batch >= n_batches) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 6;          // 0..3 on gfx906 wave64
    const int lane = tid & (WARP_SIZE - 1);
    const int lane_lo = lane & 31;      // 0..31
    const int lane_hi = lane >> 5;      // 0 or 1

    const flambeau_block_q8_0* xrow = x + (size_t) row * n_blocks_per_row;
    const flambeau_block_q8_1* ybatch = y + (size_t) batch * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = warp * 2 + lane_hi; b < n_blocks_per_row;
         b += 4 /* warps/block */ * 2) {
        const flambeau_block_q8_0* bx = xrow + b;
        const flambeau_block_q8_1* by = ybatch + b;

        const int xi = (int) bx->qs[lane_lo];
        const int yi = (int) by->qs[lane_lo];

        const float d_x = (float) bx->d;
        const float d_y = (float) by->d;

        acc += (float) (xi * yi) * d_x * d_y;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[4];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < 4) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            dst[(size_t) batch * n_rows + row] = v;
        }
    }
}
