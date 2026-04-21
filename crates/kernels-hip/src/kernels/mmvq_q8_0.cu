// mmvq_q8_0 — Q8_0 weight matrix × Q8_1 activation → F32 dst.
//
// First-class reference: candle-hip-kernels mul_mat_vec_q8_0_q8_1_cuda1
// (256-thread, one row per block — the P34 default at small M). This is the
// simplest of the V1 MMVQ kernels; Q4_K/Q5_K/Q6_K follow the same shape but
// with the multi-row DPP reduce from gfx906.cuh.
//
// Launch shape:
//   blockDim  = { 256 } threads (8 warps of 32 lanes on gfx906 / one wave64)
//   gridDim   = { N }            one block per output row
//   shared    = 0
//
// Contract:
//   x : [N, K] Q8_0 weights, row-major, K blocks-per-row = K / QK8_0
//   y : [K]    Q8_1 activation blocks, K/QK8_1
//   dst[n] = sum_k  x[n,k] . y[k]    in F32
//
// The inner loop streams blocks in pairs: 256 threads / 32 lanes-per-block
// gives 8 blocks processed cooperatively per iteration.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_BLOCK_THREADS 256
#define MMVQ_WARPS_PER_BLOCK (MMVQ_BLOCK_THREADS / WARP_SIZE)

extern "C" __global__ void flambeau_mmvq_q8_0_q8_1(
    const flambeau_block_q8_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid = threadIdx.x;             // 0..255
    const int warp = tid / WARP_SIZE;        // 0..3 on gfx906 wave64 (tid/64)
    const int lane = tid & (WARP_SIZE - 1);  // 0..63

    const flambeau_block_q8_0* xrow = x + (size_t) row * n_blocks_per_row;

    // Each warp takes every `MMVQ_WARPS_PER_BLOCK`-th block. Each lane
    // handles one of QK8_0 = 32 quant bytes inside that block — but we
    // have 64 lanes per wave, so lanes 32..63 read the *next* block. That
    // doubles throughput per iteration.
    const int lane_lo = lane & 31;           // 0..31 — which quant byte
    const int lane_hi = lane >> 5;           // 0 or 1 — which block of the pair

    float acc = 0.0f;
    for (int b = warp * 2 + lane_hi; b < n_blocks_per_row;
         b += MMVQ_WARPS_PER_BLOCK * 2) {
        const flambeau_block_q8_0* bx = xrow + b;
        const flambeau_block_q8_1* by = y + b;

        const int xi = (int) bx->qs[lane_lo];
        const int yi = (int) by->qs[lane_lo];

        const float d_x = (float) bx->d;
        const float d_y = (float) by->d;

        // Per-element product; the cross-lane sum over the block's 32 quant
        // lanes is deferred to the warp reduce below.
        acc += (float) (xi * yi) * d_x * d_y;
    }

    // Warp-wide sum (64 lanes). Every lane ends holding the full row sum
    // for its pair of blocks.
    acc = gfx906_warp_reduce_sum(acc);

    // Inter-warp reduce via shared memory — 4 warps → 4 values → lane-0 sum.
    __shared__ float s_warp[MMVQ_WARPS_PER_BLOCK];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_WARPS_PER_BLOCK) ? s_warp[lane] : 0.0f;
        // 4-way reduce is trivial via shfl_xor on the first warp.
        #pragma unroll
        for (int off = MMVQ_WARPS_PER_BLOCK / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            dst[row] = v;
        }
    }
}
