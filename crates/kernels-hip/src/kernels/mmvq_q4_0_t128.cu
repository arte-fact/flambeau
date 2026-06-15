// mmvq_q4_0_t128 — Q4_0 thin-block MMVQ, gfx906 latency-bound decode lever.
// Q4_0 weight counterpart of `mmvq_q4_1_t128`. Motivation: gfx906 at
// batch=1 sits at ~10% HBM bandwidth
// utilisation — the kernel is latency-bound, not bandwidth-bound, so the
// schedule that wins is the one that packs more in-flight blocks per CU,
// not the one that reads the fewest bytes. 128 threads/block = 2 wave64s/CU
// = up to 2 concurrent blocks per CU on gfx906 (vs the 256t baseline's
// 1 block/CU at the occupancy ceiling), giving Q4_0 the same latency-hiding
// shape Q4_1 already has via `mmvq_q4_1_t128` ().
// Q4_0 dequant via the (q − 8) DP4A bias-correction identity
// (`sumi · d_x · d_y − 8 · d_x · s_y` per block, split across 4 lanes by
// ·0.25 so the warp reduce sums to one correction per block).

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q4_0_T128_THREADS 128
#define MMVQ_Q4_0_T128_WARPS (MMVQ_Q4_0_T128_THREADS / WARP_SIZE)
#define MMVQ_Q4_0_T128_BLOCKS_PER_ITER (MMVQ_Q4_0_T128_THREADS / 4)

static __device__ __forceinline__ int flambeau_q4_0_t128_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_mmvq_q4_0_t128_q8_1(
    const flambeau_block_q4_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid & (WARP_SIZE - 1);
    const int lane4     = tid & 3;
    const int block_idx = tid >> 2;

    const flambeau_block_q4_0* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_Q4_0_T128_BLOCKS_PER_ITER) {
        const flambeau_block_q4_0* bx = xrow + b;
        const flambeau_block_q8_1* by = y + b;

        const int v = ((const int*) bx->qs)[lane4];
        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];

        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;

        int sumi = 0;
        sumi = flambeau_q4_0_t128_dp4a(vi_lo, u_lo, sumi);
        sumi = flambeau_q4_0_t128_dp4a(vi_hi, u_hi, sumi);

        const float d_x = (float) bx->d;
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        // Q4_0 bias correction (q − 8) split across 4 lanes per block.
        acc += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_Q4_0_T128_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_Q4_0_T128_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_Q4_0_T128_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            dst[row] = v;
        }
    }
}
