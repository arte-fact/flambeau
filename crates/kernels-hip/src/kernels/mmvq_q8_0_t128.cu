// mmvq_q8_0_t128 — Q8_0 thin-block MMVQ. C9-i1 sibling of `mmvq_q4_0_t128`.
// Q8_0 MMVQ is 81 % of Qwen3.6-27B-Q8_0 / Coder-30B-Q8_0 decode wall on
// gfx906 (per profile tour). The kernel sits at ~10 % HBM
// utilisation — same latency-bound regime that t128 unblocks for Q4_0
// on this silicon. 128 t/block (= 2 wave64s/CU) packs more in-flight
// blocks per CU than the 256t baseline (1 block/CU at occupancy
// ceiling), giving Q8_0 the same shape Q4_0 already has via
// `mmvq_q4_0_t128` (TP-perf-c5).
// Inner loop is byte-identical to `mmvq_q8_0_dp4a`: each thread owns
// one int32 (4 packed Q8_0 quants); 16 quant blocks per iteration
// instead of 32 (128 / 8 = 16, vs 256 / 8 = 32 in the 256t kernel).

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q8_0_T128_THREADS 128
#define MMVQ_Q8_0_T128_WARPS (MMVQ_Q8_0_T128_THREADS / WARP_SIZE)
#define MMVQ_Q8_0_T128_INT32_PER_BLOCK 8           // QK8_0 / 4 = 8 int32 per block
#define MMVQ_Q8_0_T128_BLOCKS_PER_ITER (MMVQ_Q8_0_T128_THREADS / MMVQ_Q8_0_T128_INT32_PER_BLOCK)

static __device__ __forceinline__ int flambeau_q8_0_t128_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(MMVQ_Q8_0_T128_THREADS)
void flambeau_mmvq_q8_0_t128_q8_1(
    const flambeau_block_q8_0* __restrict__ x,
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
    const int lane8     = tid & 7;
    const int block_idx = tid >> 3;

    const flambeau_block_q8_0* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_Q8_0_T128_BLOCKS_PER_ITER) {
        const flambeau_block_q8_0* bx = xrow + b;
        const flambeau_block_q8_1* by = y + b;

        const int xi = ((const int*) bx->qs)[lane8];
        const int yi = ((const int*) by->qs)[lane8];
        const int sumi = flambeau_q8_0_t128_dp4a(xi, yi, 0);

        const float d = (float) bx->d * (float) by->d;
        acc += d * (float) sumi;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_Q8_0_T128_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_Q8_0_T128_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_Q8_0_T128_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            dst[row] = v;
        }
    }
}
