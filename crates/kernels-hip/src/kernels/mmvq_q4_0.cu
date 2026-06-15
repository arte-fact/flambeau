// mmvq_q4_0 — Q4_0 weight × Q8_1 activation → {F32, F16} dst, DP4A inner loop.
// Q4_0 is the legacy 4-bit quant without a min: 18 bytes/block = 2-byte F16
// scale d + 16 bytes nibble-packed unsigned quants. Reconstruction is
// `y = d * (q - 8)` where `q ∈ [0, 15]`.
// DP4A can't subtract a scalar from packed bytes, but the identity
// (q - 8) · y = q · y - 8 · sum(y)
// lets us compute the dot product as
// sumi = dp4a(q_packed, y_packed, 0) (regular DP4A)
// dot = d_x * d_y * sumi - 8 · d_x · s_y (bias correction)
// where s_y = d_y · sum(y_qs), already baked into the Q8_1 block header.
// Same threading + nibble-pairing pattern as mmvq_q4_1:
// lane4 = tid & 3 — which int32 of the 16-byte qs
// block_idx = tid >> 2 — which Q4_0 block this thread processes
// u_lo = y.qs[lane4] — y int32 for elements 4·lane4 .. +3
// u_hi = y.qs[lane4 + 4] — y int32 for elements 4·lane4+16 .. +19
// vi_lo = (v >> 0) & 0x0F0F0F0F — 4 low nibbles
// vi_hi = (v >> 4) & 0x0F0F0F0F — 4 high nibbles
// Block/grid: blockDim=256, gridDim=n_rows (one block per output row).
//
// Two output dtypes via templated __device__ body:
//   flambeau_mmvq_q4_0_q8_1      → F32 dst (legacy scratch-then-cast path)
//   flambeau_mmvq_q4_0_q8_1_f16  → F16 dst (saturating; consumer-direct, no
//                                  scratch + no cast launch)

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "mmvq_store.cuh"

#define MMVQ_Q4_0_THREADS 256
#define MMVQ_Q4_0_WARPS (MMVQ_Q4_0_THREADS / WARP_SIZE)
#define MMVQ_Q4_0_INT32_PER_BLOCK 4
#define MMVQ_Q4_0_BLOCKS_PER_ITER (MMVQ_Q4_0_THREADS / MMVQ_Q4_0_INT32_PER_BLOCK)

static __device__ __forceinline__ int flambeau_q4_0_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template<typename OutT>
__device__ void mmvq_q4_0_q8_1_body(
    const flambeau_block_q4_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
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
    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_Q4_0_BLOCKS_PER_ITER) {
        const flambeau_block_q4_0* bx = xrow + b;
        const flambeau_block_q8_1* by = y + b;

        const int v = ((const int*) bx->qs)[lane4];
        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];

        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;

        int sumi = 0;
        sumi = flambeau_q4_0_dp4a(vi_lo, u_lo, sumi);
        sumi = flambeau_q4_0_dp4a(vi_hi, u_hi, sumi);

        const float d_x = (float) bx->d;
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        // Per-block: sumi · (d_x · d_y) - 8 · d_x · s_y. The -8·d_x·s_y
        // correction term is constant per block; split across the 4 lanes
        // (lane4 ∈ [0,4)) by dividing by 4 so the warp-reduce sums to
        // exactly one correction per Q4_0 block.
        acc += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_Q4_0_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_Q4_0_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_Q4_0_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_q4_0_q8_1(
    const flambeau_block_q4_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    mmvq_q4_0_q8_1_body<float>(x, y, dst, n_rows, n_blocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q4_0_q8_1_f16(
    const flambeau_block_q4_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    mmvq_q4_0_q8_1_body<fb_fp16_t>(x, y, dst, n_rows, n_blocks_per_row);
}
