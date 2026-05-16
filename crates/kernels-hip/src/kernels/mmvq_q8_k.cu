// mmvq_q8_K — Q8_K weight × Q8_1 activation → F32 dst, dp4a inner loop.
// blockDim = { 256 }   (4 wave64 warps)
// gridDim  = { n_rows } (one output row per block)
// Q8_K super-block = 256 i8 quants + one f32 scale `d`. One super-block
// covers 8 Q8_1 sub-blocks (QK8_1=32). For each Q8_1 sub-block s the
// partial dot is `d_x * d_y[s] * sumi_s`; d_x factors out of the inner sum.
// Each thread owns one int32 (4 packed i8 quants) within a Q8_1 sub-block.
// 256 threads / 8 int32-per-sub-block = 32 sub-blocks processed per iter.

// Two output dtypes via templated __device__ body (#120):
//   flambeau_mmvq_q8_K_q8_1      → F32 dst
//   flambeau_mmvq_q8_K_q8_1_f16  → F16 dst (saturating)

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "mmvq_store.cuh"

#define MMVQ_Q8K_THREADS 256
#define MMVQ_Q8K_WARPS (MMVQ_Q8K_THREADS / WARP_SIZE)
#define MMVQ_Q8K_INT32_PER_SUB 8
#define MMVQ_Q8K_SUBS_PER_ITER (MMVQ_Q8K_THREADS / MMVQ_Q8K_INT32_PER_SUB)

static __device__ __forceinline__ int flambeau_dp4a_q8k(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template<typename OutT>
__device__ void mmvq_q8_k_body(
    const flambeau_block_q8_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    const int n_rows,
    const int n_super_blocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid     = threadIdx.x;
    const int warp    = tid / WARP_SIZE;
    const int lane    = tid & (WARP_SIZE - 1);
    const int lane8   = tid & 7;
    const int sub_idx = tid >> 3;

    const int n_sub_per_row = n_super_blocks_per_row * 8;
    const flambeau_block_q8_K* xrow = x + (size_t) row * n_super_blocks_per_row;

    float acc = 0.0f;
    for (int s = sub_idx; s < n_sub_per_row; s += MMVQ_Q8K_SUBS_PER_ITER) {
        const int super_idx    = s >> 3;
        const int sub_in_super = s & 7;
        const flambeau_block_q8_K* bx = xrow + super_idx;
        const flambeau_block_q8_1* by = y + s;

        const int* xqs_int = (const int*) (bx->qs + sub_in_super * 32);
        const int xi = xqs_int[lane8];
        const int yi = ((const int*) by->qs)[lane8];
        const int sumi = flambeau_dp4a_q8k(xi, yi, 0);

        const float d = bx->d * (float) by->d;
        acc += d * (float) sumi;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_Q8K_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_Q8K_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_Q8K_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_q8_K_q8_1(
    const flambeau_block_q8_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_super_blocks_per_row
) {
    mmvq_q8_k_body<float>(x, y, dst, n_rows, n_super_blocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q8_K_q8_1_f16(
    const flambeau_block_q8_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_super_blocks_per_row
) {
    mmvq_q8_k_body<fb_fp16_t>(x, y, dst, n_rows, n_super_blocks_per_row);
}
