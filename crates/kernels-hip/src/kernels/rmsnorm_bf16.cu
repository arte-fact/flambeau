// rmsnorm_bf16 — RMSNorm over an [M, K] activation tensor, BF16 in/out.
// BF16 sibling of `rmsnorm_f16`. The norm weight stays F16
// because it's small (`[k]` only), loaded once per row, and F16 has 3
// more mantissa bits than BF16 — no benefit to widening it. Math is
// identical to the F16 variant, all in F32.
// Computation (per row):
// mean_sq = Σ x[i]² / K
// rsqrt = 1 / sqrt(mean_sq + eps)
// y[i] = bf16(x[i] * weight[i] * rsqrt)
// Launch shape: blockDim = {256}, gridDim = {n_rows}, shared = 4 floats.
// Each thread covers `k / 256` elements; `k` must divide 256 evenly
// (Qwen3.6 hidden=5120, head_dim=256, intermediate=25600 all qualify).

#include "block_quant.cuh"

#define RMSNORM_BF16_THREADS 256
#define RMSNORM_BF16_WARPS (RMSNORM_BF16_THREADS / 64)

extern "C" __global__ void flambeau_rmsnorm_bf16(
    const fb_bf16_t* __restrict__ x,        // [n_rows, k]
    const fb_fp16_t* __restrict__ weight,   // [k]
    fb_bf16_t* __restrict__ y,              // [n_rows, k]
    const int n_rows,
    const int k,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const fb_bf16_t* xrow = x + (size_t) row * (size_t) k;
    fb_bf16_t*       yrow = y + (size_t) row * (size_t) k;

    // --- Phase 1: sum-of-squares across the row ---
    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < k; i += RMSNORM_BF16_THREADS) {
        const float v = fb_bf16_to_f32(xrow[i]);
        sum_sq += v * v;
    }

    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }

    __shared__ float s_warp[RMSNORM_BF16_WARPS];
    if (lane == 0) {
        s_warp[warp] = sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < RMSNORM_BF16_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = RMSNORM_BF16_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 64);
        }
        if (lane == 0) {
            s_warp[0] = v;
        }
    }
    __syncthreads();
    const float total_sq = s_warp[0];

    const float mean_sq = total_sq / (float) k;
    const float rsqrt   = 1.0f / sqrtf(mean_sq + eps);

    // --- Phase 2: elementwise scale ---
    #pragma unroll 4
    for (int i = tid; i < k; i += RMSNORM_BF16_THREADS) {
        const float v = fb_bf16_to_f32(xrow[i]);
        const float w = (float) weight[i];
        yrow[i] = fb_f32_to_bf16(v * w * rsqrt);
    }
}
