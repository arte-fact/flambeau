// rmsnorm_f16 — RMSNorm over an [M, K] activation tensor, F16 in/out.
// Computation (per row):
// mean_sq = Σ x[i]² / K (reduction across the row)
// rsqrt = 1 / sqrt(mean_sq + eps)
// y[i] = x[i] * weight[i] * rsqrt (elementwise)
// Launch shape:
// blockDim = { 256 } (4 wave64 warps on gfx906)
// gridDim = { n_rows } one block per row
// shared = 4 floats (cross-warp partial sums)
// Each thread processes `K / 256` elements (must divide evenly for Qwen3.6's
// 2048 / 5120 / 15360 hidden sizes — all multiples of 256). The inner loop
// accumulates in F32 so round-off on long rows doesn't creep above the cert
// envelope.

#include <hip/hip_runtime.h>

#define RMSNORM_THREADS 256
#define RMSNORM_WARPS (RMSNORM_THREADS / 64)

typedef _Float16 fb_fp16_t;

extern "C" __global__ void flambeau_rmsnorm_f16(
    const fb_fp16_t* __restrict__ x,        // [n_rows, k]
    const fb_fp16_t* __restrict__ weight,   // [k]
    fb_fp16_t* __restrict__ y,              // [n_rows, k]
    const int n_rows,
    const int k,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 6;                  // 0..3
    const int lane = tid & 63;                  // 0..63

    const fb_fp16_t* xrow = x + (size_t) row * k;
    fb_fp16_t*       yrow = y + (size_t) row * k;

    // --- Phase 1: sum-of-squares across the row ---
    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < k; i += RMSNORM_THREADS) {
        const float v = (float) xrow[i];
        sum_sq += v * v;
    }

    // Warp reduce (full wave64). Use __shfl_xor — portable, clean, and the
    // sum-of-squares is a once-per-block operation so we don't need DPP.
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }

    // Cross-warp reduce via LDS.
    __shared__ float s_warp[RMSNORM_WARPS];
    if (lane == 0) {
        s_warp[warp] = sum_sq;
    }
    __syncthreads();

    float total_sq;
    if (warp == 0) {
        float v = (lane < RMSNORM_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = RMSNORM_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 64);
        }
        if (lane == 0) {
            s_warp[0] = v;
        }
    }
    __syncthreads();
    total_sq = s_warp[0];

    const float mean_sq = total_sq / (float) k;
    const float rsqrt   = 1.0f / sqrtf(mean_sq + eps);

    // --- Phase 2: elementwise scale ---
    #pragma unroll 4
    for (int i = tid; i < k; i += RMSNORM_THREADS) {
        const float v = (float) xrow[i];
        const float w = (float) weight[i];
        yrow[i] = (fb_fp16_t) (v * w * rsqrt);
    }
}
