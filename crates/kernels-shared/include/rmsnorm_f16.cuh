#pragma once

// Per-row RMSNorm, F16 weight + F16 in/out. One block per row, THREADS threads.
// Sum-of-squares accumulates in F32; the block-wide reduce rides the neutral
// block_reduce_sum (WARP_SIZE seam). y[row,i] = x[row,i]*weight[i]*rsqrt.
template <int THREADS>
static __device__ void rmsnorm_f16_row(
    const fb_fp16_t* __restrict__ x,
    const fb_fp16_t* __restrict__ weight,
    fb_fp16_t* __restrict__ y,
    const int n_rows,
    const int k,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid = threadIdx.x;
    const fb_fp16_t* x_row = x + (size_t) row * k;
    fb_fp16_t* y_row = y + (size_t) row * k;

    float sum_sq = 0.0f;
    for (int i = tid; i < k; i += THREADS) {
        const float v = (float) x_row[i];
        sum_sq += v * v;
    }

    __shared__ float scratch[THREADS / WARP_SIZE];
    const float total_sq = block_reduce_sum<THREADS>(sum_sq, scratch);

    const float mean_sq = total_sq / (float) k;
    const float rsqrt = 1.0f / sqrtf(mean_sq + eps);

    for (int i = tid; i < k; i += THREADS) {
        const float v = (float) x_row[i];
        const float w = (float) weight[i];
        y_row[i] = (fb_fp16_t) (v * w * rsqrt);
    }
}
