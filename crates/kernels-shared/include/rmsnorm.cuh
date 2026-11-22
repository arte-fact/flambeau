#pragma once

// Per-row RMSNorm, F32 weight + F32 in/out. One block per row, THREADS threads.
// y[row,i] = x[row,i] * weight[i] / sqrt(mean(x[row]^2) + eps). Vectorised
// float4 path when k and all three pointers are 16-byte aligned.
template <int THREADS>
static __device__ void rmsnorm_f32_row(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    const int n_rows,
    const int k,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid = threadIdx.x;
    const float* x_row = x + (size_t) row * k;
    float* y_row = y + (size_t) row * k;

    const bool vec4 = (k % 4) == 0
        && ((reinterpret_cast<size_t>(x_row) & 0xF) == 0)
        && ((reinterpret_cast<size_t>(y_row) & 0xF) == 0)
        && ((reinterpret_cast<size_t>(weight) & 0xF) == 0);

    float sum_sq = 0.0f;
    if (vec4) {
        const int k4 = k >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(x_row);
        for (int i = tid; i < k4; i += THREADS) {
            const float4 v = x4[i];
            sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int i = tid; i < k; i += THREADS) {
            const float v = x_row[i];
            sum_sq += v * v;
        }
    }

    __shared__ float scratch[THREADS / WARP_SIZE];
    const float total_sq = block_reduce_sum<THREADS>(sum_sq, scratch);

    const float mean_sq = total_sq / (float) k;
    const float rsqrt = 1.0f / sqrtf(mean_sq + eps);

    if (vec4) {
        const int k4 = k >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(x_row);
        const float4* w4 = reinterpret_cast<const float4*>(weight);
        float4* y4 = reinterpret_cast<float4*>(y_row);
        for (int i = tid; i < k4; i += THREADS) {
            const float4 v = x4[i];
            const float4 w = w4[i];
            float4 r;
            r.x = v.x * w.x * rsqrt;
            r.y = v.y * w.y * rsqrt;
            r.z = v.z * w.z * rsqrt;
            r.w = v.w * w.w * rsqrt;
            y4[i] = r;
        }
    } else {
        for (int i = tid; i < k; i += THREADS) {
            y_row[i] = x_row[i] * weight[i] * rsqrt;
        }
    }
}
