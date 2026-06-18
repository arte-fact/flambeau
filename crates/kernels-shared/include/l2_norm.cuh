#pragma once

// Per-row L2 normalization, F32 in/out. One block per row, THREADS threads.
// y[row,i] = x[row,i] / sqrt(Σ x[row]^2 + eps). Used by GDN to normalize the
// per-head Q/K projections before the recurrent state update.
template <int THREADS>
static __device__ void l2_norm_f32_row(
    const float* __restrict__ x,
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

    float local_sum = 0.0f;
    for (int i = tid; i < k; i += THREADS) {
        const float v = x_row[i];
        local_sum += v * v;
    }

    __shared__ float scratch[THREADS / WARP_SIZE];
    const float total = block_reduce_sum<THREADS>(local_sum, scratch);

    const float scale = rsqrtf(total + eps);
    for (int i = tid; i < k; i += THREADS) {
        y_row[i] = x_row[i] * scale;
    }
}
