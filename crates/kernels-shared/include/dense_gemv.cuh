#pragma once

// Per-row dense GEMV: y[row] = Σ_k w[row,k] · (float)x[k]. One block per row
// (256 threads), the dot reduced via block_reduce_sum over the WARP_SIZE seam.
// WT is the weight element type (float or fb_fp16_t); activation is always F16.
template <class WT>
static __device__ void dense_gemv_row(
    const WT* __restrict__ w,
    const fb_fp16_t* __restrict__ x,
    float* __restrict__ y,
    const int n_rows,
    const int k
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const WT* w_row = w + (size_t) row * k;

    float local = 0.0f;
    for (int i = tid; i < k; i += 256) {
        local += (float) w_row[i] * (float) x[i];
    }
    __shared__ float scratch[256 / WARP_SIZE];
    const float dot = block_reduce_sum<256>(local, scratch);
    if (tid == 0) y[row] = dot;
}

// Batched sibling: grid (n_rows, n_tokens); each token has its own activation
// row and output column.
template <class WT>
static __device__ void dense_gemv_row_batched(
    const WT* __restrict__ w,
    const fb_fp16_t* __restrict__ x,
    float* __restrict__ y,
    const int n_rows,
    const int k,
    const int n_tokens
) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    if (row >= n_rows || token >= n_tokens) return;
    const int tid = threadIdx.x;
    const WT* w_row = w + (size_t) row * k;
    const fb_fp16_t* x_row = x + (size_t) token * k;

    float local = 0.0f;
    for (int i = tid; i < k; i += 256) {
        local += (float) w_row[i] * (float) x_row[i];
    }
    __shared__ float scratch[256 / WARP_SIZE];
    const float dot = block_reduce_sum<256>(local, scratch);
    if (tid == 0) y[(size_t) token * n_rows + row] = dot;
}
