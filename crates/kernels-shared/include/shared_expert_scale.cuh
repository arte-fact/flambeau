#pragma once

// shared_expert_scale_f32 — per-token gate = sigmoid(dot(x_row, gate_w)), then
// scale shared_out in place. One block per row, THREADS threads; the dot-product
// block reduce rides block_reduce_sum (WARP_SIZE seam).
template <int THREADS>
static __device__ void shared_expert_scale_f32_row(
    float* __restrict__ shared_out,
    const float* __restrict__ x,
    const float* __restrict__ gate_w,
    const int n_tokens,
    const int hidden
) {
    const int row = blockIdx.x;
    if (row >= n_tokens) return;
    const int tid = threadIdx.x;
    const float* x_row = x + (size_t) row * hidden;
    float* o_row = shared_out + (size_t) row * hidden;

    float local = 0.0f;
    for (int i = tid; i < hidden; i += THREADS) {
        local += gate_w[i] * x_row[i];
    }
    __shared__ float scratch[THREADS / WARP_SIZE];
    const float dot = block_reduce_sum<THREADS>(local, scratch);

    const float g = 1.0f / (1.0f + __expf(-dot));
    for (int i = tid; i < hidden; i += THREADS) {
        o_row[i] = o_row[i] * g;
    }
}
