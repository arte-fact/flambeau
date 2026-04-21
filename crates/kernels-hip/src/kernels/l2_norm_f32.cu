// l2_norm_f32 — per-row L2 normalization, F32 in/out.
//
// For each row of `k` elements:
//   s  = Σ x[i]^2
//   y[i] = x[i] / sqrt(s + eps)
//
// Used by Gated-Delta-Net: Q and K projections get L2-normalized per head
// before the recurrent state update. We keep F32 precision (no F16 round-trip)
// because the GDN state recurrence amplifies Q/K noise over long sequences.
//
// Launch:
//   gridDim  = { n_rows, 1, 1 }
//   blockDim = { 256, 1, 1 }      // fixed — 4 wave64, matches RMSNorm layout
//   shared   = 4 floats for cross-warp reduce
//
// `k` must be > 0; `k >= 256` is optimal. For `k < 256` the kernel still
// works (extra threads contribute 0.0 to the reduce) but occupancy drops.

#include <hip/hip_runtime.h>

#ifndef L2_NORM_THREADS
#define L2_NORM_THREADS 256
#endif
#define L2_NORM_WARPS (L2_NORM_THREADS / 64)

extern "C" __global__ void flambeau_l2_norm_f32(
    const float* __restrict__ x,    // [n_rows, k]
    float* __restrict__ y,          // [n_rows, k]
    const int n_rows,
    const int k,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 6;            // 0..3 on wave64
    const int lane = tid & 63;

    const float* x_row = x + (size_t) row * k;
    float*       y_row = y + (size_t) row * k;

    // Per-thread sum of squares over this thread's stride.
    float local_sum = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < k; i += L2_NORM_THREADS) {
        const float v = x_row[i];
        local_sum += v * v;
    }

    // Warp reduce (wave64 → 1 lane holds the warp sum).
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        local_sum += __shfl_xor(local_sum, off, 64);
    }

    // Cross-warp reduce via LDS: each warp's lane-0 writes its sum, warp 0
    // then reduces the 4 partial sums.
    __shared__ float warp_sums[L2_NORM_WARPS];
    if (lane == 0) {
        warp_sums[warp] = local_sum;
    }
    __syncthreads();

    float total_sum;
    if (warp == 0) {
        float s = (lane < L2_NORM_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = L2_NORM_WARPS / 2; off > 0; off >>= 1) {
            s += __shfl_xor(s, off, 64);
        }
        if (lane == 0) {
            warp_sums[0] = s;
        }
    }
    __syncthreads();
    total_sum = warp_sums[0];

    const float scale = rsqrtf(total_sum + eps);

    #pragma unroll 4
    for (int i = tid; i < k; i += L2_NORM_THREADS) {
        y_row[i] = x_row[i] * scale;
    }
}
