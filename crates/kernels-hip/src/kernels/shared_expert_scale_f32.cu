// shared_expert_scale_f32 — compute per-token gate from dot(x_row, gate_w)
// + sigmoid, then scale `shared_out` in place.
// Fused form of candle's shared-expert gate path:
// gate[t] = sigmoid( Σ_i gate_w[i] · x[t, i] )
// shared_out[t, i] *= gate[t] // pointwise
// Shapes:
// x: [n_tokens, hidden] F32
// gate_w: [hidden] F32
// shared_out: [n_tokens, hidden] F32, overwritten
// Launch:
// gridDim = { n_tokens, 1, 1 }
// blockDim = { 256, 1, 1 } // 4 wave64
// shared = 4 floats for cross-warp reduce

#include <hip/hip_runtime.h>

#ifndef SHEXP_THREADS
#define SHEXP_THREADS 256
#endif
#define SHEXP_WARPS (SHEXP_THREADS / 64)

extern "C" __global__ void flambeau_shared_expert_scale_f32(
    float* __restrict__ shared_out,          // [n_tokens, hidden] in-place
    const float* __restrict__ x,             // [n_tokens, hidden]
    const float* __restrict__ gate_w,        // [hidden]
    const int n_tokens,
    const int hidden
) {
    const int row = blockIdx.x;
    if (row >= n_tokens) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const float* x_row = x + (size_t) row * hidden;
    float*       o_row = shared_out + (size_t) row * hidden;

    // Per-thread dot-product over strided chunks.
    float local_sum = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < hidden; i += SHEXP_THREADS) {
        local_sum += gate_w[i] * x_row[i];
    }

    // Warp reduce (wave64).
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        local_sum += __shfl_xor(local_sum, off, 64);
    }

    __shared__ float warp_sums[SHEXP_WARPS];
    if (lane == 0) {
        warp_sums[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float s = (lane < SHEXP_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = SHEXP_WARPS / 2; off > 0; off >>= 1) {
            s += __shfl_xor(s, off, 64);
        }
        if (lane == 0) {
            warp_sums[0] = s;
        }
    }
    __syncthreads();

    // sigmoid(sum) — numerically stable for `sum` ∈ F32.
    const float g = 1.0f / (1.0f + __expf(-warp_sums[0]));

    // Scale shared_out row by per-token gate.
    #pragma unroll 4
    for (int i = tid; i < hidden; i += SHEXP_THREADS) {
        o_row[i] = o_row[i] * g;
    }
}
