// rmsnorm_rope_neox_partial_f16 — fused per-head rmsnorm + partial NeoX
// RoPE, F16 in-place. Replaces the (rmsnorm_f16 → DtoD memcpy back →
// rope_neox_partial_f16) triplet that gemma4 Q-norm and K-norm paths
// currently fire per layer per token.
//
// Layout:
//   x:         [n_tokens, n_heads, head_dim] F16, in-place
//   norm_w:    [head_dim] F16
//   positions: [n_tokens] I32
//
// Math per (token, head):
//   1. v[i] = (float) x[token, head, i] for i ∈ [0, head_dim)
//   2. mean_sq = Σ v[i]² / head_dim
//   3. inv_rms = 1 / sqrt(mean_sq + eps)
//   4. normed[i] = v[i] * (float) norm_w[i] * inv_rms
//   5. For pair_i ∈ [0, rotated_dims/2):
//        angle = positions[token] * theta_base^(-2*pair_i/rotated_dims)
//        (x0, x1) = (normed[pair_i], normed[pair_i + rotated_dims/2])
//        x[token, head, pair_i]                      = x0*cos - x1*sin
//        x[token, head, pair_i + rotated_dims/2]     = x0*sin + x1*cos
//      Dimensions [rotated_dims, head_dim) pass through (no rotation).
//
// Launch shape (per head_dim wrapper):
//   gridDim  = { n_tokens, n_heads, 1 }
//   blockDim = { head_dim } (must be 64, 128, 256, or 512)

#include <hip/hip_runtime.h>

typedef _Float16 fb_fp16_t;

#define RNRO_MAX_HEAD_DIM 512
#define RNRO_MAX_WARPS    (RNRO_MAX_HEAD_DIM / 64)

template<int HEAD_DIM>
__device__ __forceinline__ void rmsnorm_rope_neox_partial_f16_body(
    fb_fp16_t*       __restrict__ x,
    const fb_fp16_t* __restrict__ norm_w,
    const int*       __restrict__ positions,
    const int n_heads,
    const int rotated_dims,
    const float theta_base,
    const float eps
) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int tid   = threadIdx.x;
    const int warp  = tid >> 6;
    const int lane  = tid & 63;
    const int n_warps = HEAD_DIM / 64;

    const size_t row_base = ((size_t) token * n_heads + head) * HEAD_DIM;

    // Phase 1: read x and accumulate per-thread sum-of-squares.
    const float v = (tid < HEAD_DIM) ? (float) x[row_base + tid] : 0.0f;
    float sum_sq = v * v;

    // Warp reduce (full wave64).
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }

    // Cross-warp reduce via LDS.
    __shared__ float s_warp[RNRO_MAX_WARPS];
    if (lane == 0) {
        s_warp[warp] = sum_sq;
    }
    __syncthreads();

    float total_sq;
    if (warp == 0) {
        float vv = (lane < n_warps) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = RNRO_MAX_WARPS / 2; off > 0; off >>= 1) {
            vv += __shfl_xor(vv, off, 64);
        }
        if (lane == 0) {
            s_warp[0] = vv;
        }
    }
    __syncthreads();
    total_sq = s_warp[0];

    const float inv_rms = 1.0f / sqrtf(total_sq / (float) HEAD_DIM + eps);

    // Phase 2: compute normed value (stash to LDS so the RoPE pair-access
    // can read both halves of the rotation).
    __shared__ float x_norm_lds[HEAD_DIM];
    const float normed = (tid < HEAD_DIM)
        ? v * (float) norm_w[tid] * inv_rms
        : 0.0f;
    if (tid < HEAD_DIM) {
        x_norm_lds[tid] = normed;
    }
    __syncthreads();

    // Phase 3: apply RoPE per pair. Threads in [0, rotated_dims/2) own
    // the rotation pair (tid, tid + rotated_dims/2) and write BOTH halves.
    // Threads in [rotated_dims/2, rotated_dims) skip (written by the lo
    // pair). Threads in [rotated_dims, HEAD_DIM) pass the rmsnormed value
    // through unchanged.
    if (tid >= HEAD_DIM) return;
    if (tid < rotated_dims / 2) {
        const int hi = tid + rotated_dims / 2;
        const float exponent = 2.0f * (float) tid / (float) rotated_dims;
        const float inv_freq = 1.0f / powf(theta_base, exponent);
        const float angle = (float) positions[token] * inv_freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float x0 = x_norm_lds[tid];
        const float x1 = x_norm_lds[hi];
        x[row_base + tid] = (fb_fp16_t) (x0 * c - x1 * s);
        x[row_base + hi]  = (fb_fp16_t) (x0 * s + x1 * c);
    } else if (tid >= rotated_dims) {
        x[row_base + tid] = (fb_fp16_t) x_norm_lds[tid];
    }
}

extern "C" __global__ __launch_bounds__(64, 16)
void flambeau_rmsnorm_rope_neox_partial_f16_d64(
    fb_fp16_t* __restrict__ x,
    const fb_fp16_t* __restrict__ norm_w,
    const int* __restrict__ positions,
    const int n_heads,
    const int rotated_dims,
    const float theta_base,
    const float eps
) {
    rmsnorm_rope_neox_partial_f16_body</*HEAD_DIM=*/64>(
        x, norm_w, positions, n_heads, rotated_dims, theta_base, eps);
}

extern "C" __global__ __launch_bounds__(128, 8)
void flambeau_rmsnorm_rope_neox_partial_f16_d128(
    fb_fp16_t* __restrict__ x,
    const fb_fp16_t* __restrict__ norm_w,
    const int* __restrict__ positions,
    const int n_heads,
    const int rotated_dims,
    const float theta_base,
    const float eps
) {
    rmsnorm_rope_neox_partial_f16_body</*HEAD_DIM=*/128>(
        x, norm_w, positions, n_heads, rotated_dims, theta_base, eps);
}

extern "C" __global__ __launch_bounds__(256, 4)
void flambeau_rmsnorm_rope_neox_partial_f16_d256(
    fb_fp16_t* __restrict__ x,
    const fb_fp16_t* __restrict__ norm_w,
    const int* __restrict__ positions,
    const int n_heads,
    const int rotated_dims,
    const float theta_base,
    const float eps
) {
    rmsnorm_rope_neox_partial_f16_body</*HEAD_DIM=*/256>(
        x, norm_w, positions, n_heads, rotated_dims, theta_base, eps);
}

extern "C" __global__ __launch_bounds__(512, 2)
void flambeau_rmsnorm_rope_neox_partial_f16_d512(
    fb_fp16_t* __restrict__ x,
    const fb_fp16_t* __restrict__ norm_w,
    const int* __restrict__ positions,
    const int n_heads,
    const int rotated_dims,
    const float theta_base,
    const float eps
) {
    rmsnorm_rope_neox_partial_f16_body</*HEAD_DIM=*/512>(
        x, norm_w, positions, n_heads, rotated_dims, theta_base, eps);
}
