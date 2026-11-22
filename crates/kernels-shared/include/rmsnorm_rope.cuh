#pragma once

// rmsnorm_rope_neox_partial_f16 — fused per-head RMSNorm + partial NeoX RoPE,
// F16 in place. One block per (token, head), blockDim = HEAD_DIM. Phase 1: Σv²
// reduce via block_reduce_sum over the WARP_SIZE seam. Phase 2: normed value to
// LDS. Phase 3: partial NeoX rotation of the first rotated_dims (split pairs),
// pass-through for the rest. Templated on HEAD_DIM so each instantiation carries
// its own __launch_bounds__.
template <int HEAD_DIM>
static __device__ void rmsnorm_rope_neox_partial_f16_body(
    fb_fp16_t* __restrict__ x,
    const fb_fp16_t* __restrict__ norm_w,
    const int* __restrict__ positions,
    const int n_heads,
    const int rotated_dims,
    const float theta_base,
    const float eps
) {
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    const size_t row_base = ((size_t) token * n_heads + head) * HEAD_DIM;

    const float v = (tid < HEAD_DIM) ? (float) x[row_base + tid] : 0.0f;
    __shared__ float scratch[HEAD_DIM / WARP_SIZE];
    const float total_sq = block_reduce_sum<HEAD_DIM>(v * v, scratch);
    const float inv_rms = 1.0f / sqrtf(total_sq / (float) HEAD_DIM + eps);

    __shared__ float x_norm_lds[HEAD_DIM];
    const float normed = (tid < HEAD_DIM) ? v * (float) norm_w[tid] * inv_rms : 0.0f;
    if (tid < HEAD_DIM) {
        x_norm_lds[tid] = normed;
    }
    __syncthreads();

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
        x[row_base + hi] = (fb_fp16_t) (x0 * s + x1 * c);
    } else if (tid >= rotated_dims) {
        x[row_base + tid] = (fb_fp16_t) x_norm_lds[tid];
    }
}
