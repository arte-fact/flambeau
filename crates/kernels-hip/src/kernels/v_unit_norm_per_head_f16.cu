// v_unit_norm_per_head_f16 — per-head RMSNorm in place over the V
// activation tensor before it lands in the KV cache. Unit weights
// (no learnable gamma) — matches gemma4's `attn_v_unit_norm = true`.
//
// Layout:
//   v: F16 [n_tokens, n_kv_heads, head_dim] — read and written in place.
//
// Per (token, kv_head):
//   inv_rms = 1 / sqrt(mean(v[token, kv_head, :]^2) + eps)
//   v[token, kv_head, :] *= inv_rms
//
// Launch shape:
//   blockDim = { head_dim } (one wave64 minimum; head_dim ∈ {64, 128, 256, 512})
//   gridDim  = { n_kv_heads, n_tokens }
//   shared   = HEAD_DIM_MAX_WARPS floats (cross-warp partial sums)
//
// Composes with `kv_append_f16` (for the prefill slot's contiguous K
// rows) and `kv_append_f16_batched_slots` (for the N decode rows) —
// the mixed-batch path replaces the fused
// `kv_append_v_unit_norm_f16` of the prefill_shape-only branch in
// `standard_attn`.

#include <hip/hip_runtime.h>

typedef _Float16 fb_fp16_t;

#define VUNORM_MAX_HEAD_DIM 512
#define VUNORM_MAX_WARPS    (VUNORM_MAX_HEAD_DIM / 64)

extern "C" __global__ void flambeau_v_unit_norm_per_head_f16(
    fb_fp16_t* __restrict__ v,          // [n_tokens, n_kv_heads, head_dim]
    const int n_kv_heads,
    const int head_dim,
    const float eps
) {
    const int kv_head = blockIdx.x;
    const int token   = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp    = tid >> 6;
    const int lane    = tid & 63;
    const int n_warps = blockDim.x >> 6;

    fb_fp16_t* row =
        v + ((size_t) token * n_kv_heads + kv_head) * head_dim;

    // Phase 1: sum-of-squares across head_dim elements.
    // One thread per element when blockDim == head_dim. Accumulate
    // in F32 — the cert-envelope reason from rmsnorm_f16 applies.
    float sum_sq = 0.0f;
    if (tid < head_dim) {
        const float x = (float) row[tid];
        sum_sq = x * x;
    }

    // Wave reduce.
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }

    // Cross-warp reduce via LDS — only fires when head_dim > 64.
    __shared__ float s_warp[VUNORM_MAX_WARPS];
    if (n_warps > 1) {
        if (lane == 0 && warp < VUNORM_MAX_WARPS) {
            s_warp[warp] = sum_sq;
        }
        __syncthreads();
        if (warp == 0) {
            float v = (lane < n_warps) ? s_warp[lane] : 0.0f;
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1) {
                v += __shfl_xor(v, off, 64);
            }
            if (lane == 0) {
                s_warp[0] = v;
            }
        }
        __syncthreads();
        sum_sq = s_warp[0];
    }

    const float mean_sq = sum_sq / (float) head_dim;
    const float inv_rms = 1.0f / sqrtf(mean_sq + eps);

    // Phase 2: scale in place.
    if (tid < head_dim) {
        const float x = (float) row[tid];
        row[tid] = (fb_fp16_t) (x * inv_rms);
    }
}
