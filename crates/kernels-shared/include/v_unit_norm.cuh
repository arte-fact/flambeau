#pragma once

// v_unit_norm_per_head_f16 — per-head RMSNorm (unit weights) in place over the V
// activation before it lands in the KV cache (gemma4 attn_v_unit_norm). One
// block per (kv_head, token), blockDim = head_dim (one element per thread); the
// sum-of-squares reduce rides block_reduce_sum_dyn over the WARP_SIZE seam.
#define VUNORM_MAX_HEAD_DIM 512

static __device__ void v_unit_norm_per_head_f16_body(
    fb_fp16_t* __restrict__ v,
    const int n_kv_heads,
    const int head_dim,
    const float eps
) {
    const int kv_head = blockIdx.x;
    const int token = blockIdx.y;
    const int tid = threadIdx.x;
    fb_fp16_t* row = v + ((size_t) token * n_kv_heads + kv_head) * head_dim;

    const float x = (tid < head_dim) ? (float) row[tid] : 0.0f;
    __shared__ float scratch[VUNORM_MAX_HEAD_DIM / WARP_SIZE];
    const float sum_sq = block_reduce_sum_dyn(x * x, scratch);

    const float inv_rms = 1.0f / sqrtf(sum_sq / (float) head_dim + eps);
    if (tid < head_dim) {
        row[tid] = (fb_fp16_t) (x * inv_rms);
    }
}
