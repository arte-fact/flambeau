#pragma once

// kv_append_v_unit_norm_f16 — fused per-head V RMSNorm (unit weights) + K/V
// append into the cache. One block per (token, kv_head), blockDim = HEAD_DIM.
// K is a plain copy; V is normalized by its per-head inverse-RMS (sum-of-squares
// reduce via block_reduce_sum over the WARP_SIZE seam). Templated on HEAD_DIM so
// each instantiation carries its own __launch_bounds__.
template <int HEAD_DIM>
static __device__ void kv_append_v_unit_norm_f16_body(
    const fb_fp16_t* __restrict__ k_src,
    const fb_fp16_t* __restrict__ v_src,
    fb_fp16_t* __restrict__ k_cache,
    fb_fp16_t* __restrict__ v_cache,
    const int n_kv_heads,
    const int write_pos,
    const float eps,
    const int ring_depth
) {
    const int token = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int tid = threadIdx.x;

    const int dst_pos = (ring_depth > 0) ? ((write_pos + token) % ring_depth) : (write_pos + token);
    const size_t row_src = ((size_t) token * n_kv_heads + kv_head) * HEAD_DIM;
    const size_t row_dst = ((size_t) dst_pos * n_kv_heads + kv_head) * HEAD_DIM;

    const float v_val = (tid < HEAD_DIM) ? (float) v_src[row_src + tid] : 0.0f;
    if (tid < HEAD_DIM) {
        k_cache[row_dst + tid] = k_src[row_src + tid];
    }

    __shared__ float scratch[HEAD_DIM / WARP_SIZE];
    const float total_sq = block_reduce_sum<HEAD_DIM>(v_val * v_val, scratch);
    const float inv_rms = 1.0f / sqrtf(total_sq / (float) HEAD_DIM + eps);

    if (tid < HEAD_DIM) {
        v_cache[row_dst + tid] = (fb_fp16_t) (v_val * inv_rms);
    }
}
