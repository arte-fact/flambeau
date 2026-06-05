// kv_append_v_unit_norm_f16 — fused KV-cache append with V unit-RMSNorm.
// Replaces (rmsnorm_f16 V → DtoD memcpy v_tmp→v_f16 → 2× DtoD memcpy
// kv_append) with one kernel launch. For gemma4 archs that set
// `attn_v_unit_norm = true` (i.e. RMSNorm V with unit weights before
// it lands in the KV cache).
//
// Inputs (decode n_tokens=1 or prefill n_tokens>1):
//   k_src: F16 [n_tokens, n_kv_heads, head_dim] (already RoPE'd)
//   v_src: F16 [n_tokens, n_kv_heads, head_dim] (raw, pre-norm)
//   k_cache: F16 [max_seq, n_kv_heads, head_dim] (slot)
//   v_cache: F16 [max_seq, n_kv_heads, head_dim] (slot)
//   write_pos: starting row in the slot
//   eps: RMSNorm epsilon
//
// Per (token, kv_head):
//   K: direct copy k_src[token, kv_head, :] → k_cache[write_pos+token, kv_head, :]
//   V: normalize then copy v_src[token, kv_head, :] (with inv_rms scale)
//      → v_cache[write_pos+token, kv_head, :]
//
// Launch shape: gridDim = (n_tokens, n_kv_heads); blockDim = head_dim.

#include <hip/hip_runtime.h>

typedef _Float16 fb_fp16_t;

#define KVAVN_MAX_HEAD_DIM 512
#define KVAVN_MAX_WARPS    (KVAVN_MAX_HEAD_DIM / 64)

template<int HEAD_DIM>
__device__ __forceinline__ void kv_append_v_unit_norm_f16_body(
    const fb_fp16_t* __restrict__ k_src,
    const fb_fp16_t* __restrict__ v_src,
    fb_fp16_t*       __restrict__ k_cache,
    fb_fp16_t*       __restrict__ v_cache,
    const int n_kv_heads,
    const int write_pos,
    const float eps,
    const int ring_depth
) {
    const int token = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int n_warps = HEAD_DIM / 64;

    // Ring-buffered SWA slab: each token's row wraps independently at
    // `ring_depth`, so a prefill chunk straddling the wrap is correct
    // per-token. ring_depth = 0 → absolute addressing (bit-identical).
    const int dst_pos = (ring_depth > 0) ? ((write_pos + token) % ring_depth)
                                         : (write_pos + token);
    const size_t row_src = ((size_t) token * n_kv_heads + kv_head) * HEAD_DIM;
    const size_t row_dst = ((size_t) dst_pos * n_kv_heads + kv_head) * HEAD_DIM;

    // Load V (for the norm) + K (for direct copy).
    const float v_val_f = (tid < HEAD_DIM) ? (float) v_src[row_src + tid] : 0.0f;
    if (tid < HEAD_DIM) {
        // K path: pure copy. No dependency on the V reduction.
        k_cache[row_dst + tid] = k_src[row_src + tid];
    }
    float sum_sq = v_val_f * v_val_f;

    // Warp reduce (full wave64).
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }

    // Cross-warp reduce.
    __shared__ float s_warp[KVAVN_MAX_WARPS];
    if (lane == 0) {
        s_warp[warp] = sum_sq;
    }
    __syncthreads();

    float total_sq;
    if (warp == 0) {
        float vv = (lane < n_warps) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = KVAVN_MAX_WARPS / 2; off > 0; off >>= 1) {
            vv += __shfl_xor(vv, off, 64);
        }
        if (lane == 0) {
            s_warp[0] = vv;
        }
    }
    __syncthreads();
    total_sq = s_warp[0];

    const float inv_rms = 1.0f / sqrtf(total_sq / (float) HEAD_DIM + eps);

    // V path: normalize and write to cache (unit weights, so no per-element weight).
    if (tid < HEAD_DIM) {
        v_cache[row_dst + tid] = (fb_fp16_t) (v_val_f * inv_rms);
    }
}

extern "C" __global__ __launch_bounds__(64, 16)
void flambeau_kv_append_v_unit_norm_f16_d64(
    const fb_fp16_t* __restrict__ k_src,
    const fb_fp16_t* __restrict__ v_src,
    fb_fp16_t* __restrict__ k_cache,
    fb_fp16_t* __restrict__ v_cache,
    const int n_kv_heads,
    const int write_pos,
    const float eps,
    const int ring_depth
) {
    kv_append_v_unit_norm_f16_body</*HEAD_DIM=*/64>(
        k_src, v_src, k_cache, v_cache, n_kv_heads, write_pos, eps, ring_depth);
}

extern "C" __global__ __launch_bounds__(128, 8)
void flambeau_kv_append_v_unit_norm_f16_d128(
    const fb_fp16_t* __restrict__ k_src,
    const fb_fp16_t* __restrict__ v_src,
    fb_fp16_t* __restrict__ k_cache,
    fb_fp16_t* __restrict__ v_cache,
    const int n_kv_heads,
    const int write_pos,
    const float eps,
    const int ring_depth
) {
    kv_append_v_unit_norm_f16_body</*HEAD_DIM=*/128>(
        k_src, v_src, k_cache, v_cache, n_kv_heads, write_pos, eps, ring_depth);
}

extern "C" __global__ __launch_bounds__(256, 4)
void flambeau_kv_append_v_unit_norm_f16_d256(
    const fb_fp16_t* __restrict__ k_src,
    const fb_fp16_t* __restrict__ v_src,
    fb_fp16_t* __restrict__ k_cache,
    fb_fp16_t* __restrict__ v_cache,
    const int n_kv_heads,
    const int write_pos,
    const float eps,
    const int ring_depth
) {
    kv_append_v_unit_norm_f16_body</*HEAD_DIM=*/256>(
        k_src, v_src, k_cache, v_cache, n_kv_heads, write_pos, eps, ring_depth);
}

extern "C" __global__ __launch_bounds__(512, 2)
void flambeau_kv_append_v_unit_norm_f16_d512(
    const fb_fp16_t* __restrict__ k_src,
    const fb_fp16_t* __restrict__ v_src,
    fb_fp16_t* __restrict__ k_cache,
    fb_fp16_t* __restrict__ v_cache,
    const int n_kv_heads,
    const int write_pos,
    const float eps,
    const int ring_depth
) {
    kv_append_v_unit_norm_f16_body</*HEAD_DIM=*/512>(
        k_src, v_src, k_cache, v_cache, n_kv_heads, write_pos, eps, ring_depth);
}
