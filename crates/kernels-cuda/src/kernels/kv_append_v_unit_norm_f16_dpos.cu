// Device-position fused V-unit-norm + KV append, F16. CUDA-only thin wrappers
// over the shared kv_append_v_unit_norm.cuh body (architectural rule 5): the
// write row is read from a device buffer (`pos_ptr[0]` = position) instead of a
// host scalar arg, so a captured decode graph advances the KV write cursor with
// no per-token rebind. `ring_depth` stays a baked constant. One block per
// (token, kv_head); the decode capture always launches n_tokens = 1.
#include "sm_80.cuh"
#include "block_quant.cuh"
#include "block_reduce.cuh"
#include "kv_append_v_unit_norm.cuh"

extern "C" __global__ __launch_bounds__(64, 16) void
flambeau_kv_append_v_unit_norm_f16_dpos_d64(
    const fb_fp16_t* __restrict__ k_src,
    const fb_fp16_t* __restrict__ v_src,
    fb_fp16_t* __restrict__ k_cache,
    fb_fp16_t* __restrict__ v_cache,
    const int n_kv_heads,
    const float eps,
    const int ring_depth,
    const int* __restrict__ pos_ptr
) {
    kv_append_v_unit_norm_f16_body<64>(
        k_src, v_src, k_cache, v_cache, n_kv_heads, pos_ptr[0], eps, ring_depth);
}

extern "C" __global__ __launch_bounds__(128, 8) void
flambeau_kv_append_v_unit_norm_f16_dpos_d128(
    const fb_fp16_t* __restrict__ k_src,
    const fb_fp16_t* __restrict__ v_src,
    fb_fp16_t* __restrict__ k_cache,
    fb_fp16_t* __restrict__ v_cache,
    const int n_kv_heads,
    const float eps,
    const int ring_depth,
    const int* __restrict__ pos_ptr
) {
    kv_append_v_unit_norm_f16_body<128>(
        k_src, v_src, k_cache, v_cache, n_kv_heads, pos_ptr[0], eps, ring_depth);
}

extern "C" __global__ __launch_bounds__(256, 4) void
flambeau_kv_append_v_unit_norm_f16_dpos_d256(
    const fb_fp16_t* __restrict__ k_src,
    const fb_fp16_t* __restrict__ v_src,
    fb_fp16_t* __restrict__ k_cache,
    fb_fp16_t* __restrict__ v_cache,
    const int n_kv_heads,
    const float eps,
    const int ring_depth,
    const int* __restrict__ pos_ptr
) {
    kv_append_v_unit_norm_f16_body<256>(
        k_src, v_src, k_cache, v_cache, n_kv_heads, pos_ptr[0], eps, ring_depth);
}

extern "C" __global__ __launch_bounds__(512, 2) void
flambeau_kv_append_v_unit_norm_f16_dpos_d512(
    const fb_fp16_t* __restrict__ k_src,
    const fb_fp16_t* __restrict__ v_src,
    fb_fp16_t* __restrict__ k_cache,
    fb_fp16_t* __restrict__ v_cache,
    const int n_kv_heads,
    const float eps,
    const int ring_depth,
    const int* __restrict__ pos_ptr
) {
    kv_append_v_unit_norm_f16_body<512>(
        k_src, v_src, k_cache, v_cache, n_kv_heads, pos_ptr[0], eps, ring_depth);
}
