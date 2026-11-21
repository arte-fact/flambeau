// Device-position F16 KV append. Copies one K row and one V row of `kv_width`
// F16 into the cache slabs at physical row
// `ring_depth > 0 ? position % ring_depth : position`, the position being read
// on-device from `pos_ptr[0]` so a captured decode graph advances the write
// cursor with no per-token rebind. `k_cache` / `v_cache` are the UNSLID slot
// bases. Decode-only (one row); one thread per element.
#include "sm_80.cuh"
#include "backend_compat.cuh"

extern "C" __global__ __launch_bounds__(256) void flambeau_kv_append_f16_dpos(
    const fb_fp16_t* __restrict__ k_src,
    const fb_fp16_t* __restrict__ v_src,
    fb_fp16_t* __restrict__ k_cache,
    fb_fp16_t* __restrict__ v_cache,
    const int kv_width,
    const int ring_depth,
    const int* __restrict__ pos_ptr
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= kv_width) return;
    const int position = pos_ptr[0];
    const int row = (ring_depth > 0) ? (position % ring_depth) : position;
    const long long dst = (long long) row * kv_width + j;
    k_cache[dst] = k_src[j];
    v_cache[dst] = v_src[j];
}
