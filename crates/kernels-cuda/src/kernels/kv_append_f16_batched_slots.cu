// kv_append_f16_batched_slots — single-launch K+V append across N decode slots,
// each writing one new token row into its own KV cache (per-slot base pointers
// + write positions). Strided copy, no reduce. Grid (N,1,1), 128 threads.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void flambeau_kv_append_f16_batched_slots(
    const __half * __restrict__ k_src,
    const __half * __restrict__ v_src,
    void * const * __restrict__ slot_k_dst_ptrs,
    void * const * __restrict__ slot_v_dst_ptrs,
    const int  * __restrict__ slot_write_pos,
    int n_slots,
    int kv_width,
    int ring_depth
) {
    const int slot = blockIdx.x;
    if (slot >= n_slots) return;

    __half * dst_k = reinterpret_cast<__half *>(slot_k_dst_ptrs[slot]);
    __half * dst_v = reinterpret_cast<__half *>(slot_v_dst_ptrs[slot]);
    const int write_pos = slot_write_pos[slot];
    const int dst_pos = (ring_depth > 0) ? (write_pos % ring_depth) : write_pos;

    const __half * src_k = k_src + (size_t) slot * kv_width;
    const __half * src_v = v_src + (size_t) slot * kv_width;
    __half * out_k = dst_k + (size_t) dst_pos * kv_width;
    __half * out_v = dst_v + (size_t) dst_pos * kv_width;

    for (int i = threadIdx.x; i < kv_width; i += blockDim.x) {
        out_k[i] = src_k[i];
        out_v[i] = src_v[i];
    }
}
