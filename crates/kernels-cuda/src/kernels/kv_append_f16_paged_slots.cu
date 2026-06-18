// kv_append_f16_paged_slots — write one K+V row per slot into a paged page pool
// via block-table indirection. Strided copy. grid (n_slots,), block 128.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void flambeau_kv_append_f16_paged_slots(
    const __half * __restrict__ k_src,
    const __half * __restrict__ v_src,
    __half       * __restrict__ k_pool,
    __half       * __restrict__ v_pool,
    const unsigned int * __restrict__ block_tables,
    const int          * __restrict__ slot_write_pos,
    int n_slots,
    int kv_width,
    int page_size,
    int max_pages_per_slot
) {
    const int slot = blockIdx.x;
    if (slot >= n_slots) return;

    const int write_pos = slot_write_pos[slot];
    const int page_idx_in_slot = write_pos / page_size;
    const int page_offset = write_pos - page_idx_in_slot * page_size;
    const unsigned int page = block_tables[(size_t) slot * max_pages_per_slot + page_idx_in_slot];

    const __half * src_k = k_src + (size_t) slot * kv_width;
    const __half * src_v = v_src + (size_t) slot * kv_width;
    __half * out_k = k_pool + ((size_t) page * page_size + page_offset) * kv_width;
    __half * out_v = v_pool + ((size_t) page * page_size + page_offset) * kv_width;

    for (int i = threadIdx.x; i < kv_width; i += blockDim.x) {
        out_k[i] = src_k[i];
        out_v[i] = src_v[i];
    }
}
