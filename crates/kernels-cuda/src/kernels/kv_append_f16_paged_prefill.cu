// kv_append_f16_paged_prefill — write L prefill K+V rows of one slot into its
// paged KV cache, walking the slot's block-table row. Strided copy.
// grid (L,), block 128.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void flambeau_kv_append_f16_paged_prefill(
    const __half * __restrict__ k_src,
    const __half * __restrict__ v_src,
    __half       * __restrict__ k_pool,
    __half       * __restrict__ v_pool,
    const unsigned int * __restrict__ block_table,
    int n_tokens,
    int kv_width,
    int start_pos,
    int page_size
) {
    const int t = blockIdx.x;
    if (t >= n_tokens) return;

    const int target_pos = start_pos + t;
    const int page_idx_in_slot = target_pos / page_size;
    const int page_offset = target_pos - page_idx_in_slot * page_size;
    const unsigned int page = block_table[page_idx_in_slot];

    const __half * src_k = k_src + (size_t) t * kv_width;
    const __half * src_v = v_src + (size_t) t * kv_width;
    __half * out_k = k_pool + ((size_t) page * page_size + page_offset) * kv_width;
    __half * out_v = v_pool + ((size_t) page * page_size + page_offset) * kv_width;

    for (int i = threadIdx.x; i < kv_width; i += blockDim.x) {
        out_k[i] = src_k[i];
        out_v[i] = src_v[i];
    }
}
