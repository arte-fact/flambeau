// kv_append_f16_paged_slots — Sarathi / PagedAttention sibling of
// `kv_append_f16_batched_slots`. Writes one new K+V row per slot into a
// shared `[n_pages, page_size, kv_width]` F16 page pool, with the per-slot
// destination resolved through a block-table indirection.
//
// Layout:
//   k_pool / v_pool: `[n_pages, page_size, kv_width]` f16, kv_width =
//     n_kv_heads * head_dim. Pages are NOT pre-assigned per slot; a slot
//     can hold any subset of pool pages, in any order.
//   block_tables: `[max_slots, max_pages_per_slot]` u32 row-major.
//     `block_tables[slot * max_pages_per_slot + (write_pos / page_size)]`
//     gives the global page index that holds this slot's
//     `write_pos`-th token. The host allocator populates the table
//     entry BEFORE this kernel fires; the kernel never allocates pages.
//   slot_write_pos: `[n_slots]` i32. Tail token position to write this
//     decode step (BEFORE the bump).
//   k_src / v_src: `[n_slots, kv_width]` f16 slot-major (same as the
//     contiguous batched-slots kernel — only the destination changes).
//
// `page_size` must be a power of two so the kernel can replace
// `t / page_size` and `t % page_size` with shifts and masks. Recommended
// default = 16 (matches vLLM).
//
// Grid: (n_slots,). Block: 128 threads strided across `kv_width`. Identical
// occupancy / VGPR profile to the contiguous batched-slots kernel — just
// adds one i32 + one u32 load before the copy.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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
    const int page_offset      = write_pos - page_idx_in_slot * page_size;
    const unsigned int page    =
        block_tables[(size_t) slot * max_pages_per_slot + page_idx_in_slot];

    const __half * src_k = k_src + (size_t) slot * kv_width;
    const __half * src_v = v_src + (size_t) slot * kv_width;
    __half * out_k = k_pool + ((size_t) page * page_size + page_offset) * kv_width;
    __half * out_v = v_pool + ((size_t) page * page_size + page_offset) * kv_width;

    for (int i = threadIdx.x; i < kv_width; i += blockDim.x) {
        out_k[i] = src_k[i];
        out_v[i] = src_v[i];
    }
}
