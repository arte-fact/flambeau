// kv_append_f16_paged_prefill — writes L prefill K + V rows of a
// single slot into the slot's paged KV cache, walking the slot's
// row of the block table per token. Companion to the decode-time
// `kv_append_f16_paged_slots` kernel.
//
// The host pre-acquires `ceil(L / page_size)` pages (plus any
// existing pages already owned for positions < `start_pos`) and
// patches the slot's row of `block_tables` before the kernel
// fires; this kernel never allocates pages — it only writes.
//
// Layout:
//   k_pool / v_pool: `[n_pages, page_size, kv_width]` f16 page
//     pools.
//   block_table: a pointer at the slot's row of the global block
//     table — `[max_pages_per_slot]` u32. The kernel walks it
//     starting at offset `start_pos / page_size`.
//   k_src / v_src: `[L, kv_width]` f16 — the row-major K and V
//     produced by the prefill projection + RoPE for token t goes
//     to position `start_pos + t` in the slot.
//
// Grid: (L,). Block: 128 threads strided across `kv_width`.
// `page_size` must be a power of two so the divide and modulo
// compile to shifts and AND masks.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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

    const int target_pos        = start_pos + t;
    const int page_idx_in_slot  = target_pos / page_size;
    const int page_offset       = target_pos - page_idx_in_slot * page_size;
    const unsigned int page     = block_table[page_idx_in_slot];

    const __half * src_k = k_src + (size_t) t * kv_width;
    const __half * src_v = v_src + (size_t) t * kv_width;
    __half * out_k = k_pool + ((size_t) page * page_size + page_offset) * kv_width;
    __half * out_v = v_pool + ((size_t) page * page_size + page_offset) * kv_width;

    for (int i = threadIdx.x; i < kv_width; i += blockDim.x) {
        out_k[i] = src_k[i];
        out_v[i] = src_v[i];
    }
}
