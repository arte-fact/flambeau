// kv_append_f16_batched_slots — single-launch K+V append across N
// decode slots, each writing one new token row into its own KV cache.
//
// Replaces the 2N `hipMemcpyAsync(DtoD)` calls in
// `forward_full_attn_layer_decode_batched_tp`. Each slot owns an
// independent KV cache buffer; the kernel takes `[N] u64` device
// arrays of per-slot K/V cache base pointers + a `[N] i32` array of
// per-slot write positions (tail index BEFORE this token's append).
//
// Layout assumption (F16Contig cache, mirrors the per-slot DtoD path):
//   K cache: [max_seq_len, kv_width] f16, kv_width = n_kv_heads * head_dim
//   V cache: same shape as K
//   Per slot, the new row lives at `dst + write_pos * kv_width` for both K and V.
//   Source: scratch.k_f16[N, kv_width] / scratch.v_f16[N, kv_width] — slot-major.
//
// Grid: (N, 1, 1). Block: 128 threads strided across `kv_width`. K and V
// copied in the same pass to halve loop overhead.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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
    // Ring-buffered SWA slab: the decode row wraps at `ring_depth`.
    // ring_depth = 0 → absolute addressing (bit-identical).
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
