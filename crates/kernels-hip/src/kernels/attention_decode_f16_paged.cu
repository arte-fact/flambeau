// attention_decode_f16_paged — PagedAttention sibling of
// `attention_decode_f16_batched`. Same flash-attn-v2 online-softmax body;
// the only difference is the per-`t` K/V row address is resolved through
// a block-table indirection instead of a contiguous per-slot base
// pointer.
//
// Layout:
//   k_pool / v_pool: `[n_pages, page_size, kv_width]` f16 page pool,
//     shared across slots. kv_width = n_heads_kv * head_dim.
//   block_tables: `[max_slots, max_pages_per_slot]` u32 row-major.
//     For token `t` in slot `s`, the page that holds it is
//     `block_tables[s * max_pages_per_slot + t / page_size]`. The
//     in-page row is `t % page_size`. `page_size` is a power of two
//     so divide and modulo become shifts and AND masks at runtime.
//   q_batched / out_batched: `[n_slots, n_heads_q, head_dim]` f16 —
//     same as the contiguous batched kernel.
//   n_tokens_kv: `[n_slots]` i32 per-slot KV tail.
//
// Block layout:
//   blockDim = { head_dim }
//   gridDim  = { n_heads_q, n_slots }
//   shared   = q_shared[256] + out_shared[256] + score_parts[4] = 2064 B
// Supported head_dim ∈ {64, 128, 256}.
//
// VGPR delta vs the contiguous kernel: ~+3 (page_idx, page_offset,
// page) computed once per outer t iteration. Block-table memory
// pattern is sequential u32 reads — 4 bytes per `page_size = 16`
// inner iterations, fully amortised in L2.
//
// Correctness oracle: with `block_tables[s][p] = s * max_pages_per_slot
// + p` (identity map) and `n_pages = max_slots * max_pages_per_slot`,
// the output is bit-identical to `attention_decode_f16_batched`
// running on the same K/V data. That's the parity regression guard.

#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

typedef _Float16 fb_fp16_t;

// Max head_dim the kernel tolerates. Must match the dispatcher / ops-layer
// guard. Bump alongside both, never silently.
#define ATTN_MAX_HEAD_DIM 512
// Max warps per block = ATTN_MAX_HEAD_DIM / wave64.
#define ATTN_MAX_WARPS (ATTN_MAX_HEAD_DIM / 64)

extern "C" __global__ void flambeau_attention_decode_f16_paged(
    const fb_fp16_t* __restrict__ q_batched,         // [n_slots, n_heads_q, head_dim]
    const fb_fp16_t* __restrict__ k_pool,            // [n_pages, page_size, kv_width]
    const fb_fp16_t* __restrict__ v_pool,            // [n_pages, page_size, kv_width]
    const unsigned int* __restrict__ block_tables,   // [n_slots, max_pages_per_slot]
    fb_fp16_t* __restrict__ out_batched,             // [n_slots, n_heads_q, head_dim]
    const int* __restrict__ n_tokens_kv,             // [n_slots] per-slot KV tail
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,                              // 64, 128, or 256
    const int n_slots,
    const int page_size,
    const int max_pages_per_slot,
    const float scale                                // 1 / sqrt(head_dim)
) {
    const int q_head = blockIdx.x;
    const int slot   = blockIdx.y;
    if (q_head >= n_heads_q) return;
    if (slot   >= n_slots)   return;

    const int group   = n_heads_q / n_heads_kv;
    const int kv_head = q_head / group;

    const int tid     = threadIdx.x;
    const int warp    = tid >> 6;
    const int lane    = tid & 63;
    const int n_warps = blockDim.x >> 6;

    const int n_tokens = n_tokens_kv[slot];
    const int kv_width = n_heads_kv * head_dim;

    const unsigned int* slot_table =
        block_tables + (size_t) slot * max_pages_per_slot;

    const fb_fp16_t* q_row = q_batched
        + ((size_t) slot * n_heads_q + q_head) * (size_t) head_dim;
    fb_fp16_t* out_row = out_batched
        + ((size_t) slot * n_heads_q + q_head) * (size_t) head_dim;

    // --- Load Q for this (slot, q_head) ---
    __shared__ float q_shared[ATTN_MAX_HEAD_DIM];
    if (tid < head_dim) {
        q_shared[tid] = (float) q_row[tid];
    }

    // --- Running stats ---
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    __shared__ float out_shared[ATTN_MAX_HEAD_DIM];
    if (tid < head_dim) {
        out_shared[tid] = 0.0f;
    }
    __shared__ float score_parts[ATTN_MAX_WARPS];
    __syncthreads();

    // --- Inner loop over context positions ---
    // We assume `page_size` is a power of two. The dispatcher rejects
    // non-power-of-two page sizes, so the divides and modulos below
    // compile to shifts and AND masks on a power-of-two `page_size`.
    const int page_mask = page_size - 1;
    for (int t = 0; t < n_tokens; ++t) {
        const int page_idx_in_slot = t / page_size;
        const int p_off            = t & page_mask;
        const unsigned int page    = slot_table[page_idx_in_slot];

        const size_t kv_row = ((size_t) page * page_size + p_off) * kv_width
                            + (size_t) kv_head * head_dim;

        // 1. Q · K[t, kv_head]
        float my_partial = 0.0f;
        if (tid < head_dim) {
            my_partial = q_shared[tid] * (float) k_pool[kv_row + tid];
        }
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            my_partial += __shfl_xor(my_partial, off, 64);
        }
        if (lane == 0) {
            score_parts[warp] = my_partial;
        }
        __syncthreads();
        float score_t = 0.0f;
        #pragma unroll
        for (int w = 0; w < ATTN_MAX_WARPS; ++w) {
            if (w < n_warps) score_t += score_parts[w];
        }
        score_t *= scale;

        // 2. Online softmax update.
        float new_max   = fmaxf(running_max, score_t);
        float scale_old = __expf(running_max - new_max);
        float coeff_t   = __expf(score_t - new_max);

        // 3. Per-element update of running_out.
        if (tid < head_dim) {
            const float v_t = (float) v_pool[kv_row + tid];
            out_shared[tid] = out_shared[tid] * scale_old + coeff_t * v_t;
        }

        running_sum = running_sum * scale_old + coeff_t;
        running_max = new_max;

        __syncthreads();
    }

    if (tid < head_dim) {
        const float norm = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        out_row[tid] = (fb_fp16_t) (out_shared[tid] * norm);
    }
}
