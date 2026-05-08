// attention_decode_f16_batched — single-launch GQA decode attention over
// N (Q-row, per-slot KV-cache) pairs.
// Replaces the per-slot loop in `forward_full_attn_layer_decode_batched_*`,
// which structurally caps hybrid throughput at ~1.0× (per #267 cert).
// Each (q_head, slot) block runs the same flash-attn-v2 online-softmax
// body as `attention_decode_f16`; the slot index is carried through
// `blockIdx.y`, and per-slot KV-cache base pointers + KV-tail lengths
// are read from device-side tables built once per call by the dispatcher.
// Block math is bit-identical to `attention_decode_f16` for any single
// (q_head, slot) pair — the kernel reuses the same body verbatim with
// `q`, `k_cache`, `v_cache`, `out`, `n_tokens` rebound per block from
// the slot tables. Therefore N=1 dispatches produce bit-identical
// output to the single-slot kernel (regression guard for #266c wiring).
// Shapes:
// q_batched[N, n_heads_q, head_dim] F16
// k_cache_ptrs[N] u64 (device ptr per slot)
// v_cache_ptrs[N] u64 (device ptr per slot)
// out_batched[N, n_heads_q, head_dim] F16
// n_tokens_kv[N] i32 (per-slot KV tail)
// Per slot, the pointed buffers cover [n_tokens_kv[slot], n_heads_kv, head_dim] F16.
// Launch shape (caller-provided):
// blockDim = { head_dim } (one thread per output element)
// gridDim = { n_heads_q, n_slots }
// shared = q_shared[256] + out_shared[256] + score_parts[4]
// = 2064 bytes
// Supported head_dim: {64, 128, 256}. Block dim equals head_dim, so the
// number of wave64 warps is head_dim / 64 ∈ {1, 2, 4}.
// Correctness oracle: per-slot loop of `attention_decode_f16` (already certed).
// Element-wise max delta tolerance is the flash-attn-v2 reordering bound
// (~1e-3 in F16 — same bar as `attention_decode_f16_splitk`).

#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

typedef _Float16 fb_fp16_t;

// Max head_dim the kernel tolerates. Must match the dispatcher / ops-layer
// guard. Bump alongside both, never silently.
#define ATTN_MAX_HEAD_DIM 256
// Max warps per block = ATTN_MAX_HEAD_DIM / wave64.
#define ATTN_MAX_WARPS (ATTN_MAX_HEAD_DIM / 64)

extern "C" __global__ void flambeau_attention_decode_f16_batched(
    const fb_fp16_t* __restrict__ q_batched,         // [n_slots, n_heads_q, head_dim]
    const uint64_t* __restrict__ k_cache_ptrs,       // [n_slots] device pointers
    const uint64_t* __restrict__ v_cache_ptrs,       // [n_slots] device pointers
    fb_fp16_t* __restrict__ out_batched,             // [n_slots, n_heads_q, head_dim]
    const int* __restrict__ n_tokens_kv,             // [n_slots] per-slot KV tail
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,                              // 64, 128, or 256
    const int n_slots,
    const float scale                                // 1 / sqrt(head_dim)
) {
    const int q_head = blockIdx.x;
    const int slot   = blockIdx.y;
    if (q_head >= n_heads_q) return;
    if (slot   >= n_slots)   return;

    const int group   = n_heads_q / n_heads_kv;
    const int kv_head = q_head / group;

    const int tid     = threadIdx.x;
    const int warp    = tid >> 6;                    // 0..(n_warps-1)
    const int lane    = tid & 63;                    // 0..63
    const int n_warps = blockDim.x >> 6;             // {1, 2, 4} for head_dim ∈ {64, 128, 256}

    // Per-slot rebind of the kernel's read targets and loop bound.
    const fb_fp16_t* k_cache = (const fb_fp16_t*) k_cache_ptrs[slot];
    const fb_fp16_t* v_cache = (const fb_fp16_t*) v_cache_ptrs[slot];
    const int n_tokens       = n_tokens_kv[slot];

    // Q row for (slot, q_head). Stride = n_heads_q * head_dim per slot.
    const fb_fp16_t* q_row = q_batched
        + ((size_t) slot * n_heads_q + q_head) * (size_t) head_dim;
    fb_fp16_t* out_row = out_batched
        + ((size_t) slot * n_heads_q + q_head) * (size_t) head_dim;

    // --- Load Q for this (slot, q_head) ---
    __shared__ float q_shared[ATTN_MAX_HEAD_DIM];
    if (tid < head_dim) {
        q_shared[tid] = (float) q_row[tid];
    }

    // --- Running stats (per-block, replicated across threads) ---
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    __shared__ float out_shared[ATTN_MAX_HEAD_DIM];
    if (tid < head_dim) {
        out_shared[tid] = 0.0f;
    }
    __shared__ float score_parts[ATTN_MAX_WARPS];
    __syncthreads();

    // --- Inner loop over context positions ---
    for (int t = 0; t < n_tokens; ++t) {
        const size_t kv_row = ((size_t) t * n_heads_kv + kv_head) * head_dim;

        // 1. Compute Q · K[t, kv_head] cooperatively — each thread owns one lane.
        float my_partial = 0.0f;
        if (tid < head_dim) {
            my_partial = q_shared[tid] * (float) k_cache[kv_row + tid];
        }
        // Warp reduce across 64 lanes.
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            my_partial += __shfl_xor(my_partial, off, 64);
        }
        // Cross-warp sum: lane 0 of each warp writes, all threads sum.
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
            const float v_t = (float) v_cache[kv_row + tid];
            out_shared[tid] = out_shared[tid] * scale_old + coeff_t * v_t;
        }

        running_sum = running_sum * scale_old + coeff_t;
        running_max = new_max;

        __syncthreads();   // ensure out_shared updates visible before next iter
    }

    // --- Normalise & write ---
    if (tid < head_dim) {
        const float norm = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        out_row[tid] = (fb_fp16_t) (out_shared[tid] * norm);
    }
}
