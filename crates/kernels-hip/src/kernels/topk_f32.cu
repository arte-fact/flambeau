// topk_f32 — per-token top-k selection over F32 logits.
// For each token, pick the k largest logits; return their expert ids +
// softmax-normalised weights over the selected k. No candle prior art —
// Qwen3.x MoE uses 128–256 experts × top-8 which is small enough for an
// iterative find-max + mask kernel (the dedicated bitonic top-k path
// is a perf follow-up).
// Layout:
// blockDim = { n_experts } (128 for Qwen3.5, 256 for Qwen3.6)
// gridDim = { n_tokens }
// shared = n_experts floats + 2 stats + k floats
// Supported `n_experts` ceiling: TOPK_MAX_EXPERTS (compile-time). Bumping
// this requires a matching bump in `s_logits`, `s_max_v`, `s_max_i` +
// a cert shape at the new ceiling — do not silently raise.
// Algorithm:
// Load all logits into LDS. Loop k times: warp-reduce find-max across
// the block, record (index, value), write -INF at the winner, repeat.
// After the k-loop, softmax the k captured raw values → weights.
// Stability: ties are broken by lower-index-wins (via explicit
// `idx_other < local_idx` in the reduce) so CPU and GPU agree.

#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

#define TOPK_MAX_K 16
// Qwen3.6 is 256 experts top-8. Qwen3-Coder-Next-80B (qwen3next) is
// 512 experts top-10 → bumped from 256 to 512.
#define TOPK_MAX_EXPERTS 512
#define TOPK_MAX_WARPS (TOPK_MAX_EXPERTS / 64)  // 8 at EXPERTS=512

extern "C" __global__ void flambeau_topk_softmax_f32(
    const float* __restrict__ logits,     // [n_tokens, n_experts]
    int*   __restrict__ out_indices,      // [n_tokens, k]
    float* __restrict__ out_weights,      // [n_tokens, k]
    const int n_tokens,
    const int n_experts,
    const int k
) {
    const int token = blockIdx.x;
    if (token >= n_tokens) return;

    const int tid     = threadIdx.x;
    const int warp    = tid >> 6;
    const int lane    = tid & 63;
    const int n_warps = blockDim.x >> 6;  // 2 at EXPERTS=128, 4 at 256

    __shared__ float s_logits[TOPK_MAX_EXPERTS];
    __shared__ float top_vals[TOPK_MAX_K];
    __shared__ int   top_idxs[TOPK_MAX_K];
    // Cross-warp max-reduce scratch — one slot per warp.
    __shared__ float s_max_v[TOPK_MAX_WARPS];
    __shared__ int   s_max_i[TOPK_MAX_WARPS];

    // Load this thread's logit into LDS. Threads outside n_experts fill with -INF
    // so the reduce ignores them.
    if (tid < TOPK_MAX_EXPERTS) {
        float v = (tid < n_experts)
            ? logits[(size_t) token * n_experts + tid]
            : -INFINITY;
        s_logits[tid] = v;
    }
    __syncthreads();

    // --- k rounds of "find max + mask" ---
    for (int round = 0; round < k; ++round) {
        float local_val = (tid < TOPK_MAX_EXPERTS) ? s_logits[tid] : -INFINITY;
        int   local_idx = tid;

        // Warp reduce (full wave64).
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            float other_v   = __shfl_xor(local_val, off, 64);
            int   other_idx = __shfl_xor(local_idx, off, 64);
            // Lower-index-wins for stability on equal values.
            if (other_v > local_val ||
                (other_v == local_val && other_idx < local_idx)) {
                local_val = other_v;
                local_idx = other_idx;
            }
        }

        // Cross-warp via LDS — `n_warps` slots.
        if (lane == 0) {
            s_max_v[warp] = local_val;
            s_max_i[warp] = local_idx;
        }
        __syncthreads();

        float best_v = s_max_v[0];
        int   best_i = s_max_i[0];
        #pragma unroll
        for (int w = 1; w < TOPK_MAX_WARPS; ++w) {
            if (w >= n_warps) continue;
            const float v = s_max_v[w];
            const int   i = s_max_i[w];
            if (v > best_v || (v == best_v && i < best_i)) {
                best_v = v;
                best_i = i;
            }
        }

        // Thread 0 records the winner for this round.
        if (tid == 0) {
            top_vals[round] = best_v;
            top_idxs[round] = best_i;
        }
        // Mask the winner so the next round picks a different expert.
        __syncthreads();
        if (tid == best_i) {
            s_logits[tid] = -INFINITY;
        }
        __syncthreads();
    }

    // --- Softmax over the k captured raw values ---
    if (tid < k) {
        // Single-thread stable softmax: k is at most TOPK_MAX_K = 16 so
        // scalar math is fine. All k threads compute the same result in
        // parallel.
        float m = top_vals[0];
        for (int i = 1; i < k; ++i) {
            if (top_vals[i] > m) m = top_vals[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += __expf(top_vals[i] - m);
        }
        const float my_w = __expf(top_vals[tid] - m) / sum;
        out_indices[(size_t) token * k + tid] = top_idxs[tid];
        out_weights[(size_t) token * k + tid] = my_w;
    }
}

// Note: `softmax-then-topk-then-renormalize` is mathematically
// equivalent to `topk-then-softmax-of-k` (the algorithm above) when
// the renormalize step divides by the sum of top-k softmax probs.
// Proof:
//   softmax-then-topk: p_i = exp(l_i - M) / Σ_all exp(l_j - M); take
//   top-k by p_i (≡ top-k by l_i since softmax monotonic).
//   Renormalize: np_i = p_i / Σ_topk p_j
//                     = exp(l_i - M) / Σ_topk exp(l_j - M).
//   Equivalent to softmax-of-top-k raw logits (max(l in topk) == M
//   because top-k contains the global max).
// So gemma4 `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` with `norm_w=true`
// (its actual setting per `src/models/gemma4-iswa.cpp:165`) IS what
// `flambeau_topk_softmax_f32` above computes. No new kernel needed.
