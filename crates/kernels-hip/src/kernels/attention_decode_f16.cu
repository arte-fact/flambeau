// Supported head_dim values: {64, 128, 256, 512}. The caller launches with
// `block = head_dim` threads (1 wave64 at d=64, 2 at d=128, 4 at d=256,
// 8 at d=512). LDS scales with ATTN_MAX_HEAD_DIM (~4 KB at 512 vs 2 KB
// at 256) — well under the 64 KB/CU budget; LDS-limited occupancy is
// not the binding constraint at any of these head_dims.
//
// attention_decode_f16 — GQA decode attention, F16 KV cache.
// Fused decode-step attention: Q has 1 token, K and V cover the whole
// context. For each Q head q_idx (of n_heads_q) we compute:
// scores[t] = scale * (Q[q_idx] · K[t, kv_head_of(q_idx)])
// out[q_idx, :] = Σ_t softmax(scores)[t] * V[t, kv_head_of(q_idx), :]
// Where `kv_head_of(q) = q / (n_heads_q / n_heads_kv)`.
// Online (flash-attn-v2) softmax: running (max, sum, out) stats so we
// never materialise the [n_tokens] scores array in LDS. That makes
// long-context decode (≥32k tokens) fit regardless of LDS budget.
// Supported head_dim values: {128, 256}. The caller launches with
// `block = head_dim` threads; the kernel reduces across the resulting
// wave64 warps ({2, 4}) via LDS `score_parts`. Q and out shared arrays
// are sized to the max (256) so the same hsaco serves both — at
// head_dim=128 the upper half is simply unused (still cheap in LDS on
// gfx906: ~2 KB per block, well under the 64 KB budget).
// Launch shape (caller-provided):
// blockDim = { head_dim } (one thread per output element)
// gridDim = { n_heads_q }
// shared = q_shared[256] + out_shared[256] + score_parts[4] =
// (256+256)*4 + 4*4 = 2064 bytes
// head_dim=128 perf note: block size drops from "hardcoded 128" to
// "= head_dim" — same 2 wave64 warps, identical occupancy. The added
// zero-init of the unused out_shared upper half is 64 thread-cycles per
// launch; noise vs the LDS/HBM costs.
// Correctness oracle: decomposed F32 reference (CPU) in the cert harness.
// The harness exercises both head_dim=128 (Qwen3.5 / GQA-32/4) and
// head_dim=256 (Qwen3.6 / GQA-16/2).

#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

typedef _Float16 fb_fp16_t;

// Max head_dim the kernel tolerates. Bump alongside the dispatcher /
// ops-layer guard + new cert shapes, never silently.
#define ATTN_MAX_HEAD_DIM 512
// Max warps per block = ATTN_MAX_HEAD_DIM / wave64.
#define ATTN_MAX_WARPS (ATTN_MAX_HEAD_DIM / 64)

extern "C" __global__ void flambeau_attention_decode_f16(
    const fb_fp16_t* __restrict__ q,          // [n_heads_q, head_dim]
    const fb_fp16_t* __restrict__ k_cache,    // [n_tokens, n_heads_kv, head_dim]
    const fb_fp16_t* __restrict__ v_cache,    // [n_tokens, n_heads_kv, head_dim]
    fb_fp16_t* __restrict__ out,              // [n_heads_q, head_dim]
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,                       // must be 64, 128, 256, or 512
    const int n_tokens,
    const float scale,                        // 1 / sqrt(head_dim) (caller-supplied; gemma4 passes 1.0)
    const int window_size                     // SWA radius, 0 = unbounded causal
) {
    const int q_head  = blockIdx.x;
    if (q_head >= n_heads_q) return;
    const int group  = n_heads_q / n_heads_kv;
    const int kv_head = q_head / group;

    const int tid   = threadIdx.x;
    const int warp  = tid >> 6;                   // 0..(n_warps-1)
    const int lane  = tid & 63;                   // 0..63
    const int n_warps = blockDim.x >> 6;          // {1,2,4,8} for head_dim ∈ {64,128,256,512}

    // --- Load Q for this head ---
    __shared__ float q_shared[ATTN_MAX_HEAD_DIM];
    if (tid < head_dim) {
        q_shared[tid] = (float) q[(size_t) q_head * head_dim + tid];
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

    // SWA: query position is the last token in the cache (n_tokens - 1).
    // Keys older than (query_pos - window_size + 1) are masked. window_size=0
    // disables the window — full causal range.
    int t_start = 0;
    if (window_size > 0) {
        const int qpos = n_tokens - 1;
        t_start = qpos - window_size + 1;
        if (t_start < 0) t_start = 0;
    }

    // --- Inner loop over context positions ---
    for (int t = t_start; t < n_tokens; ++t) {
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
        // Cross-warp sum: lane 0 of each warp writes, warp 0 sums all.
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
        const float norm = 1.0f / running_sum;
        out[(size_t) q_head * head_dim + tid] =
            (fb_fp16_t) (out_shared[tid] * norm);
    }
}
