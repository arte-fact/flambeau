// sampler_topk_softmax_f32 — single-block top-K + softmax-normalize over a
// large vocabulary (V up to ~256k). Designed for the chat decoder's
// per-token sampler hot path so the host doesn't have to download all V
// logits and run an O(V log V) sort in CPU code.
//
// Inputs:
//   logits     [V] F32                   penalty-adjusted decoder logits
//   inv_temp   scalar                    1.0 / temperature (or 1.0 if temp <= 0)
//   V          int                       vocab size
//   K          int                       top-K to return (must satisfy 1 <= K <= 256)
// Outputs (only first K entries written, sorted by descending prob):
//   out_ids    [K] i32                   token ids, sorted descending by prob
//   out_probs  [K] F32                   softmax-normalised probabilities,
//                                        renormalised to sum to ~1 over the K kept
//
// Launch shape:
//   blockDim = 256, gridDim = 1
//
// Algorithm:
//   1. Online (max, sum_exp) reduction over the full V via grid-stride loop +
//      warp/block tree reduction (same recurrence as softmax_masked_f16).
//   2. Each thread walks V/256 ≈ V/blockDim logits a SECOND time and keeps a
//      register-resident top-K_PER_THREAD (id, value) heap. Min-heap stored as
//      a flat array; insert-and-shuffle on each new candidate.
//   3. All threads dump their K_PER_THREAD candidates into shared memory
//      (256 * K_PER_THREAD = 4096 slots).
//   4. Block-level bitonic sort on the 4096 candidates → descending order.
//   5. First K threads (tid < K) compute prob = exp(value * inv_temp - max) /
//      sum_exp and write (id, prob) to out_*. We renormalise at the end so
//      probs over the kept K sum to 1.
//
// Why a single block:
//   K up to 256, V up to ~256k → 4096 candidates fits comfortably in 64 KB LDS
//   (4096 * 8 = 32 KB). Avoids cross-block synchronisation. The kernel is
//   bandwidth-bound on the V logit reads (~600 KB at V=151k), which the
//   gfx906 HBM2 ~1 TB/s line-rate handles in ~600 µs. Adding more blocks
//   would parallelise the reads but require a second merge kernel — not
//   worth it for V ≤ 256k.
//
// Stability:
//   Ties on logit values are broken by lower-id-wins (encoded into the sort
//   key as a packed (val, idx) where the high 32 bits are the float bit
//   pattern with sign flip and the low 32 bits are the negated index).

#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

#define SAMPLER_THREADS 256
#define SAMPLER_WARPS (SAMPLER_THREADS / 64)
#define SAMPLER_K_PER_THREAD 16
#define SAMPLER_K_MAX (SAMPLER_THREADS * SAMPLER_K_PER_THREAD)  // 4096
// Caller-visible upper bound on K. Higher K would need a different layout.
#define SAMPLER_K_OUT_MAX 256

// Encode (val, idx) into a single u64 such that ASCENDING uint64 compare
// matches ASCENDING float-value compare, with TIES on value broken by
// LOWER idx winning (i.e., for two equal floats, the entry with the
// smaller idx gets the LARGER packed key — and so wins under descending
// sort, which is how the bitonic stage below sorts).
//
// Layout:
//   high 32: monotone-by-float u32 — positive: set MSB; negative: invert
//            all bits. Result: -INF maps to 0x00000000, -0.0 to
//            0x7FFFFFFF, +0.0 to 0x80000000, +INF to 0xFF800000. So
//            ascending u32 = ascending float across the zero crossing.
//   low 32:  (0x7FFFFFFF - idx) so that ASCENDING low32 = DESCENDING idx.
//            With the sort below in DESCENDING-by-key order, this means
//            equal floats are emitted with LOWER idx first.
__device__ __forceinline__ unsigned long long pack_key(float v, int idx) {
    unsigned int b = __float_as_uint(v);
    b = (b & 0x80000000u) ? (~b) : (b | 0x80000000u);
    unsigned int idx_inv = 0x7fffffffu - (unsigned int) idx;
    return ((unsigned long long) b << 32) | idx_inv;
}

__device__ __forceinline__ int unpack_idx(unsigned long long key) {
    unsigned int idx_inv = (unsigned int) (key & 0xffffffffu);
    return (int) (0x7fffffffu - idx_inv);
}

__device__ __forceinline__ float unpack_val(unsigned long long key) {
    unsigned int b = (unsigned int) (key >> 32);
    b = (b & 0x80000000u) ? (b & 0x7fffffffu) : (~b);
    return __uint_as_float(b);
}

extern "C" __global__ void flambeau_sampler_topk_softmax_f32(
    const float* __restrict__ logits,    // [V]
    int*   __restrict__ out_ids,         // [K]
    float* __restrict__ out_probs,       // [K]
    const int V,
    const int K,
    const float inv_temp
) {
    if (blockIdx.x != 0) return;
    if (K <= 0 || K > SAMPLER_K_OUT_MAX) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    // ------------------------------------------------------------------
    // Phase 1: online (max, sum_of_exp) reduction over V logits.
    // ------------------------------------------------------------------
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    for (int i = tid; i < V; i += SAMPLER_THREADS) {
        const float v = logits[i] * inv_temp;
        if (v > local_max) {
            local_sum = local_sum * __expf(local_max - v) + 1.0f;
            local_max = v;
        } else {
            local_sum += __expf(v - local_max);
        }
    }

    // Warp reduce (wave64).
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        const float other_m = __shfl_xor(local_max, off, 64);
        const float other_s = __shfl_xor(local_sum, off, 64);
        const float new_m = fmaxf(local_max, other_m);
        local_sum = local_sum * __expf(local_max - new_m)
                  + other_s    * __expf(other_m  - new_m);
        local_max = new_m;
    }

    // Cross-warp reduce via LDS.
    __shared__ float s_max[SAMPLER_WARPS];
    __shared__ float s_sum[SAMPLER_WARPS];
    if (lane == 0) {
        s_max[warp] = local_max;
        s_sum[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float wm = (lane < SAMPLER_WARPS) ? s_max[lane] : -INFINITY;
        float ws = (lane < SAMPLER_WARPS) ? s_sum[lane] : 0.0f;
        #pragma unroll
        for (int off = SAMPLER_WARPS / 2; off > 0; off >>= 1) {
            const float om = __shfl_xor(wm, off, 64);
            const float os = __shfl_xor(ws, off, 64);
            const float new_m = fmaxf(wm, om);
            ws = ws * __expf(wm - new_m) + os * __expf(om - new_m);
            wm = new_m;
        }
        if (lane == 0) {
            s_max[0] = wm;
            s_sum[0] = ws;
        }
    }
    __syncthreads();
    const float row_max = s_max[0];
    const float row_sum = s_sum[0];

    // ------------------------------------------------------------------
    // Phase 2: per-thread top-K_PER_THREAD via register-resident sorted
    //          array. We want to keep the K_PER_THREAD candidates with
    //          the SMALLEST packed keys (= largest float values), so we
    //          maintain the array sorted DESCENDING by key (slot[0] =
    //          worst-kept = largest packed key = smallest float value).
    //          On new candidate: replace slot[0] if cand has a smaller
    //          key (= larger value), then bubble it down to its sorted
    //          position. K_PER_THREAD is small (16) so the linear scan
    //          beats heapify overhead.
    // ------------------------------------------------------------------
    // Convention: keep the K_PER_THREAD candidates with the LARGEST
    // packed keys. local_keys is sorted ASCENDING so slot[0] is the
    // smallest kept key (= worst kept = eviction candidate). New
    // candidate displaces slot[0] iff cand > slot[0], then bubbles up
    // to maintain ascending order.
    unsigned long long local_keys[SAMPLER_K_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < SAMPLER_K_PER_THREAD; ++i) {
        // pack_key(-INF, MAX_IDX) is the SMALLEST possible key — any
        // real candidate will displace these. (Note: with our new
        // ascending-by-value encoding, -INF maps to 0x00000000 in the
        // high 32 bits, so this is genuinely the smallest u64 key.)
        local_keys[i] = pack_key(-INFINITY, 0x7fffffff);
    }

    for (int i = tid; i < V; i += SAMPLER_THREADS) {
        const float v = logits[i] * inv_temp;
        const unsigned long long cand = pack_key(v, i);
        if (cand > local_keys[0]) {
            local_keys[0] = cand;
            int j = 0;
            while (j + 1 < SAMPLER_K_PER_THREAD && local_keys[j] > local_keys[j + 1]) {
                const unsigned long long t = local_keys[j];
                local_keys[j] = local_keys[j + 1];
                local_keys[j + 1] = t;
                ++j;
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 3: dump per-thread heaps into shared memory.
    // Layout: s_keys[tid * K_PER_THREAD + slot].
    // ------------------------------------------------------------------
    __shared__ unsigned long long s_keys[SAMPLER_K_MAX];
    #pragma unroll
    for (int i = 0; i < SAMPLER_K_PER_THREAD; ++i) {
        s_keys[tid * SAMPLER_K_PER_THREAD + i] = local_keys[i];
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 4: block-level bitonic sort, DESCENDING by packed key
    //          (largest key first ⇔ largest float value first).
    // ------------------------------------------------------------------
    // 4096 elements, log2 = 12 levels. SAMPLER_THREADS=256 threads each
    // handle SAMPLER_K_MAX/2/SAMPLER_THREADS = 8 compare-swaps per stage.
    // Standard bitonic: outer "size" doubles, inner "stride" halves.
    // The `descending` direction at each stage is inverted from the
    // textbook ascending sort.
    const int N = SAMPLER_K_MAX;
    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int j = tid; j < N / 2; j += SAMPLER_THREADS) {
                const int pos    = 2 * j - (j & (stride - 1));
                const int paired = pos | stride;
                if (paired < N) {
                    // For DESCENDING sort, invert the ascending
                    // direction at every stage.
                    const bool descending = ((pos & size) == 0);
                    const unsigned long long a = s_keys[pos];
                    const unsigned long long b = s_keys[paired];
                    const bool swap = descending ? (a < b) : (a > b);
                    if (swap) {
                        s_keys[pos]    = b;
                        s_keys[paired] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    // After the sort, s_keys is descending → s_keys[0..K] are the largest
    // packed keys, i.e., the K largest float values.

    // ------------------------------------------------------------------
    // Phase 5: compute renormalised probs over the kept K.
    // ------------------------------------------------------------------
    // First pass: each thread computes its own kept-token's exp share and
    // we reduce the sum across threads.
    float my_share = 0.0f;
    if (tid < K) {
        const unsigned long long key = s_keys[tid];
        const float v = unpack_val(key);
        my_share = __expf(v - row_max) / row_sum;
    }
    // Warp + block reduce sum of `my_share`.
    float kept_sum = my_share;
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        kept_sum += __shfl_xor(kept_sum, off, 64);
    }
    __shared__ float s_kept_sum[SAMPLER_WARPS];
    if (lane == 0) {
        s_kept_sum[warp] = kept_sum;
    }
    __syncthreads();
    if (warp == 0) {
        float ws = (lane < SAMPLER_WARPS) ? s_kept_sum[lane] : 0.0f;
        #pragma unroll
        for (int off = SAMPLER_WARPS / 2; off > 0; off >>= 1) {
            ws += __shfl_xor(ws, off, 64);
        }
        if (lane == 0) {
            s_kept_sum[0] = ws;
        }
    }
    __syncthreads();
    const float inv_kept = 1.0f / s_kept_sum[0];

    if (tid < K) {
        const unsigned long long key = s_keys[tid];
        const int idx = unpack_idx(key);
        const float v = unpack_val(key);
        const float p = __expf(v - row_max) / row_sum;
        out_ids[tid]   = idx;
        out_probs[tid] = p * inv_kept;
    }
}
