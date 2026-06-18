#pragma once

// Single-block top-K + full-vocab softmax over a large vocabulary. Phase 1: the
// online (max, sum_exp) reduction rides block_reduce_softmax (WARP_SIZE seam).
// Phases 2-5: per-thread register top-K heaps → LDS → block bitonic sort →
// write top-K (id, full-vocab softmax prob). The heap + bitonic sort use only
// LDS + __syncthreads (no warp ops), so they are wave-agnostic as written.

#ifndef SAMPLER_INFINITY
#define SAMPLER_INFINITY __builtin_huge_valf()
#endif
#define SAMPLER_THREADS 256
#define SAMPLER_K_PER_THREAD 16
#define SAMPLER_K_MAX (SAMPLER_THREADS * SAMPLER_K_PER_THREAD)  // 4096
#define SAMPLER_K_OUT_MAX 2048

// Pack (val, idx) into a u64 so ascending u64 compare == ascending float, ties
// broken by lower idx. high32: monotone-by-float u32; low32: 0x7fffffff - idx.
static __device__ __forceinline__ unsigned long long sampler_pack_key(float v, int idx) {
    unsigned int b = __float_as_uint(v);
    b = (b & 0x80000000u) ? (~b) : (b | 0x80000000u);
    const unsigned int idx_inv = 0x7fffffffu - (unsigned int) idx;
    return ((unsigned long long) b << 32) | idx_inv;
}
static __device__ __forceinline__ int sampler_unpack_idx(unsigned long long key) {
    return (int) (0x7fffffffu - (unsigned int) (key & 0xffffffffu));
}
static __device__ __forceinline__ float sampler_unpack_val(unsigned long long key) {
    unsigned int b = (unsigned int) (key >> 32);
    b = (b & 0x80000000u) ? (b & 0x7fffffffu) : (~b);
    return __uint_as_float(b);
}

static __device__ void sampler_topk_softmax_f32_body(
    const float* __restrict__ logits,
    int* __restrict__ out_ids,
    float* __restrict__ out_probs,
    const int V,
    const int K,
    const float inv_temp
) {
    if (blockIdx.x != 0) return;
    if (K <= 0 || K > SAMPLER_K_OUT_MAX) return;
    const int tid = threadIdx.x;

    // Phase 1: online (max, sum_exp) over V.
    float local_max = -SAMPLER_INFINITY;
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
    __shared__ float scratch_max[SAMPLER_THREADS / WARP_SIZE];
    __shared__ float scratch_sum[SAMPLER_THREADS / WARP_SIZE];
    block_reduce_softmax<SAMPLER_THREADS>(local_max, local_sum, scratch_max, scratch_sum);
    const float row_max = local_max;
    const float row_sum = local_sum;

    // Phase 2: per-thread top-K_PER_THREAD, local_keys sorted ascending so
    // slot[0] is the eviction candidate.
    unsigned long long local_keys[SAMPLER_K_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < SAMPLER_K_PER_THREAD; ++i) {
        local_keys[i] = sampler_pack_key(-SAMPLER_INFINITY, 0x7fffffff);
    }
    for (int i = tid; i < V; i += SAMPLER_THREADS) {
        const float v = logits[i] * inv_temp;
        const unsigned long long cand = sampler_pack_key(v, i);
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

    // Phase 3: dump per-thread heaps into LDS.
    __shared__ unsigned long long s_keys[SAMPLER_K_MAX];
    #pragma unroll
    for (int i = 0; i < SAMPLER_K_PER_THREAD; ++i) {
        s_keys[tid * SAMPLER_K_PER_THREAD + i] = local_keys[i];
    }
    __syncthreads();

    // Phase 4: block bitonic sort, descending by key.
    const int N = SAMPLER_K_MAX;
    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int j = tid; j < N / 2; j += SAMPLER_THREADS) {
                const int pos = 2 * j - (j & (stride - 1));
                const int paired = pos | stride;
                if (paired < N) {
                    const bool descending = ((pos & size) == 0);
                    const unsigned long long a = s_keys[pos];
                    const unsigned long long b = s_keys[paired];
                    const bool swap = descending ? (a < b) : (a > b);
                    if (swap) {
                        s_keys[pos] = b;
                        s_keys[paired] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Phase 5: write top-K (id, full-vocab softmax prob).
    for (int slot = tid; slot < K; slot += SAMPLER_THREADS) {
        const unsigned long long key = s_keys[slot];
        out_ids[slot] = sampler_unpack_idx(key);
        out_probs[slot] = __expf(sampler_unpack_val(key) - row_max) / row_sum;
    }
}
