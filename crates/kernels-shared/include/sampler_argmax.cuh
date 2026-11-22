#pragma once
// Greedy draw on device, matching the host `argmax` of flambeau-runtime bit
// for bit: sparse additive adjustments applied in list order by one thread
// (the same float sequence as the host loop), then an argmax in two passes.
// A value is a candidate only if it is > -inf (NaN and -inf never win), ties
// go to the lowest index, and no candidate at all yields index 0.

#define SAMPLER_ARGMAX_THREADS 256
#define SAMPLER_ARGMAX_NONE    0x7FFFFFFF
#define SAMPLER_ARGMAX_NEG_INF (-__builtin_huge_valf())

static __device__ __forceinline__ void sampler_sparse_add_f32_body(
    float* __restrict__ logits, const int* __restrict__ ids, const float* __restrict__ vals, int n)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < n; ++i) logits[ids[i]] += vals[i];
    }
}

static __device__ __forceinline__ bool sampler_argmax_better(float va, int ia, float vb, int ib) {
    return va > vb || (va == vb && ia < ib);
}

// Block-wide reduction of (v, i) pairs; every thread returns the block winner.
static __device__ __forceinline__ void sampler_argmax_block(float& v, int& i) {
    __shared__ float s_v[SAMPLER_ARGMAX_THREADS];
    __shared__ int   s_i[SAMPLER_ARGMAX_THREADS];
    const int t = threadIdx.x;
    s_v[t] = v;
    s_i[t] = i;
    __syncthreads();
    for (int s = SAMPLER_ARGMAX_THREADS / 2; s > 0; s >>= 1) {
        if (t < s && sampler_argmax_better(s_v[t + s], s_i[t + s], s_v[t], s_i[t])) {
            s_v[t] = s_v[t + s];
            s_i[t] = s_i[t + s];
        }
        __syncthreads();
    }
    v = s_v[0];
    i = s_i[0];
}

// Pass 1: one (value, index) winner per block over a grid-strided range.
static __device__ __forceinline__ void sampler_argmax_partial_f32_body(
    const float* __restrict__ logits, int vocab, float* __restrict__ part_v, int* __restrict__ part_i)
{
    float best_v = SAMPLER_ARGMAX_NEG_INF;
    int   best_i = SAMPLER_ARGMAX_NONE;
    for (int j = blockIdx.x * SAMPLER_ARGMAX_THREADS + threadIdx.x; j < vocab;
         j += gridDim.x * SAMPLER_ARGMAX_THREADS) {
        const float v = logits[j];
        if (v > SAMPLER_ARGMAX_NEG_INF && sampler_argmax_better(v, j, best_v, best_i)) {
            best_v = v;
            best_i = j;
        }
    }
    sampler_argmax_block(best_v, best_i);
    if (threadIdx.x == 0) {
        part_v[blockIdx.x] = best_v;
        part_i[blockIdx.x] = best_i;
    }
}

// Pass 2: one block reduces the n_parts block winners into out[0].
static __device__ __forceinline__ void sampler_argmax_final_f32_body(
    const float* __restrict__ part_v, const int* __restrict__ part_i, int n_parts, int* __restrict__ out)
{
    float best_v = SAMPLER_ARGMAX_NEG_INF;
    int   best_i = SAMPLER_ARGMAX_NONE;
    for (int j = threadIdx.x; j < n_parts; j += SAMPLER_ARGMAX_THREADS) {
        if (sampler_argmax_better(part_v[j], part_i[j], best_v, best_i)) {
            best_v = part_v[j];
            best_i = part_i[j];
        }
    }
    sampler_argmax_block(best_v, best_i);
    if (threadIdx.x == 0) out[0] = best_i == SAMPLER_ARGMAX_NONE ? 0 : best_i;
}
