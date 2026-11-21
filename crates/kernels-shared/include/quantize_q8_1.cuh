#pragma once
// F32 activation -> Q8_1 block quantization, shared by both backends.
// One thread block per QK8_1 (=32) elements; 32 active threads, each owns one
// element and one reduce across the block:
//   d = max(|x|) / 127
//   qs[j] = round(x[j] / d)   (clamped to [-127, 127])
//   s = d * sum(qs)           (used by the Q4_1/Q5_1/Q8_1 vec_dot `s` term)
// Backend-neutral (architectural rule 5): the only cross-lane op is
// `__shfl_xor`, which backend_compat.cuh maps to the sync form on CUDA and is
// native on HIP. fb_fp16_t + the runtime header arrive via block_quant.cuh.
#include "block_quant.cuh"

static __device__ __forceinline__ void quantize_q8_1_block(
    const float* __restrict__ x,
    flambeau_block_q8_1* __restrict__ y,
    int n_elems)
{
    const int ib   = blockIdx.x;        // block index
    const int t    = threadIdx.x;       // 0..31 lane-in-block
    const int base = ib * QK8_1;

    if (base + t >= n_elems) return;

    const float xi = x[base + t];
    float amax = fabsf(xi);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }

    const float d  = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const int   qi = min(127, max(-127, (int) rintf(xi * id)));

    int sum = qi;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, 32);
    }

    if (t == 0) {
        y[ib].d = (fb_fp16_t) d;
        y[ib].s = (fb_fp16_t) (d * (float) sum);
    }
    y[ib].qs[t] = (int8_t) qi;
}
