// swiglu_f32_to_q8_1 — fused `silu(a) * b` (F32 in) → Q8_1 blocks. One block
// per 32-element Q8_1 block, 32 threads, lane t owns element t. The amax/sum
// reductions are width-32 __shfl_xor (one warp on wave32, a sub-warp on
// wave64).

#include "block_quant.cuh"

extern "C" __global__ void flambeau_swiglu_f32_to_q8_1(
    const float* __restrict__ a,
    const float* __restrict__ b,
    flambeau_block_q8_1* __restrict__ y,
    const int n_elems
) {
    const int ib = blockIdx.x;
    const int t  = threadIdx.x;
    const int base = ib * QK8_1;

    if (base + t >= n_elems) return;

    const float av = a[base + t];
    const float bv = b[base + t];
    const float silu = av / (1.0f + __expf(-av));
    const float xi   = silu * bv;

    float amax = fabsf(xi);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor(amax, offset, 32);
        amax = fmaxf(amax, other);
    }

    const float d  = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    const int qi = min(127, max(-127, (int) rintf(xi * id)));

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
