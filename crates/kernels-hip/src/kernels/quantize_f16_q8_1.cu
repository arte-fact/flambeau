// quantize_f16_q8_1 — F16 activation → Q8_1 blocks.
//
// F16 sibling of quantize_q8_1 (which takes F32 input). V1.7.3-g: replaces
// the host-roundtrip placeholder in full_attn forward (F16 swiglu output
// → host F32 → device F32 → Q8_1) with one on-device launch.
//
// Same math as the F32 variant — upcast to float for the amax/reduce so
// denormals don't bite us.
//
// Grid: one thread block per 32 elements. 32 threads per block.

#include "block_quant.cuh"

extern "C" __global__ void flambeau_quantize_row_f16_q8_1(
    const fb_fp16_t* __restrict__ x,
    flambeau_block_q8_1* __restrict__ y,
    const int n_elems
) {
    const int ib = blockIdx.x;
    const int t  = threadIdx.x;
    const int base = ib * QK8_1;
    if (base + t >= n_elems) return;

    const float xi = (float) x[base + t];
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
