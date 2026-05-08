// quantize_f16_q8_0 — F16 input → Q8_0 blocks.
// Sibling of `quantize_f16_q8_1` — drops the
// `s` (sum) field so the output matches `flambeau_block_q8_0`'s 18 B
// layout (used by the Q8 KV cache + the `attention_decode_q8_kv` kernel).
// Grid: one thread block per 32 elements. 32 threads per block.

#include "block_quant.cuh"

extern "C" __global__ void flambeau_quantize_row_f16_q8_0(
    const fb_fp16_t* __restrict__ x,
    flambeau_block_q8_0* __restrict__ y,
    const int n_elems
) {
    const int ib = blockIdx.x;
    const int t  = threadIdx.x;
    const int base = ib * QK8_0;
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

    if (t == 0) {
        y[ib].d = (fb_fp16_t) d;
    }
    y[ib].qs[t] = (int8_t) qi;
}
