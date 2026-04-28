// swiglu_f32_to_q8_1 — CN-80B-19c/d fused `silu(a) * b` (F32 inputs) → Q8_1
// blocks, replacing two unfused chains:
//
//   GDN tail (forward/gdn.rs):
//     swiglu_f32(z, out_normed)         → gated_f32  (1 launch + 1 HBM RT)
//     quantize_row_q8_1(gated_f32)      → gated_q8_1 (1 launch)
//
//   Shared expert (forward/moe.rs):
//     swiglu_f32_to_f16(gate_f32, up_f32) → activated_f16  (1 launch + 1 HBM RT)
//     quantize_row_f16_q8_1(activated_f16) → activated_q8_1 (1 launch)
//
// In both cases the intermediate buffer is consumed only by the next
// step, so fusing eliminates one launch + one HBM round-trip per layer
// per token.
//
// Layout matches `quantize_row_q8_1`: one thread block per 32-element
// Q8_1 block, 32 threads per block, lane t owns element t. Math is
// `silu(a[i]) * b[i]` per lane, then per-block amax / sum reductions
// for the Q8_1 scale + sum fields.

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
