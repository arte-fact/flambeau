// quantize_q8_1 — F32 activation → Q8_1 blocks.
//
// Each block of QK8_1 (=32) F32 inputs maps to one `flambeau_block_q8_1`:
//   d = max(|x|) / 127
//   qs[j] = round(x[j] / d)   (clamped to [-127,127])
//   s = d * sum(qs)            (used by Q4_1/Q5_1/Q8_1 vec_dot)
//
// Grid: one thread block per 32 elements (one Q8_1 block). 32 threads, each
// owns one element → one reduce across the block.
//
// Port basis: candle-hip-kernels quantize_row_q8_1 + ggml reference. No
// gfx906-specific intrinsics here; the reduction is small enough that a
// shared-memory tree is both portable and fast.

#include "block_quant.cuh"

extern "C" __global__ void flambeau_quantize_row_q8_1(
    const float* __restrict__ x,
    flambeau_block_q8_1* __restrict__ y,
    const int n_elems
) {
    const int ib = blockIdx.x;           // block index
    const int t  = threadIdx.x;          // 0..31 lane-in-block
    const int base = ib * QK8_1;

    if (base + t >= n_elems) return;

    const float xi = x[base + t];
    float amax = fabsf(xi);

    // Warp-within-block max over 32 lanes. Works on gfx906 wave64 because
    // only the first 32 lanes participate (threadIdx.x < 32); the other
    // lanes are inactive under `blockDim.x == 32`.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor(amax, offset, 32);
        amax = fmaxf(amax, other);
    }

    const float d  = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    // Quantise this lane's element.
    const int qi = min(127, max(-127, (int) rintf(xi * id)));

    // Per-block scalar sum of qs (needed for the `s` field).
    int sum = qi;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, 32);
    }

    // Thread 0 writes the scales; every thread writes its quant slot.
    if (t == 0) {
        y[ib].d = (fb_fp16_t) d;
        y[ib].s = (fb_fp16_t) (d * (float) sum);
    }
    y[ib].qs[t] = (int8_t) qi;
}
