#pragma once
// F32 activation -> block_q8_1_mmq (DS4 144-B MMQ layout) prequantization, shared
// by both backends. One thread block per (big_block, col); 128 threads, each
// quantises one of 128 K-elements. The 128-element MMQ block holds four 32-element
// sub-blocks, each with its own (d, ssum) pair: `sub = tid/32` picks the sub,
// `lane_in_sub = tid%32`. The reduce is 32-wide (`__shfl_xor(v, m, 32)`), so it is
// warp-size agnostic (works on wave64 and wave32 alike). Byte layout: 16 B header
// = 4 × (d, ssum) fp16 pairs, then 128 int8 qs. ssum is the RAW activation sum
// (Σx, not Σq) — the K-quant vec_dot uses it verbatim for the min/bias term.
// Backend-neutral (rule 5): the only cross-lane op is `__shfl_xor` (sync form on
// CUDA via backend_compat, native on HIP); rounding via `rintf`.
#include "block_quant.cuh"

static __device__ __forceinline__ void quantize_q8_1_mmq_block(
    const float* __restrict__ x,    // [total_b, ncols]
    void* __restrict__ vy,          // [n_big_blocks * total_b] × 144 B
    int ncols,                      // K
    int total_b)                    // batch rows
{
    const int b   = blockIdx.x;     // big-block along K
    const int c   = blockIdx.y;     // output col (batch row)
    const int tid = threadIdx.x;    // 0..127
    const int sub = tid / 32;       // 0..3 sub-block
    const int lane_in_sub = tid % 32;

    const int ki = b * 128 + tid;
    const float xi = (ki < ncols) ? x[(size_t) c * ncols + ki] : 0.0f;

    float amax = fabsf(xi);
    float ssum = xi;
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, m, 32));
        ssum = ssum + __shfl_xor(ssum, m, 32);
    }

    const float d = amax / 127.0f;
    const int8_t q =
        (amax == 0.0f) ? (int8_t) 0 : (int8_t) (int) rintf(xi / d);

    uint8_t* y_bytes = (uint8_t*) vy
                     + (size_t) (b * total_b + c) * sizeof(flambeau_block_q8_1_mmq);

    ((int8_t*) (y_bytes + 16))[tid] = q;

    // Header: 4 × (d, ssum) fp16 pairs. Write as two fb_fp16_t (== the half2 the
    // consumer reads as ds.x = d, ds.y = ssum).
    if (lane_in_sub == 0) {
        fb_fp16_t* ds = (fb_fp16_t*) y_bytes;
        ds[sub * 2 + 0] = (fb_fp16_t) d;
        ds[sub * 2 + 1] = (fb_fp16_t) ssum;
    }
}
