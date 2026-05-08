// quantize_q8_1_mmq — F32 → block_q8_1_mmq prequantisation for the
// turbo / candle 4-warp MMQ activation path.
// Port of candle's `quantize_q8_1_mmq_q4_0` (mmq_turbo.cu:324), renamed
// without the `_q4_0` suffix because it serves every MMQ weight dtype
// that consumes the DS4 Q8_1 layout (Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 /
// K-quants). One kernel, one layout, shared across all MMQ kernels.
// Input : x [total_b, ncols] F32, row-major (col dim is K)
// Output : vy [n_big_blocks, total_b] block_q8_1_mmq, 144 B each
// One WG per (big_block, col):
// grid = (K / 128, total_b)
// block = 128 threads (2 × wave64)
// Each thread quantises 1 of 128 K-elements for its (big_block, col).
// Sub-block reduce: 4 sub-blocks of 32 elements; each sub-block has its
// own (d, d*sum). Lane `tid/32` picks the sub, `tid%32` is the sub-lane.
// `__shfl_xor(val, m, 32)` does a 32-wide reduce (not full wave64).
// Source parity: candle mmq_turbo.cu:324-360.

#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef QK8_1_MMQ_BYTES
#define QK8_1_MMQ_BYTES 144
#endif

extern "C" __global__ void flambeau_quantize_q8_1_mmq(
    const float* __restrict__ x,    // [total_b, ncols]
    void*        __restrict__ vy,   // [n_big_blocks * total_b] × 144 B
    const int ncols,                // K
    const int total_b                // batch rows
) {
    const int b   = blockIdx.x;     // big-block index along K
    const int c   = blockIdx.y;     // output col (batch row)
    const int tid = threadIdx.x;    // 0..127
    const int sub = tid / 32;       // 0..3 sub-block within the 128-element MMQ block
    const int lane_in_sub = tid % 32;

    const int ki = b * 128 + tid;
    const float xi = (ki < ncols) ? x[(size_t) c * ncols + ki] : 0.0f;

    // Sub-block amax + ssum reduce (32-wide, NOT full wave64).
    float amax = fabsf(xi);
    float ssum = xi;
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, m, 32));
        ssum = ssum + __shfl_xor(ssum, m, 32);
    }

    const float d = amax / 127.0f;
    const int8_t q =
        (amax == 0.0f) ? (int8_t) 0
                       : (int8_t) __float2int_rn(xi / d);

    // Block byte layout: 0..15 = 4 × half2 ds, 16..143 = 128 int8 qs.
    uint8_t* y_bytes = (uint8_t*) vy
                     + ((size_t) (b * total_b + c)) * QK8_1_MMQ_BYTES;

    // qs: one byte per thread.
    ((int8_t*) (y_bytes + 16))[tid] = q;

    // ds: one half2 per sub-block, written by lane_in_sub==0.
    // ds.x = d (scale), ds.y = Σ xi (raw sum). The Q4_1 vec_dot computes
    // sumi * (dm4.x * ds.x) + dm4.y * ds.y
    // where dm4.y = Q4_1 min (m_x). Bias term = m_x * Σ xi which equals
    // m_x * d * Σ q_y — exactly the Q4_1×Q8_1 dequant bias. Storing ssum
    // (not d*ssum) lets the kernel use it verbatim.
    if (lane_in_sub == 0) {
        __half2* ds = (__half2*) y_bytes;
        ds[sub] = __floats2half2_rn(d, ssum);
    }
}
