// quantize_f16_q8_1_mmq — F16 activation → BlockQ8_1Mmq (DS4 layout).
// F16 sibling of `quantize_q8_1_mmq` (F32 variant) for the forward path:
// rmsnorm → F16 hidden → this kernel → BlockQ8_1Mmq buffer consumed by
// the 4-warp LDS-tiled MMQ kernel at prefill (m ≥ 128).
// Same block layout and reduction as the F32 version. Only difference is
// reading `__half` and upcasting to float for the sub-block reduce so
// accumulation precision isn't bound by F16.

#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef QK8_1_MMQ_BYTES
#define QK8_1_MMQ_BYTES 144
#endif

extern "C" __global__ void flambeau_quantize_f16_q8_1_mmq(
    const fb_fp16_t* __restrict__ x,    // [total_b, ncols]
    void*            __restrict__ vy,   // [n_big_blocks * total_b] × 144 B
    const int ncols,                    // K
    const int total_b                   // batch rows
) {
    const int b   = blockIdx.x;         // big-block index along K
    const int c   = blockIdx.y;         // output col (batch row)
    const int tid = threadIdx.x;        // 0..127
    const int sub = tid / 32;           // 0..3
    const int lane_in_sub = tid % 32;

    const int ki = b * 128 + tid;
    const float xi = (ki < ncols) ? (float) x[(size_t) c * ncols + ki] : 0.0f;

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

    uint8_t* y_bytes = (uint8_t*) vy
                     + ((size_t) (b * total_b + c)) * QK8_1_MMQ_BYTES;
    ((int8_t*) (y_bytes + 16))[tid] = q;
    if (lane_in_sub == 0) {
        __half2* ds = (__half2*) y_bytes;
        ds[sub] = __floats2half2_rn(d, ssum);
    }
}
