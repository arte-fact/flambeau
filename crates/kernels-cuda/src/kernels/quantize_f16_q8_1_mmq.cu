// quantize_f16_q8_1_mmq — F16 activation → BlockQ8_1Mmq (DS4, 144 B). One block
// per (big-block, batch-row), 128 threads = 4 sub-blocks of 32; the amax/sum
// reduce is width-32 __shfl_xor (one warp on wave32, a sub-warp on wave64).

#include "block_quant.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef QK8_1_MMQ_BYTES
#define QK8_1_MMQ_BYTES 144
#endif

extern "C" __global__ void flambeau_quantize_f16_q8_1_mmq(
    const fb_fp16_t* __restrict__ x,    // [total_b, ncols]
    void*            __restrict__ vy,   // [n_big_blocks * total_b] × 144 B
    const int ncols,
    const int total_b
) {
    const int b   = blockIdx.x;
    const int c   = blockIdx.y;
    const int tid = threadIdx.x;
    const int sub = tid / 32;
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
    const int8_t q = (amax == 0.0f) ? (int8_t) 0 : (int8_t) __float2int_rn(xi / d);

    uint8_t* y_bytes = (uint8_t*) vy + ((size_t) (b * total_b + c)) * QK8_1_MMQ_BYTES;
    ((int8_t*) (y_bytes + 16))[tid] = q;
    if (lane_in_sub == 0) {
        __half2* ds = (__half2*) y_bytes;
        ds[sub] = __floats2half2_rn(d, ssum);
    }
}
