// cast_f32_f16 — pointwise F32 → F16.
//
// Decode-path bridge: MMVQ writes its accumulator in F32 (per-block scale
// chain needs full precision); attention / rmsnorm / swiglu all operate
// on F16. Saves adding an F16 accumulator variant of every MMVQ kernel.
//
// Launch: 1D, ceil(n/256) blocks × 256 threads. One element per thread.

#include <hip/hip_runtime.h>
#include "block_quant.cuh"

#ifndef CAST_F32_F16_THREADS
#define CAST_F32_F16_THREADS 256
#endif

extern "C" __global__ void flambeau_cast_f32_f16(
    const float* __restrict__ x,
    fb_fp16_t* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = (fb_fp16_t) x[i];
}
