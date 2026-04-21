// cast_f16_f32 — pointwise F16 → F32.
//
// Inverse of cast_f32_f16. Used by GDN path to cast attn_norm F16 output
// into F32 for the internal recurrent arithmetic, and by the V1.7.3-b
// swiglu→quantize_q8_1 bridge (replacing the host-roundtrip placeholder).
//
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <hip/hip_runtime.h>
#include "block_quant.cuh"

extern "C" __global__ void flambeau_cast_f16_f32(
    const fb_fp16_t* __restrict__ x,
    float* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = (float) x[i];
}
