// cast_f16_f32 — pointwise F16 → F32. CUDA mirror of the HIP kernel.
// Launch: 1D, ceil(n/256) blocks × 256 threads.

#include <cuda_runtime.h>
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
