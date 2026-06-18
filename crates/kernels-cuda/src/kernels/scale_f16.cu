// scale_f16 — elementwise `y[i] = (fp16)((float)x[i] * scale)`. F16 I/O.
// CUDA mirror of the HIP kernel. In-place safe.

#include <cuda_runtime.h>
#include "block_quant.cuh"

extern "C" __global__ void flambeau_scale_f16(
    const fb_fp16_t* __restrict__ x,
    fb_fp16_t* __restrict__ y,
    const int n,
    const float scale
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = (fb_fp16_t) (((float) x[i]) * scale);
}
