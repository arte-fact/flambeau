// scale_f16 — elementwise `y[i] = (fp16)((float)x[i] * scale)`. Mirrors
// `scale_f32` but keeps the I/O in F16. Used by Gemma 4 to apply each
// layer's `layer_output_scale` (per-layer F32 [1] scalar) to the F16
// residual stream at the end of a layer. In-place safe.

#include <hip/hip_runtime.h>

typedef _Float16 fb_fp16_t;

#ifndef SCALE_F16_THREADS
#define SCALE_F16_THREADS 256
#endif

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
