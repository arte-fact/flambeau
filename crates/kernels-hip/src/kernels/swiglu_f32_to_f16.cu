// swiglu_f32_to_f16 — 3.b.2 fused `y_f16[i] = (fp16)(silu(a[i]) * b[i])`.
// Replaces the `swiglu_f32 + cast_f32_f16` pair on MoE decode + shared-expert
// paths. The next step after cast is always quantize_row_f16_q8_1, so the
// intermediate F32 buffer is only touched once by swiglu_f32 then once by
// cast_f32_f16 — fusing both in one kernel saves the HBM round-trip and a
// kernel launch per layer per token.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 fb_fp16_t;

extern "C" __global__ void flambeau_swiglu_f32_to_f16(
    const float* __restrict__ a,
    const float* __restrict__ b,
    fb_fp16_t* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float av = a[i];
    const float silu = av / (1.0f + __expf(-av));
    float v = silu * b[i];
    // Saturate at ±F16_MAX — see cast_f32_f16 / gelu_f32_to_f16.
    if (v > 65504.0f) v = 65504.0f;
    else if (v < -65504.0f) v = -65504.0f;
    y[i] = (fb_fp16_t) v;
}
