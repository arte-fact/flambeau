// gelu_f32_to_f16 — fused `y_f16[i] = (fp16)(gelu(a[i]) * b[i])`. Mirrors
// the swiglu_f32_to_f16 shape but uses ggml's tanh-approximation GELU
// (parity with llama.cpp's `LLM_FFN_GELU`). Gemma4 dense FFN composes
// `down(GELU(gate) * up)` — this fused kernel replaces the
// `gelu_f32 → mul → cast` chain that would otherwise round-trip F32
// through HBM twice.
// Formula (ggml-cpu/vec.h:986):
//     0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x²)))

#include <hip/hip_runtime.h>

typedef _Float16 fb_fp16_t;

#define GELU_SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define GELU_COEF_A         0.044715f

extern "C" __global__ void flambeau_gelu_f32_to_f16(
    const float* __restrict__ a,
    const float* __restrict__ b,
    fb_fp16_t* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = a[i];
    const float t = GELU_SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x);
    const float gelu = 0.5f * x * (1.0f + tanhf(t));
    float v = gelu * b[i];
    // Saturate at ±F16_MAX — F32-overflow as +inf in F16 poisons the
    // downstream `quantize_f16_q8_1`'s per-block `d` and cascades to
    // all-NaN partial. Matches the saturating clamp in cast_f32_f16
    // and mmvq_store<fb_fp16_t>. (#108)
    if (v > 65504.0f) v = 65504.0f;
    else if (v < -65504.0f) v = -65504.0f;
    y[i] = (fb_fp16_t) v;
}
