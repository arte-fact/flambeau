// gelu_mul_f32_to_q8_1 — fused `gelu_tanh(a) * b` (F32 inputs) → Q8_1
// blocks. Gemma-4 dense-FFN twin of swiglu_f32_to_q8_1: folds the
// gelu(gate)*up activation and the row-Q8_1 quantize into one launch,
// removing the F16 gated intermediate write + re-read. GELU is the
// ggml tanh-approximation form (LLM_FFN_GELU parity), computed
// identically to gelu_mul_f32 so the fused output is bit-exact vs the
// unfused (gelu_mul_f32 → quantize_row_q8_1) chain.
// Layout matches quantize_row_q8_1: one thread block per 32-element
// Q8_1 block, 32 threads per block, lane t owns element t.

#include "block_quant.cuh"

#define GELU_SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define GELU_COEF_A         0.044715f

extern "C" __global__ void flambeau_gelu_mul_f32_to_q8_1(
    const float* __restrict__ a,
    const float* __restrict__ b,
    flambeau_block_q8_1* __restrict__ y,
    const int n_elems
) {
    const int ib = blockIdx.x;
    const int t  = threadIdx.x;
    const int base = ib * QK8_1;

    if (base + t >= n_elems) return;

    const float av = a[base + t];
    const float bv = b[base + t];
    const float tg = GELU_SQRT_2_OVER_PI * av * (1.0f + GELU_COEF_A * av * av);
    const float gelu = 0.5f * av * (1.0f + tanhf(tg));
    const float xi   = gelu * bv;

    float amax = fabsf(xi);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor(amax, offset, 32);
        amax = fmaxf(amax, other);
    }

    const float d  = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    const int qi = min(127, max(-127, (int) rintf(xi * id)));

    int sum = qi;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, 32);
    }

    if (t == 0) {
        y[ib].d = (fb_fp16_t) d;
        y[ib].s = (fb_fp16_t) (d * (float) sum);
    }
    y[ib].qs[t] = (int8_t) qi;
}
