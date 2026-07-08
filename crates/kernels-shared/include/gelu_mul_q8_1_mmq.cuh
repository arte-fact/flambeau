#pragma once
// Fused gemma-4 dense-FFN activation: gelu_tanh(a) * b (F32 inputs) ->
// block_q8_1_mmq (DS4 144-B MMQ layout), shared by both backends. Twin of
// quantize_q8_1_mmq.cuh with the gelu(gate)*up activation folded in, so the
// prefill down-projection reads the MMQ-layout activation without the F16
// gated intermediate write + re-read. GELU is the ggml tanh-approximation
// form (LLM_FFN_GELU parity), identical to gelu_mul_f32, so the fused output
// equals the unfused (gelu_mul_f32 -> quantize_q8_1_mmq) chain bit-for-bit.
// One thread block per (big_block, col); 128 threads, each quantises one of
// 128 K-elements. The 128-element block holds four 32-element sub-blocks,
// each with its own (d, ssum) pair. The reduce is 32-wide, warp-size
// agnostic. ssum is the RAW activation sum (Σx, not Σq) — the MMQ vec_dot
// consumes it verbatim, matching quantize_q8_1_mmq.
// Backend-neutral (rule 5): the only cross-lane op is `__shfl_xor` (sync form
// on CUDA via backend_compat, native on HIP); rounding via `rintf`.
#include "block_quant.cuh"

#define GELU_MUL_MMQ_SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define GELU_MUL_MMQ_COEF_A         0.044715f

static __device__ __forceinline__ void gelu_mul_f32_to_q8_1_mmq_block(
    const float* __restrict__ a,    // [total_b, ncols] gate
    const float* __restrict__ b,    // [total_b, ncols] up
    void* __restrict__ vy,          // [n_big_blocks * total_b] × 144 B
    int ncols,                      // intermediate (K of the down projection)
    int total_b)                    // batch rows (tokens)
{
    const int bb  = blockIdx.x;     // big-block along K
    const int c   = blockIdx.y;     // output col (batch row)
    const int tid = threadIdx.x;    // 0..127
    const int sub = tid / 32;       // 0..3 sub-block
    const int lane_in_sub = tid % 32;

    const int ki = bb * 128 + tid;
    float xi = 0.0f;
    if (ki < ncols) {
        const float av = a[(size_t) c * ncols + ki];
        const float bv = b[(size_t) c * ncols + ki];
        const float tg = GELU_MUL_MMQ_SQRT_2_OVER_PI * av
                       * (1.0f + GELU_MUL_MMQ_COEF_A * av * av);
        xi = 0.5f * av * (1.0f + tanhf(tg)) * bv;
    }

    float amax = fabsf(xi);
    float ssum = xi;
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, m, 32));
        ssum = ssum + __shfl_xor(ssum, m, 32);
    }

    const float d = amax / 127.0f;
    const int8_t q =
        (amax == 0.0f) ? (int8_t) 0 : (int8_t) (int) rintf(xi / d);

    uint8_t* y_bytes = (uint8_t*) vy
                     + (size_t) (bb * total_b + c) * sizeof(flambeau_block_q8_1_mmq);

    ((int8_t*) (y_bytes + 16))[tid] = q;

    if (lane_in_sub == 0) {
        fb_fp16_t* ds = (fb_fp16_t*) y_bytes;
        ds[sub * 2 + 0] = (fb_fp16_t) d;
        ds[sub * 2 + 1] = (fb_fp16_t) ssum;
    }
}
