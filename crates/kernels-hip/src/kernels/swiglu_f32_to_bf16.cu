// swiglu_f32_to_bf16 — fused `y_bf16[i] = (bf16)(silu(a[i]) * b[i])`.
//
// MTP-4-C-5: BF16 analogue of `swiglu_f32_to_f16`. In the BF16 MTP MLP,
// `gate` and `up` come from the F32 mmvq accumulator outputs; the next
// step is the BF16 down matmul, so we fuse swiglu + cast into BF16 to
// avoid the intermediate F32 round-trip + extra launch.

#include "block_quant.cuh"

extern "C" __global__ void flambeau_swiglu_f32_to_bf16(
    const float* __restrict__ a,
    const float* __restrict__ b,
    fb_bf16_t* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float av = a[i];
    const float silu = av / (1.0f + __expf(-av));
    y[i] = fb_f32_to_bf16(silu * b[i]);
}
