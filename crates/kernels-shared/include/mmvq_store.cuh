#pragma once
// Final-store helper for MMVQ kernels parameterised on output dtype.
//
// Lets one templated `__device__` body emit two `extern "C"` thunks: one
// writing F32 (existing scratch-then-cast path) and one writing F16
// directly (saves the F32 scratch buffer + the cast kernel launch).
//
// F16 specialisation saturates at ±65504 rather than producing ±inf —
// matches what `__float2half_rn` should do but doesn't, and defends the
// downstream per-head rmsnorm from inf propagation.

#include "block_quant.cuh"

template<typename OutT>
__device__ __forceinline__ void mmvq_store(OutT* y, int row, float acc);

template<>
__device__ __forceinline__ void mmvq_store<float>(float* y, int row, float acc) {
    y[row] = acc;
}

template<>
__device__ __forceinline__ void mmvq_store<fb_fp16_t>(fb_fp16_t* y, int row, float acc) {
    float v = acc;
    if (v > 65504.0f) v = 65504.0f;
    else if (v < -65504.0f) v = -65504.0f;
    y[row] = (fb_fp16_t) v;
}
