#pragma once
// Uncontracted float arithmetic: each operation rounds on its own, as the
// host's Rust f32 code does. nvcc fuses `a * b - c` into an FMA by default;
// the `_rn` intrinsics are never fused.

#include <cuda_runtime.h>

static __device__ __forceinline__ float fb_mul_sub_rn(float a, float b, float c) {
    return __fsub_rn(__fmul_rn(a, b), c);
}
