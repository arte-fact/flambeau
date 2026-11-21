#pragma once
// Uncontracted float arithmetic: each operation rounds on its own, as the
// host's Rust f32 code does. hipcc contracts `a * b - c` into an FMA by
// default (-ffp-contract=fast-honor-pragmas); the pragma turns that off for
// this body.

#include <hip/hip_runtime.h>

static __device__ __forceinline__ float fb_mul_sub_rn(float a, float b, float c) {
#pragma clang fp contract(off)
    return a * b - c;
}
