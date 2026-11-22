// quantize_f32_q8_1_few — F32 activation → the few-column B-fragment layout
// (quantize_q8_1_few.cuh).

#include "quantize_q8_1_few.cuh"

// x: [total_b, ncols] F32, ncols % 256 == 0. Grid (pad8(total_b), ncols / 256),
// block 64.
extern "C" __global__ void flambeau_quantize_f32_q8_1_few_d4(
    const float* __restrict__ x, int4* __restrict__ vy, const int ncols, const int total_b) {
    quantize_few_unit<false>(x, vy, ncols, total_b);
}

extern "C" __global__ void flambeau_quantize_f32_q8_1_few_ds4(
    const float* __restrict__ x, int4* __restrict__ vy, const int ncols, const int total_b) {
    quantize_few_unit<true>(x, vy, ncols, total_b);
}
