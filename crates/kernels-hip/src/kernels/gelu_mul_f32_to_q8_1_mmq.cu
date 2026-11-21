// gelu_mul_f32_to_q8_1_mmq — fused gemma-4 dense-FFN gelu_tanh(a)*b (F32) ->
// block_q8_1_mmq (DS4 144-B MMQ layout). Thin wrapper over the
// backend-neutral kernels-shared device function; same entry-point name + ABI
// as the kernels-cuda twin so the registry/dispatch address it identically.
// Launch: grid = { ncols / 128, total_b }, block = { 128 }, shared = 0.

#include "gelu_mul_q8_1_mmq.cuh"

extern "C" __global__ void flambeau_gelu_mul_f32_to_q8_1_mmq(
    const float* __restrict__ a,
    const float* __restrict__ b,
    void*        __restrict__ vy,
    const int ncols,
    const int total_b
) {
    gelu_mul_f32_to_q8_1_mmq_block(a, b, vy, ncols, total_b);
}
