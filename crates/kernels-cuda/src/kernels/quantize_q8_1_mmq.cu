// quantize_q8_1_mmq — F32 activation -> block_q8_1_mmq (DS4 144-B MMQ layout).
// CUDA twin of the kernels-hip wrapper; both include the backend-neutral
// kernels-shared device function and expose the same entry-point name + ABI.
// Launch: grid = { K / 128, total_b }, block = { 128 }, shared = 0.

#include "quantize_q8_1_mmq.cuh"

extern "C" __global__ void flambeau_quantize_q8_1_mmq(
    const float* __restrict__ x,
    void*        __restrict__ vy,
    const int ncols,
    const int total_b
) {
    quantize_q8_1_mmq_block(x, vy, ncols, total_b);
}
